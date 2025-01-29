# imports <<<
import sys
import os
from itertools import product

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np

import src.optimizers as optimizers
import src.losses as losses
from src.analognn import Model
from src.initializers import Initializers
from src.dataset_handler import DataSet, BatchGenerator
from src.metrics import Metrics

from src.analog_layers import (
    DCVoltageLayer,
    DenseLayer,
    DiodeLayer,
    AmplificationLayer,
    CurrentNudgingLayer,
    ConcatenateLayer,
)
# >>>

# default arguments <<<
ARGS = {
    "index": -1,
    "evaluate_only": False,
    "write_netlist": False,
    "n_processes_train": 4,
    "n_processes_test": 4,
    "beta": 2.73e-11,
    "lr1": 5.18e-10,
    "lr2or": 4.54e-11,
    "lr2and": 2.32e-11,
    "lr2xor": 2.38e-11,
    "layer_1_trainable": True,
    "layer_2_trainable": True,
    "optimizer": "sgd",
    "loss": "mse",
    "num_epoch": 10,
    "validate_every_epoch": 1,
    "save_state": True,
    "load_state": False,
    "optimizer_path": "",
    "verbose": True,
}
# >>>

# update args from user dict <<<
def update_args_from_user_dict(arg_dict):
    ARGS.update(arg_dict)
# >>>

# generate save name <<<
def generate_save_name(args):
    index_string = f"{args['index']}" if args['index'] > -1 else ""

    parameters = [
        index_string,
        "gates",
        args['optimizer'],
        args['loss'],
        f"{args['beta']:.1e}",
        f"{args['lr1']:.1e}",
        f"{args['lr2or']:.1e}"
        f"{args['lr2and']:.1e}"
        f"{args['lr2xor']:.1e}"
    ]
    save_name = "_".join(parameters)
    return save_name
# >>>

# generate_datasets <<<
def generate_voltage_values(values=[0, 1], count=3):
    positive = list(product(values, repeat=count))
    voltages = [list(positive[i]) for i in range(2**count)]
    return voltages

def make_digital_gates_dataset(bit_0_voltage, bit_1_voltage, count=2):
    X = np.array(generate_voltage_values(values=[bit_0_voltage, bit_1_voltage], count=count))
    OR = np.array([[0],[1],[1],[1]])
    AND = np.array([[0],[0],[0],[1]])
    XOR = np.array([[0],[1],[1],[0]])
    return X, [OR, AND, XOR]
# >>>

def train(arg_dict):

    update_args_from_user_dict(arg_dict)
    save_name = generate_save_name(ARGS)

    # setup: input/output <<<

    # shift output levels
    output_shift = 0
    output_midpoint = 0.5

    # inputs to the model
    X, Y = make_digital_gates_dataset(-2, 2)

    # construct dataset
    train_dataset = DataSet(
            inputs = {'xp':X, 'xn':-X},
            outputs = {'or': Y[0], 'and': Y[1], 'xor': Y[2]}
    )

    test_dataset = DataSet(
            inputs = {'xp':X, 'xn':-X},
            outputs = {'or': Y[0], 'and': Y[1], 'xor': Y[2]}
    )

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, 4)
    test_dataloader = BatchGenerator(test_dataset, 4, shuffle=True)

    # >>>

    # setup: model parameters <<<

    # layer shapes
    input_units = X.shape[-1]
    hidden_1_units = 10
    hidden_2_units = 10
    output_units_1 = Y[0].shape[-1]
    output_units_2 = Y[1].shape[-1]
    output_units_3 = Y[2].shape[-1]

    # >>>

    # setup: circuit parameters <<<

    ###################
    #  bias voltages  #
    ###################
    bias_p = np.array([0]) + 1
    bias_n = np.array([0]) - 1

    # bias voltages of nonlinearity layers
    diode_bias_down = 0
    diode_bias_up = 0

    ############
    #  weight  #
    ############
    weight_initialzier = Initializers(
            init_type="glorot_uniform",
            params = {
                "init_low": 1e-7,
                "init_high": 1e-5,
                "clip_low": 1e-7,
                "clip_high": 1e-5,
                }
            )

    ##################
    #  nonlinearity  #
    ##################
    behaviorial_diode_param = {"VTH": 0.3, "RON": 1.0, "ROFF": 1e20}

    ###############
    #  amplifier  #
    ###############
    amp_param = {"shift": 0, "gain": 4}

    # >>>

    # setup: training parameters <<<
    beta = ARGS['beta']
    lr1 = ARGS['lr1']
    lr2or = ARGS['lr2or']
    lr2and = ARGS['lr2and']
    lr2xor = ARGS['lr2xor']
    # >>>

    # loss function setup <<<
    if ARGS['loss'] == "crossentropy":
        activation = "softmax"
        loss_fn = losses.CrossEntropyLoss(beta=beta)
    else:
        activation = ""
        loss_fn = losses.MSE(beta=beta, output_midpoint=output_midpoint)
    # >>>

    # build model <<<

    # input layer
    xp  = DCVoltageLayer(units=input_units, name='xp')
    xn = DCVoltageLayer(units=input_units, name='xn')
    b1_p = DCVoltageLayer(units=1, name="b1_p", input_voltages=bias_p)
    j1 = ConcatenateLayer(name='j1')([xp, xn, b1_p])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=lr1, name='d1', initializer=weight_initialzier, trainable=True)(j1)
    a1_1 = DiodeLayer(name="act1_1", direction="down", bias_voltage=diode_bias_down, kind="behavioral", param=behaviorial_diode_param, lr=0.0, trainable=False)(d1)
    a1_2 = DiodeLayer( name="act1_2", direction="up", bias_voltage=diode_bias_up, kind="behavioral", param=behaviorial_diode_param, lr=0.0, trainable=False)(a1_1)
    g1 = AmplificationLayer(name='amp1', param=amp_param)(a1_2)

    # layer before output
    b2_p = DCVoltageLayer(units=1, name='b2_p', input_voltages=bias_p)
    j2 = ConcatenateLayer(name='j2')([g1, b2_p])

    # output dense layer 1
    d_out1 = DenseLayer(units=2 * output_units_1, lr=lr2or, name='d_out1', initializer=weight_initialzier, trainable=True)(j2)
    c_out1 = CurrentNudgingLayer(name='or', fold=True, activation=activation)(d_out1)

    # output dense layer 2
    d_out2 = DenseLayer(units=2 * output_units_2, lr=lr2and, name='d_out2', initializer=weight_initialzier, trainable=True)(j2)
    c_out2 = CurrentNudgingLayer(name='and', fold=True, activation=activation)(d_out2)

    # output dense layer 3
    d_out3 = DenseLayer(units=2 * output_units_3, lr=lr2xor, name='d_out3', initializer=weight_initialzier, trainable=True)(j2)
    c_out3 = CurrentNudgingLayer(name='xor', fold=True, activation=activation)(d_out3)

    # specify model
    model = Model(
            inputs=[xp, xn, b1_p, b2_p],
            outputs=[c_out1, c_out2, c_out3],
            n_processes_train = ARGS['n_processes_train'],
            n_processes_test = ARGS['n_processes_test']
            )
    # >>>

    # training setup <<<

    # optimizer setup <<<
    if ARGS["optimizer"] == "adam":
        optimizer = optimizers.Adam(model, beta=beta)
    else:
        optimizer = optimizers.SGD(model, beta=beta) # don't forget to lower lr and beta for SGD

    if ARGS["load_state"]:
        if ARGS["optimizer_path"] != "":
            optimizer.load_state(f"{ARGS['optimizer_path']}")
            print("Done Loading State: ", ARGS["optimizer_path"])
        else:
            optimizer.load_state(f"{save_name}_optimizer.pickle")
            print("Done Loading State: ", f"{save_name}_optimizer.pickle")
    # >>>

    metrics = Metrics(
        model,
        verbose=ARGS["verbose"],
        save_phase_data=False,
        save_batch_data=False,
        save_injected_currents=False,
        validate_every={"epoch_num": ARGS["validate_every_epoch"]},
    )

    # >>>

    # train <<<
    num_epochs = ARGS["num_epoch"]

    if ARGS["write_netlist"]:
        model.write_netlist(train_dataloader, "cct.sp")
        exit()

    if ARGS["evaluate_only"]:
        results = model.evaluate(train_dataset, loss_fn=loss_fn)
        # predictions = model.predict(train_dataset, loss_fn=loss_fn)
        return results

    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        metrics=metrics,
    )

    if ARGS["save_state"]:
        optimizer.save_state(save_name + "_optimizer.pickle")
        model.save_history(save_name + "_history.pickle")


    return model.accumulator.get_test_accuracy(), model.accumulator.get_training_loss()

    # >>>

if __name__ == "__main__":
    result = train(ARGS)
    # print(result)
