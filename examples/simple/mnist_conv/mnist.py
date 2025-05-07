# imports <<<
import sys
import os

# Add the parent directory sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np

import src.optimizers as optimizers
import src.losses as losses
from src.analognn import Model
from src.initializers import Initializers
from src.dataset_handler import DataSet, BatchGenerator
from src.metrics import Metrics
from src.schedules import CosineDecay

from src.analog_layers import (
    DCVoltageLayer,
    LocallyConnected2DLayer,
    DenseLayer,
    DiodeLayer,
    AmplificationLayer,
    CurrentNudgingLayer,
    StackLayer,
    ReshapeLayer,
)
# >>>

# default arguments <<<
ARGS = {
    "index": -1,
    "evaluate_only": False,
    "write_netlist": False,
    "n_processes_train": 40,
    "n_processes_test": 40,
    "dataset_size": 500,
    "train_batch_size": 40,
    "test_batch_size": 250,
    "n_filters": 2,
    "kernel_size": 5,
    "stride": 3,
    "padding": "valid",
    "beta": 1e-6,
    "lr1": 1e-7,
    "lr2": 1e-7,
    "layer_1_trainable": True,
    "layer_2_trainable": True,
    "bias_voltage_shift": 0.0,
    "optimizer": "adam",
    "loss": "crossentropy",
    "update_rule": "ep_sq",
    "quantization_bits": None,
    "quantization_scale": None,
    "num_epoch": 5,
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

    print("---")
    print(ARGS)
    print("---")
# >>>

# generate save name <<<
def generate_save_name(args):
    quantization_string = f"q{args['quantization_bits']}" if args['quantization_bits'] else ""
    index_string = f"{args['index']}" if args['index'] > -1 else ""

    parameters = [
        index_string,
        "iris",
        args['optimizer'],
        args['loss'],
        str(args['train_batch_size']),
        args['update_rule'],
        quantization_string,
        f"{args['beta']:.1e}",
        f"{args['lr1']:.1e}",
        f"{args['lr2']:.1e}"
    ]
    save_name = "_".join(parameters)
    return save_name
# >>>

# mnist dataset <<<
def load_dataset(n_samples=1000, x_scale=1, y_scale=1, normalize=True):
    # Load dataset
    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(PATH))
    X_train = np.load('../../../mnist_full/x_train.npy')[:n_samples]
    Y_train = np.load('../../../mnist_full/y_train.npy')[:n_samples]
    X_test = np.load('../../../mnist_full/x_test.npy')[:n_samples]
    Y_test = np.load('../../../mnist_full/y_test.npy')[:n_samples]

    # X_train = X_train / 255
    # X_test = X_test / 255

    if normalize:
        mean = 33.318421449829934
        std = 78.56748998339798
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    X_train = x_scale * X_train
    X_test = x_scale * X_test

    Y_train = y_scale * Y_train
    Y_test = y_scale * Y_test

    X_test = X_train
    Y_test = Y_train

    return (X_train, Y_train),( X_test, Y_test)
# >>>

# crossbar update rules <<<
def grad_func(free, nudge):
    if ARGS['update_rule'] == "ep_sq":
        return nudge**2 - free**2
    elif ARGS['update_rule'] == "ep_analog":
        return np.abs(nudge)**2.8 - np.abs(free)**2.8
    elif ARGS['update_rule'] == "ep_abs":
        return np.abs(nudge) - np.abs(free)
    elif ARGS['update_rule'] == "ep_mult":
        return 2 * free * (nudge - free)
    elif ARGS['update_rule'] == "ep_quantize":
        return uniform_quantization(free, ARGS['quantization_bits'], -ARGS['quantization_scale'], ARGS['quantization_scale']) * (nudge - free)

# uniform quantization <<<
def uniform_quantization(arr, num_bits, min_value=None, max_value=None):
    levels = [0.4, 0.55, 0.65, 0.8]
    factor = [-0.6, -0.2, 0.2, 0.6]

    vcm = 0.6
    arr = arr + vcm

    quantized_arr = np.where(arr < levels[0], factor[0],
                    np.where(arr < levels[1], factor[1],
                    np.where(arr > levels[3], factor[3],
                    np.where(arr > levels[2], factor[2], 0))))

    return quantized_arr
# >>>

# >>>

def train(arg_dict):

    update_args_from_user_dict(arg_dict)
    save_name = generate_save_name(ARGS)

    # setup: input/output <<<

    # inputs to the model
    x_scale = 1
    y_scale = 1
    (X_train, Y_train),( X_test, Y_test) = load_dataset(ARGS['dataset_size'], x_scale, y_scale)

    voltage_shift = 0.0    # shift all voltage levels

    # shift output levels
    output_shift = 0
    output_midpoint = y_scale / 2
    output_midpoint = 0.5

    # construct dataset
    train_dataset = DataSet(
            inputs = {"xp": X_train + voltage_shift, "xn": -X_train + voltage_shift},
            outputs = {"c1": Y_train + output_shift},
    )

    test_dataset = DataSet(
            inputs = {"xp": X_test + voltage_shift, "xn": -X_test + voltage_shift},
            outputs = {"c1": Y_test + output_shift},
    )

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, ARGS['train_batch_size'], shuffle=True)
    test_dataloader = BatchGenerator(test_dataset, ARGS['test_batch_size'], shuffle=True)
    # >>>

    # setup: model parameters <<<

    # layer shapes
    input_units = X_train.shape[-1]
    output_units = Y_train.shape[-1]
    # >>>

    # setup: circuit parameters <<<

    ###################
    #  bias voltages  #
    ###################
    bias_p = np.array([0]) + 1.0
    bias_n = np.array([0]) - 1.0

    # bias voltages of nonlinearity layers
    diode_bias_down = 0.5
    diode_bias_up = - 0.1

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
    behaviorial_diode_param = {"VTH": 0.15, "RON": 1.0, "ROFF": 1e20}

    ###############
    #  amplifier  #
    ###############
    amp_param = {"shift": voltage_shift, "gain": 10}

    # >>>

    # setup: training parameters <<<

    ##############
    #  training  #
    ##############
    beta = ARGS['beta']
    lr1 = ARGS['lr1']
    lr2 = ARGS['lr2']

    # >>>

    # loss function setup <<<
    if ARGS['loss'] == "crossentropy":
        activation = "softmax"
        loss_fn = losses.CrossEntropyLoss(beta=beta)
    else:
        activation = ""
        loss_fn = losses.MSEProb(beta=beta, output_midpoint=output_midpoint)
    # >>>

    # build model <<<

    # input positive layer
    xp = DCVoltageLayer(units=input_units, name="xp")
    rp = ReshapeLayer(name="rxp", shape=(28, 28))(xp)

    # input negative layer
    xn = DCVoltageLayer(units=input_units, name="xn")
    rn = ReshapeLayer(name="rxn", shape=(28, 28))(xn)

    # stack positive and negative layers to produce an arry of dim 28x28x2
    j1 = StackLayer(name="j1", axis=-1)([rp, rn])

    # locally connected 2d layer for xp
    d1 = LocallyConnected2DLayer(
            name="d1",
            kernel_size=(ARGS["kernel_size"],ARGS["kernel_size"]),
            stride=(ARGS["stride"],ARGS["stride"]),
            padding=ARGS["padding"],
            filters=ARGS["n_filters"],
            initializer=weight_initialzier,
            lr=lr1,
            trainable=ARGS['layer_1_trainable'],
            grad_func=grad_func,
    )(j1)

    # nonlinearity layer
    a1_1 = DiodeLayer(name="act1_1", direction="down", bias_voltage=diode_bias_down, kind="behavioral", param=behaviorial_diode_param, lr=0.0, trainable=False)(d1)
    a1_2 = DiodeLayer( name="act1_2", direction="up", bias_voltage=diode_bias_up, kind="behavioral", param=behaviorial_diode_param, lr=0.0, trainable=False)(a1_1)
    g1 = AmplificationLayer(name="amp1", param=amp_param)(a1_2)

    # Reshape Layers
    r2 = ReshapeLayer(name="reshape_g1", shape=(-1,))(g1)

    # output dense layer
    d_out = DenseLayer(units=2*output_units, lr=lr2, name="d_out", initializer=weight_initialzier, trainable=ARGS['layer_2_trainable'], grad_func=grad_func)(r2)

    c_out = CurrentNudgingLayer(name="c1", fold=True, activation=activation)(d_out)

    model = Model(
            inputs=[xp, xn],
            outputs=[c_out],
            n_processes_train = ARGS['n_processes_train'],
            n_processes_test = ARGS['n_processes_test'],
            )
    # >>>

    # setup: training <<<

    # optimizer setup <<<
    if ARGS['optimizer'] == "adam":
        optimizer = optimizers.Adam(model, beta=beta)
    else:
        optimizer = optimizers.SGD(model, beta=beta) # don't forget to lower lr and beta for SGD

    if ARGS['load_state']:
        if ARGS['optimizer_path'] != "":
            optimizer.load_state(f"{ARGS['optimizer_path']}")
            print("Done Loading State: ", ARGS['optimizer_path'])
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
    num_epochs = ARGS['num_epoch']

    if ARGS['write_netlist']:
        model.write_netlist(train_dataloader, "cct.sp")
        exit()

    if ARGS['evaluate_only']:
        predictions = model.evaluate(train_dataset, loss_fn=loss_fn)
        return predictions

    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        metrics=metrics,
    )

    if ARGS['save_state']:
        optimizer.save_state(save_name + "_optimizer.pickle")
        model.save_history(save_name + "_history.pickle")


    return model.accumulator.get_test_accuracy(), model.accumulator.get_training_loss()

    # >>>

if __name__ == "__main__":
    train(ARGS)
