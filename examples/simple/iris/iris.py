# imports <<<
import sys
import os

# Add the parent directory of to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.datasets import load_iris

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
    "n_processes_train": 8,
    "n_processes_test": 8,
    "dataset_size": 105,
    "train_batch_size": 35,
    "test_batch_size": 45,
    "test_all": True,
    "beta": 8e-9,
    "lr1": 9.6e-10,
    "lr2": 5.9e-11,
    "layer_1_trainable": True,
    "layer_2_trainable": True,
    "optimizer": "sgd",
    "loss": "mse",
    "update_rule": "ep_sq",
    "quantization_bits": None,
    "quantization_scale": None,
    "num_epoch": 20,
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

# iris dataset <<<
def get_iris_dataset(x_scale=1.0, y_scale=1.0, dataset_size=105):
    iris = load_iris()
    X = iris["data"]
    Y = iris["target"]

    X = np.round(x_scale * preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X), 2)
    Y = y_scale * OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

    p = np.random.permutation(len(Y))
    X_shuffled, Y_shuffled = X[p], Y[p]

    X_train, X_test = X_shuffled[:dataset_size], X_shuffled[dataset_size:]
    Y_train, Y_test = Y_shuffled[:dataset_size], Y_shuffled[dataset_size:]

    return (X_train, Y_train), (X_test, Y_test)
# >>>

# crossbar update rules <<<
def grad_func(free, nudge):
    if ARGS["update_rule"] == "ep_sq":
        return nudge**2 - free**2
    elif ARGS["update_rule"] == "ep_analog":
        return np.abs(nudge)**2.8 - np.abs(free)**2.8
    elif ARGS["update_rule"] == "ep_abs":
        return np.abs(nudge) - np.abs(free)
    elif ARGS["update_rule"] == "ep_mult":
        return 2 * free * (nudge - free)
    elif ARGS["update_rule"] == "ep_quantize":
        return uniform_quantization(free, ARGS["quantization_bits"], -ARGS["quantization_scale"], ARGS["quantization_scale"]) * (nudge - free)

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
    x_scale = 0.6 # values will be between -x_scale and +x_scale.
    y_scale = 1.2 # bit 1 will be mapped to y_scale, bit 0 will be mapped to 0.0.
    (X_train, Y_train), (X_test, Y_test) = get_iris_dataset(x_scale, y_scale, ARGS["dataset_size"])

    # shift output levels
    voltage_shift = 0.6     # amount by which to shift all voltage levels in the circuit.
    output_shift = 0.0      # amount by which to shift the output voltage level
    output_midpoint = 0.6   # mid point value above which the output is considered positive (MSE)

    # construct dataset
    train_dataset = DataSet(
            inputs = {
                "xp": X_train + voltage_shift,
                "xn": -X_train + voltage_shift
            },
            outputs = {
                "c1": Y_train + output_shift
            },
    )

    # test the entire dataset
    if ARGS["test_all"]:
        X_test_all = np.vstack((X_train, X_test))
        Y_test_all = np.vstack((Y_train, Y_test))

        test_dataset = DataSet(
                inputs = {
                    "xp": X_test_all + voltage_shift,
                    "xn": -X_test_all + voltage_shift
                },
                outputs = {
                    "c1": Y_test_all + output_shift
                },
        )
        test_batch_size = X_test_all.shape[0]

    # test only the dataset given by (X_test, Y_test)
    else:
        test_dataset = DataSet(
                inputs = {
                    "xp": X_test + voltage_shift,
                    "xn": -X_test + voltage_shift
                    },
                outputs = {
                    "c1": Y_test + output_shift
                    },
        )
        test_batch_size = ARGS["test_batch_size"]

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, ARGS["train_batch_size"], shuffle=True)
    test_dataloader = BatchGenerator(test_dataset, test_batch_size, shuffle=True)

    # >>>

    # setup: model paramters <<<

    # layer shapes
    input_units = X_train.shape[-1]
    output_units = Y_train.shape[-1]
    hidden_1_units = 10

    # >>>

    # setup: circuit parameters <<<

    #################
    # bias voltages #
    #################

    #  bias voltages
    bias_p = 0 + x_scale / 2 + voltage_shift
    bias_n = 0 - x_scale / 2 + voltage_shift

    # bias voltages of nonlinearity layers
    diode_bias_down = 0 + voltage_shift
    diode_bias_up = 0 - voltage_shift

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
    amp_param = {"shift": voltage_shift, "gain": 4}

    # >>>

    # setup: training parameters <<<
    beta = ARGS["beta"]
    lr1 = ARGS["lr1"]
    lr2 = ARGS["lr2"]
    # >>>

    # loss function setup <<<
    if ARGS["loss"] == "crossentropy":
        activation = "softmax"
        loss_fn = losses.CrossEntropyLoss(beta=beta)
    elif ARGS["loss"] == "mseprob":
        activation = "softmax"
        loss_fn = losses.MSEProb(beta=beta)
    else:
        activation = ""
        loss_fn = losses.MSE(beta=beta, output_midpoint=output_midpoint)
    # >>>

    # build model <<<

    # input layer
    xp = DCVoltageLayer(units=input_units, name="xp")
    xn = DCVoltageLayer(units=input_units, name="xn")
    b1_p = DCVoltageLayer(units=1, name="b1_p", input_voltages=bias_p)
    b1_n = DCVoltageLayer(units=1, name="b1_n", input_voltages=bias_n)
    j1 = ConcatenateLayer(name="j1")([xp, xn, b1_p, b1_n])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=lr1, name="d1", initializer=weight_initialzier, trainable=ARGS["layer_1_trainable"], grad_func=grad_func)(j1)

    # nonlinearity layer
    a1_1 = DiodeLayer(name="act1_1", direction="down", bias_voltage=diode_bias_down, kind="behavioral", param=behaviorial_diode_param, lr=0, trainable=False)(d1)
    a1_2 = DiodeLayer( name="act1_2", direction="up", bias_voltage=diode_bias_up, kind="behavioral", param=behaviorial_diode_param, lr=0, trainable=False)(a1_1)
    g1 = AmplificationLayer(name="amp1", param=amp_param)(a1_2)

    # output layer inputs
    b2_p = DCVoltageLayer(units=1, name="b2_p", input_voltages=bias_p)
    b2_n = DCVoltageLayer(units=1, name="b2_n", input_voltages=bias_n)
    j2 = ConcatenateLayer(name="j2")([g1, b2_p, b2_n])

    # output dense layer
    d_out = DenseLayer(units=2*output_units, lr=lr2, name="d_out", initializer=weight_initialzier, trainable=ARGS["layer_2_trainable"], grad_func=grad_func)(j2)

    # current layer
    c_out = CurrentNudgingLayer(name="c1", fold=True, activation=activation)(d_out)

    # specify model
    model = Model(
            inputs=[xp, xn, b1_p, b2_p, b1_n, b2_n],
            outputs=[c_out],
            n_processes_train = ARGS["n_processes_train"],
            n_processes_test = ARGS["n_processes_test"]
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
        else:
            optimizer.load_state(f"{save_name}_optimizer.pickle")
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
        result = model.evaluate(train_dataset, loss_fn=loss_fn)
        # result = model.predict(train_dataset, loss_fn=loss_fn)
        return result

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
    train(ARGS)
