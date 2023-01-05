#######################################################################
#                               imports                               #
#######################################################################


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.datasets import load_iris

import pickle
import numpy as np
import src.metrics as metrics
import src.optimizers as optimizers
import src.losses as losses
from src.analognn import Model
from src.initializers import Initializers
from src.batchgenerator import BatchGenerator
from src.analoglayers import (
    InputVoltageLayer,
    BiasVoltageLayer,
    DenseLayer,
    DiodeLayer,
    AmplificationLayer,
    CurrentLayer,
    ConcatenateLayer,
)

#######################################################################
#                        fashion mnist dataset                        #
#######################################################################

def load_fashion_dataset():
    # This is a custom dataset that was generated from the original fashion mnist
    # dataset in order to reduce the dimensionality

    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(PATH))

    X_train = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_x_train.npy')
    Y_train = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_y_train.npy')
    X_test = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_x_test.npy')
    Y_test = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_y_test.npy')

    X_train = 2 * X_train
    X_test = 2 * X_test

    return (X_train, Y_train),( X_test, Y_test)


#######################################################################
#                             model setup                             #
#######################################################################


if __name__ == "__main__":

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    (X_train, Y_train),( X_test, Y_test) = load_fashion_dataset()

    # shift output levels
    output_shift = 0
    output_midpoint = 0.5

    # construct dataset
    train_dataset = {
            'inputs' : {'xp':X_train},
            'outputs' : {'c1': Y_train}
        }

    test_dataset = {
            'inputs' : {'xp': X_test },
            'outputs' : {'c1': Y_test}
        }

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, 200)
    test_dataloader = BatchGenerator(test_dataset, 200, shuffle=True)

    #######################################################################
    #                           model parameers                           #
    #######################################################################

    # layer shapes
    input_units = X_train.shape[-1]
    hidden_1_units = 100
    hidden_2_units = 100
    output_units = Y_train.shape[-1]

    #######################################################################
    #                         circuit parameters                          #
    #######################################################################

    ###################
    #  bias voltages  #
    ###################

    bias_p = np.array([0]) + 1
    bias_n = np.array([0]) - 1

    # bias voltages of nonlinearity layers
    diode_bias_down = np.zeros(shape=(hidden_1_units,))
    diode_bias_up = np.zeros(shape=(hidden_1_units,))

    ############
    #  weight  #
    ############

    weight_initialzier = Initializers(
        init_type="glorot", params={"L": 1e-7, "U": 8e-6, "g_max": 2e-5, "g_min": 1e-7}
    )

    ##################
    #  nonlinearity  #
    ##################

    behaviorial_diode_param = {"VTH": 0.1, "RON": 1.0}

    real_diode_param = {
        "path": "./spice_models/diodes.lib",
        "model_name": "1N4148",
    }

    mos_model = {
        "path": "./spice_models/NMOS_VTG.inc",
        "model_name": "NMOS_VTG",
        "length": 50e-9,
        "width": 200e-9,
    }

    ###############
    #  amplifier  #
    ###############

    amp_param = {"shift": 0, "gain": 4}

    ##############
    #  training  #
    ##############

    beta = 1e-6

    #######################################################################
    #                             build model                             #
    #######################################################################

    # input layer
    xp = InputVoltageLayer(units=input_units, name="xp")
    # xn = InputVoltageLayer(units=input_units, name="xn")
    b1_p = BiasVoltageLayer(units=1, name="b1_p", bias_voltage=bias_p)
    b1_n = BiasVoltageLayer(units=1, name="b1_n", bias_voltage=bias_n)
    j1 = ConcatenateLayer(name="j1")([xp, b1_p, b1_n])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=5e-7, name="d1", initializer=weight_initialzier, trainable=True)(j1)
    a1_1 = DiodeLayer(name="act1_1", direction="down", bias_voltage=diode_bias_down, kind="behavioral", param=behaviorial_diode_param)(d1)
    a1_2 = DiodeLayer( name="act1_2", direction="up", bias_voltage=diode_bias_up, kind="behavioral", param=behaviorial_diode_param,)(a1_1)
    g1 = AmplificationLayer(name="amp1", param=amp_param)(d1)

    # layer before output
    b2_p = BiasVoltageLayer(units=1, name="b2_p", bias_voltage=bias_p)
    b2_n = BiasVoltageLayer(units=1, name="b2_n", bias_voltage=bias_n)
    j2 = ConcatenateLayer(name="j2")([g1, b2_p, b2_n])

    # output layer
    d_out = DenseLayer(units=2*output_units, lr=5e-7, name="d_out", initializer=weight_initialzier, trainable=True)(j2)
    c_out = CurrentLayer(name="c1")(d_out)

    model = Model(inputs=[xp, b1_p, b2_p, b1_n, b2_n], outputs=[c_out])

    #######################################################################
    #                                train                                #
    #######################################################################
    save_name = "fashion_test"

    optimizer = optimizers.Adam(model, beta=beta)
    # optimizer.load_state(f"{save_name}_like_model_optimizer.pickle")

    metrics = metrics.Metrics(
        model,
        # save_output_voltages="last",
        # save_power_params="last",
        verbose = True,
        validate_every={"epoch_num": 5},
    )

    loss_fn = losses.MSE(output_midpoint=output_midpoint)

    model.fit(
        train_dataloader=train_dataloader,
        beta=beta,
        epochs=100,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        metrics=metrics,
    )

    predictions = model.evaluate(train_dataset, loss_fn=loss_fn)

    optimizer.save_state(save_name + "_optimizer.pickle")
    model.save_history(save_name + "_history.pickle")
