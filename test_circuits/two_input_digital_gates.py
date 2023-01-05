#######################################################################
#                               imports                               #
#######################################################################


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import src.metrics as metrics
import src.optimizers as optimizers
import src.losses as losses
from src.analognn import Model
from src.initializers import Initializers
from src.utils import generate_voltage_values
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
#                             xor dataset                             #
#######################################################################

def make_digital_gates_dataset(bit_0_voltage, bit_1_voltage, count=2):
    X = np.array(generate_voltage_values(values=[bit_0_voltage, bit_1_voltage], count=count))
    OR = np.array([[0],[1],[1],[1]])
    AND = np.array([[0],[0],[0],[1]])
    XOR = np.array([[0],[1],[1],[0]])
    return X, [OR, AND, XOR]


#######################################################################
#                             model setup                             #
#######################################################################


if __name__ == "__main__":

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    X, Y = make_digital_gates_dataset(-2, 2)

    # shift output levels
    output_shift = 0
    output_midpoint = 0.5

    # construct dataset
    train_dataset = {
            'inputs' : {'xp':X, 'xn':-X},
            'outputs' : {'or': Y[0], 'and': Y[1], 'xor': Y[2]}
        }

    test_dataset = {
            'inputs' : {'xp':X, 'xn':-X},
            'outputs' : {'or': Y[0], 'and': Y[1], 'xor': Y[2]}
        }

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, 4)
    test_dataloader = BatchGenerator(test_dataset, 4, shuffle=True)

    #######################################################################
    #                           model parameers                           #
    #######################################################################

    # layer shapes
    input_units = X.shape[-1]
    hidden_1_units = 10
    hidden_2_units = 10
    output_units_1 = Y[0].shape[-1]
    output_units_2 = Y[1].shape[-1]
    output_units_3 = Y[2].shape[-1]

    #######################################################################
    #                         circuit parameters                          #
    #######################################################################

    ###################
    #  bias voltages  #
    ###################

    bias_p = np.array([0]) + 1
    bias_n = np.array([0]) - 1

    # bias voltages of nonlinearity layers
    down_diode_bias = np.zeros(shape=(hidden_1_units, ))
    up_diode_bias = np.zeros(shape=(hidden_1_units, ))

    ############
    #  weight  #
    ############

    weight_initialzier = Initializers(
            init_type="glorot",
            params = {
                "L": 1e-7,
                "U": 8e-6,
                "g_max": 2e-5,
                "g_min": 1e-7
                }
            )

    ##################
    #  nonlinearity  #
    ##################

    behaviorial_diode_param = {
            "VTH": 0.3,
            "RON": 1.0
            }

    real_diode_param = {
            'path': "./spice_models/diodes.lib",
            'model_name': "1N4148",
            }

    mos_model = {
            'path': "./spice_models/NMOS_VTG.inc",
            'model_name': "NMOS_VTG",
            'length': 50e-9,
            'width': 200e-9,
            }

    ###############
    #  amplifier  #
    ###############

    amp_param = {"shift": 0, "gain": 4}

    ##############
    #  training  #
    ##############

    beta = 1e-7

    #######################################################################
    #                             build model                             #
    #######################################################################

    # input layer
    xp  = InputVoltageLayer(units=input_units, name='xp')
    xn = InputVoltageLayer(units=input_units, name='xn')
    b1_p = BiasVoltageLayer(units=1, name='b1_p', bias_voltage=bias_p)
    j1 = ConcatenateLayer(name='j1')([xp, xn, b1_p])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=4e-8, name='d1', initializer=weight_initialzier, trainable=True)(j1)
    a1_1 = DiodeLayer(name='act1_1', direction='down', bias_voltage=down_diode_bias, trainable=False, kind="behavioral", param=behaviorial_diode_param)(d1)
    a1_2 = DiodeLayer(name='act1_2', direction='up', bias_voltage=up_diode_bias, trainable=False, kind="behavioral", param=behaviorial_diode_param)(a1_1)
    g1 = AmplificationLayer(name='amp1', param=amp_param)(d1)

    # layer before output
    b2_p = BiasVoltageLayer(units=1, name='b2_p', bias_voltage=bias_p)
    j2 = ConcatenateLayer(name='j2')([g1, b2_p])

    # output dense layer 1
    d_out1 = DenseLayer(units=2 * output_units_1, lr=4e-8, name='d_out1', initializer=weight_initialzier, trainable=True)(j2)
    c_out1 = CurrentLayer(name='or')(d_out1)

    # output dense layer 2
    d_out2 = DenseLayer(units=2 * output_units_2, lr=4e-8, name='d_out2', initializer=weight_initialzier, trainable=True)(j2)
    c_out2 = CurrentLayer(name='and')(d_out2)

    # output dense layer 3
    d_out3 = DenseLayer(units=2 * output_units_3, lr=4e-8, name='d_out3', initializer=weight_initialzier, trainable=True)(j2)
    c_out3 = CurrentLayer(name='xor')(d_out3)

    model = Model(inputs=[xp, xn, b1_p, b2_p], outputs=[c_out1, c_out2, c_out3])

    #######################################################################
    #                                train                                #
    #######################################################################
    save_name = "digital_gates_test"

    optimizer = optimizers.Adam(model, beta=beta)
    #optimizer.load_state(f"{save_name}_optimizer.pickle")

    metrics = metrics.Metrics(
        model,
        # save_output_voltages="last",
        # save_power_params="last",
        verbose = True,
        validate_every={"epoch_num": 10},
    )

    loss_fn = losses.MSE(output_midpoint=output_midpoint)

    model.fit(
        train_dataloader=train_dataloader,
        beta=beta,
        epochs=50,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        metrics=metrics,
    )

    predictions = model.evaluate(train_dataset, loss_fn=loss_fn)

    optimizer.save_state(save_name + "_optimizer.pickle")
    model.save_history(save_name + "_history.pickle")
