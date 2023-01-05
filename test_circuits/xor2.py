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

def make_xor_gate_dataset(bit_0_voltage, bit_1_voltage, count=2):
    X = np.array(generate_voltage_values(values=[bit_0_voltage, bit_1_voltage], count=count))
    XOR = np.array([[0],[1],[1],[0]])
    return X, XOR


#######################################################################
#                             model setup                             #
#######################################################################


if __name__ == "__main__":

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    X, Y = make_xor_gate_dataset(-0.5, 0.5)

    # shift output levels
    output_shift = -0.5
    output_midpoint = 0.0

    # construct dataset
    train_dataset = {
            #'inputs' : {'xp':X_train},
            'inputs' : {'xp':X , 'xn':-X},
            'outputs' : {'xor':Y + output_shift}
        }

    test_dataset = {
            #'inputs' : {'xp':X_test},
            'inputs' : {'xp':X, 'xn':-X},
            'outputs' : {'xor':Y + output_shift}
        }

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, 4)
    test_dataloader = BatchGenerator(test_dataset, 4, shuffle=True)

    #######################################################################
    #                           model parameers                           #
    #######################################################################

    # layer shapes
    input_units = X.shape[-1]
    output_units = Y.shape[-1]
    hidden_1_units = 2

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
    b1_n = BiasVoltageLayer(units=1, name='b1_n', bias_voltage=bias_n)
    j1 = ConcatenateLayer(name='j1')([xp, xn, b1_p, b1_n])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=4e-8, name='d1', initializer=weight_initialzier, trainable=True)(j1)
    a1_1 = DiodeLayer(name='act1_1', direction='down', bias_voltage=down_diode_bias, trainable=False, kind="behavioral", param=behaviorial_diode_param)(d1)
    a1_2 = DiodeLayer(name='act1_2', direction='up', bias_voltage=up_diode_bias, trainable=False, kind="behavioral", param=behaviorial_diode_param)(a1_1)
    g1 = AmplificationLayer(name='amp1', param=amp_param)(d1)

    # layer before last
    b2_p = BiasVoltageLayer(units=1, name='b2_p', bias_voltage=bias_p)
    b2_n = BiasVoltageLayer(units=1, name='b2_n', bias_voltage=bias_n)
    j2 = ConcatenateLayer(name='j2')([g1, b2_p, b2_n])

    # output layer
    d_out = DenseLayer(units=2 * output_units, lr=4e-8, name='d_out', initializer=weight_initialzier, trainable=True)(j2)
    c_out = CurrentLayer(name='xor')(d_out)

    model = Model(inputs=[xp, xn, b1_p, b2_p, b1_n, b2_n], outputs=[c_out])

    #######################################################################
    #                                train                                #
    #######################################################################
    save_name = "xor2_test"

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
        epochs=100,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        metrics=metrics,
    )

    predictions = model.evaluate(train_dataset, loss_fn=loss_fn)

    optimizer.save_state(save_name + "_optimizer.pickle")
    model.save_history(save_name + "_history.pickle")
