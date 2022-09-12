#######################################################################
#                               imports                               #
#######################################################################

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analoglayers import InputVoltageLayer, BiasVoltageLayer, DenseLayer, DiodeLayer, AmplificationLayer, CurrentLayer, ConcatenateLayer
from src.batchgenerator import BatchGenerator
from src.analognn import Model
from src.utils import generate_voltage_values
import src.losses as losses
import src.optimizers as optimizers

import numpy as np
import pickle

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
    #                         circuit parameters                          #
    #######################################################################

    mos_model = {
            'path': "./spice_models/transistors.lib",
            'model_name': "NMOS_VTH",
            'length': 100e-9,
            'width': 150e-9,
            }

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    X, Y = make_xor_gate_dataset(-2, 2)

    # construct dataset
    train_dataset = {
            'inputs' : {'xp':X, 'xn':-X},
            'outputs' : {'xor': Y}
        }

    test_dataset = {
            'inputs' : {'xp':X, 'xn':-X},
            'outputs' : {'xor': Y}
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
    output_units = Y.shape[-1]

    # bias voltages
    bias1 = np.zeros(shape=(1, )) + 1
    bias2 = np.zeros(shape=(1, )) + 1

    # bias voltages of nonlinearity layers
    a1_bias = np.zeros(shape=(hidden_1_units, )) + 0.3

    #######################################################################
    #                             build model                             #
    #######################################################################

    # input layer
    xp = InputVoltageLayer(units=input_units, name='xp', trainable=False)
    xn = InputVoltageLayer(units=input_units, name='xn', trainable=False)
    b1 = BiasVoltageLayer(units=1, name='b1', bias_voltage=bias1, trainable=False)
    j1 = ConcatenateLayer(name='j1')([xp, xn, b1])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=0.0001, name='d1', init_type='glorot', trainable=True)(j1)
    a1_1 = DiodeLayer(name='act1_1', direction='down', bias_voltage=a1_bias, trainable=False, use_mos=True, model=mos_model)(d1)
    a1_2 = DiodeLayer(name='act1_2', direction='up', bias_voltage=-a1_bias, trainable=False, use_mos=True, model=mos_model)(a1_1)
    g1 = AmplificationLayer(gain=4, name='amp1')(a1_2)

    # layer before output
    b2 = BiasVoltageLayer(units=1, name='b2', bias_voltage=bias2, trainable=False)
    j2 = ConcatenateLayer(name='j2')([g1, b2])

    # output layer
    d_out = DenseLayer(units=2 * output_units, lr=0.0001, name='d_out', init_type='glorot', trainable=True)(j2)
    c_out = CurrentLayer(name='xor')(d_out)

    model = Model(inputs=[xp, xn, b1, b2], outputs=[c_out])

    #######################################################################
    #                                train                                #
    #######################################################################

    optimizer = optimizers.Adam(model)
    #optimizer.load_state('digital_gates.pickle')

    loss_fn = losses.MSE()
    model.fit(train_dataloader, beta=0.001, epochs=100,
            loss_fn=loss_fn,
            optimizer=optimizer,
            test_dataloader=test_dataloader,
            validate_every = {'epoch_num': 20})

    optimizer.save_state('xor_gate_optimizer.pickle')
    model.save_history('xor_gate_history.pickle')
