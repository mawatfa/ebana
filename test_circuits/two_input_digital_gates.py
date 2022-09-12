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
#                        digital gates dataset                        #
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
    a1_1 = DiodeLayer(name='act1_1', direction='down', bias_voltage=a1_bias, trainable=False)(d1)
    a1_2 = DiodeLayer(name='act1_2', direction='up', bias_voltage=-a1_bias, trainable=False)(a1_1)
    g1 = AmplificationLayer(gain=4, name='amp1')(a1_2)

    # layer before output
    b2 = BiasVoltageLayer(units=1, name='b2', bias_voltage=bias2, trainable=False)
    j2 = ConcatenateLayer(name='j3')([g1, b2])

    # output dense layer 1
    d_out1 = DenseLayer(units=2 * output_units_1, lr=0.0001, name='d_out1', init_type='glorot', trainable=True)(j2)
    c_out1 = CurrentLayer(name='or')(d_out1)

    # output dense layer 2
    d_out2 = DenseLayer(units=2 * output_units_2, lr=0.0001, name='d_out2', init_type='glorot', trainable=True)(j2)
    c_out2 = CurrentLayer(name='and')(d_out2)

    # output dense layer 3
    d_out3 = DenseLayer(units=2 * output_units_3, lr=0.0001, name='d_out3', init_type='glorot', trainable=True)(j2)
    c_out3 = CurrentLayer(name='xor')(d_out3)

    model = Model(inputs=[xp, xn, b1, b2], outputs=[c_out1, c_out2, c_out3])

    #######################################################################
    #                                train                                #
    #######################################################################

    optimizer = optimizers.Adam(model)
    #optimizer.load_state('digital_gates.pickle')

    loss_fn = losses.MSE()
    model.fit(train_dataloader, beta=0.001, epochs=150,
            loss_fn=loss_fn,
            optimizer=optimizer,
            test_dataloader=test_dataloader,
            validate_every = {'epoch_num': 10})

    optimizer.save_state('digital_gates_optimizer.pickle')
    model.save_history('digital_gates_history.pickle')
