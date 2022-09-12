#######################################################################
#                               imports                               #
#######################################################################

import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PATH))

from src.analoglayers import InputVoltageLayer, BiasVoltageLayer, DenseLayer, DiodeLayer, AmplificationLayer, CurrentLayer, ConcatenateLayer
from src.batchgenerator import BatchGenerator
from src.analognn import Model
import src.losses as losses
import src.optimizers as optimizers

import numpy as np
import pickle

#######################################################################
#                        fashion mnist dataset                        #
#######################################################################

def load_fashion_dataset():
    # This is a custom dataset that was generated from the original fashion mnist
    # dataset in order to reduce the dimensionality
    X_train = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_x_train.npy')
    Y_train = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_y_train.npy')
    X_test = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_x_test.npy')
    Y_test = np.load(PATH + '/mnist_fashion/fashion_batchnorm_layer_y_test.npy')

    X_train = 2 * X_train
    X_test = 2 * X_test

    return (X_train, Y_train),( X_test, Y_test)

if __name__ == "__main__":

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    (X_train, Y_train),( X_test, Y_test) = load_fashion_dataset()

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

    # bias voltages
    bias1 = np.zeros(shape=(1, )) + 1
    bias2 = np.zeros(shape=(1, )) + 1
    bias3 = np.zeros(shape=(1, )) + 1

    # bias voltages of nonlinearity layers
    a1_bias = np.zeros(shape=(hidden_1_units, )) + 0.0
    a2_bias = np.zeros(shape=(hidden_2_units, )) - 0.0

    #######################################################################
    #                             build model                             #
    #######################################################################


    xp = InputVoltageLayer(units=input_units, name='xp', trainable=False)
    #xn = InputVoltageLayer(units=input_units, name='xn', trainable=True)
    b1 = BiasVoltageLayer(units=1, name='b1', bias_voltage=bias1, trainable=False)
    j1 = ConcatenateLayer(name='j1')([xp, b1])

    # hidden dense layer 1
    d1 = DenseLayer(units=hidden_1_units, lr=0.01, name='d1', init_type='glorot', trainable=True)(j1)
    a1_1 = DiodeLayer(name='act1_1', direction='down', bias_voltage=a1_bias, trainable=False)(d1)
    a1_2 = DiodeLayer(name='act1_2', direction='up', bias_voltage=-a1_bias, trainable=False)(a1_1)
    g1 = AmplificationLayer(gain=4, name='amp1')(a1_2)

    # layer before output
    b2 = BiasVoltageLayer(units=1, name='b3', bias_voltage=bias2, trainable=False)
    j2 = ConcatenateLayer(name='j3')([g1, b2])

    # output layer
    d_out = DenseLayer(units=2 * output_units, lr=0.0001, name='d3', init_type='glorot', trainable=True)(j2)
    c_out = CurrentLayer(name='c1')(d_out)

    model = Model(inputs=[xp, b1, b2], outputs=[c_out])

    #######################################################################
    #                                train                                #
    #######################################################################

    optimizer = optimizers.Adam(model)
    #optimizer.load_state('fashion_optimizer.pickle')

    loss_fn = losses.CrossEntropyLoss()
    model.fit(train_dataloader, beta=0.01, epochs=1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        test_dataloader=test_dataloader,
        validate_every = {'batch_num': 5},
        )

    #######################################################################
    #                           evaluate model                            #
    #######################################################################

    optimizer.save_state('fashion_optimizer.pickle')
    model.save_history('fashion_history.pickle')

    predictions = model.evaluate(train_dataset, loss_fn=loss_fn)
    predictions = model.evaluate(test_dataset, loss_fn=loss_fn)
