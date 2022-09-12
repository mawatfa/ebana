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
#                            iris dataset                             #
#######################################################################

from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def get_iris_dataset(scale=4, split_size=0.7):
    iris = load_iris()
    X = iris['data']
    Y = iris['target']

    X = np.round(scale * preprocessing.scale(X), 2)
    Y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))

    p = np.random.permutation(len(Y))
    X_shuffled, Y_shuffled = X[p], Y[p]

    n_train = int(split_size * len(Y_shuffled))

    X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
    Y_train, Y_test = Y_shuffled[:n_train], Y_shuffled[n_train:]

    return (X_train, Y_train), (X_test, Y_test)

#######################################################################
#                             model setup                             #
#######################################################################

if __name__ == "__main__":

    #######################################################################
    #                       input/output parameters                       #
    #######################################################################

    # inputs to the model
    (X_train, Y_train),( X_test, Y_test) = get_iris_dataset(scale=4)

    # construct dataset
    train_dataset = {
            'inputs' : {'xp':X_train, 'xn':-X_train},
            'outputs' : {'c1':Y_train}
        }

    test_dataset = {
            'inputs' : {'xp':X_test, 'xn':-X_test},
            'outputs' : {'c1':Y_test}
        }

    # dataloader for training in batches
    train_dataloader = BatchGenerator(train_dataset, 105)
    test_dataloader = BatchGenerator(test_dataset, 45, shuffle=True)

    #######################################################################
    #                           model parameers                           #
    #######################################################################

    # layer shapes
    input_units = X_train.shape[-1]
    hidden_1_units = 10
    output_units = Y_train.shape[-1]

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
    b2 = BiasVoltageLayer(units=1, name='b3', bias_voltage=bias2, trainable=False)
    j2 = ConcatenateLayer(name='j3')([g1, b2])

    # output layer
    d_out = DenseLayer(units=2 * output_units, lr=0.0001, name='d3', init_type='glorot', trainable=True)(j2)
    c_out = CurrentLayer(name='c1')(d_out)

    model = Model(inputs=[xp, xn, b1, b2], outputs=[c_out])

    #######################################################################
    #                                train                                #
    #######################################################################

    optimizer = optimizers.Adam(model)
    #optimizer.load_state('iris_optimizer.pickle')

    loss_fn = losses.CrossEntropyLoss()
    model.fit(train_dataloader, beta=0.01, epochs=5,
       loss_fn=loss_fn,
       optimizer=optimizer,
       test_dataloader=test_dataloader,
       validate_every = {'epoch_num': 10},
    )


    #######################################################################
    #                           evaluate model                            #
    #######################################################################

    #predictions = model.evaluate(test_dataset, loss_fn=loss_fn)
    predictions = model.evaluate(train_dataset, loss_fn=loss_fn)

    optimizer.save_state('iris_optimizer.pickle')
    model.save_history('iris_history.pickle')