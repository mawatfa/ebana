#######################################################################
#                               imports                               #
#######################################################################
from itertools import product

#######################################################################
#                           useful methods                            #
#######################################################################

def generate_voltage_values(values=[0, 1], count=3):
    positive = list(product(values, repeat=count))
    voltages = [list(positive[i]) for i in range(2**count)]
    return voltages
