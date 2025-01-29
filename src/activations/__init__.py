from .softmax import softmax

def __no_activaton(arr):
    return arr

def set_layer_activation(activation:str):
    if activation == "":
        return __no_activaton
    elif activation == "softmax":
        return softmax
