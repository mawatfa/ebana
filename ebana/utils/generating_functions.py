import numpy as np


def generate_names_from_shape(shape: tuple, prefix: str = "P", use_underscore=True):
    """
    Generates a nested array of string names based on the input tuple shape.
    """
    if use_underscore:
        uscore_prefix = "_"
    else:
        uscore_prefix = ""
    array = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        array[idx] = prefix + "".join(f"{uscore_prefix}{i + 1}" for i in idx)
    return array


def numbered_names(shape: tuple, prefix: str):
    """prefix1, prefix2, ... in `shape`, numbered like the netlist generators number the elements."""
    return np.array([f"{prefix}{i + 1}" for i in range(int(np.prod(shape)))], dtype=object).reshape(shape)
