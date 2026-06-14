from pycircuitsim.netlisting import SubCircuit

def DCCurrentSources(name: str, net_names: list, c_arr):
    sub = SubCircuit(name, list(net_names))
    for i in range(len(c_arr)):
        sub.I(f"{i + 1}", (net_names[i], "0"), c_arr[i])
    return sub
