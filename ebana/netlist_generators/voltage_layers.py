from pycircuitsim.netlisting import SubCircuit

def DCVoltageSources(name: str, net_names: list, v_arr):
    sub = SubCircuit(name, list(net_names))
    for i in range(len(v_arr)):
        sub.V(f"{i + 1}", (net_names[i], "0"), v_arr[i])
    return sub
