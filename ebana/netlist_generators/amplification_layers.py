from pycircuitsim.netlisting import SubCircuit

def BidirectionalAmplifier(name, in_net_names, out_net_names, param):
    sub = SubCircuit(name, [*in_net_names, *out_net_names])
    shift = param["shift"]
    gain = param["gain"]
    for i in range(len(out_net_names)):
        inn = in_net_names[i]
        out = out_net_names[i]
        sub.B(f"{i+1}", (out, "0"), voltage_expression=f"{{{shift} + {gain} * (V({inn}) - {shift})}}")
        sub.B(f"i{i+1}", (inn, "0"), current_expression=f"{{{-1/gain} * I(B{i+1})}}")
    return sub
