from pycircuitsim.netlisting import SubCircuit

def DiodeBehavioral(name, out_net_names, direction, bias, param):
    sub = SubCircuit(name, list(out_net_names))
    VTH = param["VTH"]
    RON = param["RON"]
    ROFF = param["ROFF"]

    for i, out in enumerate(out_net_names, start=1):
        node_name = f"N{i}"

        if direction == 'down':
            x_expr = f"(V({out}) - V({node_name}) - {VTH})"
            nodes = (out, node_name)
        else:  # direction == 'up'
            x_expr = f"-(V({out}) + V({node_name}) + {VTH})"
            nodes = (node_name, out)

        on_expr = f"({x_expr}/{RON})"
        off_expr = f"({x_expr}/{ROFF})"
        sub.B(f"d{i}", nodes, current_expression=f"{{ max({off_expr}, {on_expr}) }}")
        sub.V(f"{i}", (node_name, "0"), bias[i-1])
    return sub

def DiodeReal(name, out_net_names, direction, bias, param):
    sub = SubCircuit(name, list(out_net_names))
    model_name = param['model_name']

    for i, out in enumerate(out_net_names, start=1):
        node_name = f"N{i}"
        nodes = (out, node_name) if direction == 'down' else (node_name, out)
        sub.D(f"{i+1}", nodes, model=model_name)
        sub.V(f"{i}", (node_name, "0"), bias[i-1])
    return sub
