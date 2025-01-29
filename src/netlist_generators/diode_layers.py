from PySpice.Spice.Netlist import SubCircuit

class DiodeBehavioral(SubCircuit):
    def __init__(self, name, out_net_names, direction, bias, param):
        super().__init__(name, *out_net_names)
        VTH = param["VTH"]
        RON = param["RON"]
        ROFF = param["ROFF"]

        for i, out in enumerate(out_net_names, start=1):
            node_name = f"N{i}"

            if direction == 'down':
                x_expr = f"(V({out}) - V({node_name}) - {VTH})"
            else:  # direction == 'up'
                x_expr = f"-(V({out}) + V({node_name}) + {VTH})"

            on_expr = f"({x_expr}/{RON})"
            off_expr = f"({x_expr}/{ROFF})"
            current_expr = f"{{ max({off_expr}, {on_expr}) }}"

            self.B(f"d{i}", out if direction=='down' else node_name,
                   node_name if direction=='down' else out,
                   current_expression=current_expr)
            self.V(i, node_name, self.gnd, bias[i-1])

class DiodeReal(SubCircuit):
    def __init__(self, name, out_net_names, direction, bias, param):
        SubCircuit.__init__(self, name, *out_net_names)
        model_name = param['model_name']

        for i, out in enumerate(out_net_names, start=1):
            node_name = f"N{i}"
            if direction == 'down':
                self.D(i+1, out, node_name, model=model_name)
            else:  # direction == 'up'
                self.D(i+1, node_name, out, model=model_name)
            self.V(i, node_name, self.gnd, bias[i-1])
