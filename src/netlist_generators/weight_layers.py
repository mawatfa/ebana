from PySpice.Spice.Netlist import SubCircuit

class DenseResistorMatrix(SubCircuit):
    def __init__(self, name, in_net_names, out_net_names, W):
        SubCircuit.__init__(self, name, *in_net_names, *out_net_names)
        for i in range(len(out_net_names)):
            for j in range(len(in_net_names)):
                self.R(f"{str(j+1)}_{str(i+1)}", in_net_names[j], out_net_names[i], W[j][i])


class DenseResistorMatrixMeasuringCurrent(SubCircuit):
    def __init__(self, name, in_net_names, out_net_names, W):
        SubCircuit.__init__(self, name, *in_net_names, *out_net_names)

        # Create row nodes with zero-valued voltage sources from inputs
        row_nodes = []
        for j, in_net in enumerate(in_net_names, start=1):
            row_node = f"row_{j}"
            self.V(f"r{j}", in_net, row_node, 0)
            row_nodes.append(row_node)

        # Create column nodes with zero-valued voltage sources to outputs
        col_nodes = []
        for i, out_net in enumerate(out_net_names, start=1):
            col_node = f"col_{i}"
            self.V(f"c{i}", col_node, out_net, 0)
            col_nodes.append(col_node)

        # Place resistors between row and column nodes
        for j in range(len(in_net_names)):
            for i in range(len(out_net_names)):
                self.R(f"R_{j+1}_{i+1}", row_nodes[j], col_nodes[i], W[j][i])
