import numpy as np
import pytest

from ebana import Model
from ebana.devices import IdealDiode, SpiceDiode
from ebana.initializers import Constant
from ebana.layers import Ammeter, Amplifier, Bias, CurrentOutput, Dense, Diode, Input, Join, Reshape
from ebana.simulation import Simulator


def solve(outputs, values):
    model = Model(outputs=outputs)
    sim = Simulator()
    sim.load(model.build_netlist({name: np.array([v]) for name, v in values.items()}))
    sim.solve()
    return sim


def test_dense_drops_and_currents():
    x = Input("x", 2)
    meter = Ammeter("meter")(x)
    d = Dense("d", 1, initializer=Constant(1e-3))(meter)
    sim = solve([CurrentOutput("out")(d)], {"x": [1.0, 3.0]})
    np.testing.assert_allclose(sim.voltages(d.output), [2.0])
    np.testing.assert_allclose(d.weight.measure(sim), [[-1.0], [1.0]])
    np.testing.assert_allclose(meter.currents(sim), [-1e-3, 1e-3], rtol=1e-6)
    np.testing.assert_allclose(x.measure(sim), [1e-3, -1e-3], rtol=1e-6)


def test_diodes_and_amplifier_keep_the_shape():
    x = Input("x", 4)
    a = Diode("a", direction="down", model=IdealDiode(vth=0.3, ron=1.0))(Reshape("r", (2, 2))(x))
    g = Amplifier("g", gain=4)(a)
    sim = solve([g], {"x": [1.0, 0.0, 0.5, 2.0]})
    np.testing.assert_allclose(np.abs(a.bias.measure(sim)), [[0.7, 0.0], [0.2, 1.7]], atol=1e-9)
    np.testing.assert_allclose(sim.voltages(g.output), [[4.0, 0.0], [2.0, 8.0]])


def test_spice_diode_from_a_library(tmp_path):
    (tmp_path / "d.lib").write_text(".model dmod D(IS=1e-14)\n")
    a = Diode("a", direction="down", model=SpiceDiode("dmod", path=str(tmp_path / "d.lib")))(Input("x", 2))
    current = np.abs(a.bias.measure(solve([a], {"x": [0.6, 0.0]})))
    assert current[0] > 1e-5 and current[1] < 1e-9


def test_join():
    x = Input("x", 1)
    a = Dense("a", 1, initializer=Constant(1e-3))(x)
    b = Dense("b", 1, initializer=Constant(1e-3))(x)
    c = Dense("c", 1, initializer=Constant(1e-3))(Bias("zero", [0.0]))
    sim = solve([Join("j")([a, b, c])], {"x": [1.0]})
    assert sim.voltages(a.output) == pytest.approx([2 / 3])


def test_layers_are_ordered_parents_first():
    x = Input("x", 2)
    out = CurrentOutput("out")(Dense("d", 1, initializer=Constant(1e-3))(x))
    Diode("side", direction="down")(x)
    names = [layer.name for layer in Model(outputs=[out]).layers]
    assert set(names) == {"x", "d", "out", "side"}
    assert names.index("x") < names.index("d") < names.index("out")
