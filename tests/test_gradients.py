import numpy as np
import pytest

from ebana import Dataset, Model, gradient_check
from ebana.algorithms import EquilibriumPropagation, SymmetricEquilibriumPropagation
from ebana.devices import CurrentSource, IdealDiode, Resistor, SpiceDiode
from ebana.initializers import Constant, RandomUniform
from ebana.layers import Amplifier, Bias, Concatenate, CurrentOutput, Dense, Diode, Input
from ebana.losses import MSE

EP = SymmetricEquilibriumPropagation(1e-9)
DATA = Dataset({"x": np.array([[0.9, -0.9, 0.5]])}, {"out": np.array([[0.2, -0.1]])})


def build(gain=1.0, up=0.0, down=0.0, diode=IdealDiode(vth=0.1, ron=10)):
    x = Input("x", 3)
    v = Bias("v", [0.7])
    i = Bias("i", [2e-4], device=CurrentSource())
    rows = Concatenate("rows")([x, v, i])
    h = Dense("h", 4, initializer=RandomUniform(1e-4, 1e-3, seed=1), device=Resistor(decimals=None))(rows)
    clamp_down = Diode("down", "down", model=diode, bias_voltage=down)(h)
    clamp_up = Diode("up", "up", model=diode, bias_voltage=up)(clamp_down)
    amp = Amplifier("amp", gain=gain)(clamp_up)
    o = Dense("o", 2, initializer=RandomUniform(1e-4, 1e-3, seed=2), device=Resistor(decimals=None))(amp)
    out = CurrentOutput("out")(o)
    return Model(outputs=[out])


def ratios(model):
    names = [p.full_name for p in model.parameters() if p.full_name != "amp/gain"]
    results = gradient_check(model, DATA, MSE(), EP, parameters=names, step=1e-5)
    scale = max(np.abs(r["numerical"]).max() for r in results.values())
    out = {}
    for name, r in results.items():
        on = np.abs(r["numerical"]) > 1e-8 * scale
        out[name] = r["estimate"][on] / r["numerical"][on]
    return out


@pytest.mark.parametrize("up, down, diode", [(0.0, 0.0, IdealDiode(vth=0.1, ron=10)),
                                             (0.3, 0.5, IdealDiode(vth=0.1, ron=10)),
                                             (0.0, -0.2, SpiceDiode(IS=1e-14, N=1))])
def test_gradients_match_finite_differences(up, down, diode):
    r = ratios(build(up=up, down=down, diode=diode))
    assert len(r["down/bias"]) + len(r["up/bias"]) > 0
    for name, ratio in r.items():
        np.testing.assert_allclose(ratio, 1, rtol=1e-4, err_msg=name)


def test_amplifier_scales_the_gradients_before_it():
    for name, ratio in ratios(build(gain=2.0)).items():
        np.testing.assert_allclose(ratio, 1 if name == "o/weight" else 0.25, rtol=1e-4, err_msg=name)


def test_one_sided_ep_converges():
    x = Input("x", 2)
    d = Dense("d", 1, initializer=Constant(1e-3), device=Resistor(decimals=None))(x)
    d.weight.value[:] = [[1e-3], [3e-3]]
    out = CurrentOutput("out")(d)
    model = Model(outputs=[out])
    data = Dataset({"x": np.array([[1.0, -0.5]])}, {"out": np.array([[0.2]])})
    for beta, rtol in [(1e-4, 3e-2), (1e-6, 3e-4)]:
        r = gradient_check(model, data, MSE(), EquilibriumPropagation(beta), step=1e-5)["d/weight"]
        np.testing.assert_allclose(r["estimate"], r["numerical"], rtol=rtol)


def test_gain_needs_a_grad_func():
    with pytest.raises(NotImplementedError):
        gradient_check(build(), DATA, MSE(), EP, parameters=["amp/gain"])
