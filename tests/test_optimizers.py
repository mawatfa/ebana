import numpy as np
import pytest

from ebana.layers import Parameter
from ebana.optimizers import SGD, Adadelta, Adam, ClipByNorm, SGDMomentum
from ebana.schedules import CosineDecay, PiecewiseConstantDecay

LR = 0.1


class Owner:
    name = "d"


class FakeModel:
    def __init__(self):
        self.p = Parameter(Owner(), "w", np.random.default_rng(0).uniform(1, 2, (3, 2)), device=None, lr=LR,
                           trainable=True)

    def trainable_parameters(self):
        return [self.p]


def sgd(w, gs):
    for g in gs:
        w = w - LR * g
    return w


def momentum(w, gs, m=0.9):
    v = 0
    for g in gs:
        v = m * v + g
        w = w - LR * v
    return w


def adam(w, gs, b1=0.9, b2=0.999, eps=1e-8):
    m = v = 0
    for t, g in enumerate(gs, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - LR * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return w


def adadelta(w, gs, rho=0.95, eps=1e-6):
    eg = ex = 0
    for g in gs:
        eg = rho * eg + (1 - rho) * g * g
        dx = np.sqrt(ex + eps) / np.sqrt(eg + eps) * g
        ex = rho * ex + (1 - rho) * dx * dx
        w = w - LR * dx
    return w


@pytest.mark.parametrize("optimizer, reference", [(SGD, sgd), (SGDMomentum, momentum), (Adam, adam),
                                                  (Adadelta, adadelta)])
def test_matches_the_formula(optimizer, reference):
    model = FakeModel()
    start = model.p.value.copy()
    grads = np.random.default_rng(1).normal(size=(6, 3, 2))
    opt = optimizer(model)
    for g in grads:
        model.p.grad = g
        opt.step()
    np.testing.assert_allclose(model.p.value, reference(start, grads), rtol=1e-12)


def test_state_dict_resumes():
    grads = np.random.default_rng(1).normal(size=(6, 3, 2))
    straight = FakeModel()
    opt = Adam(straight)
    for g in grads:
        straight.p.grad = g
        opt.step()

    first = FakeModel()
    opt = Adam(first)
    for g in grads[:3]:
        first.p.grad = g
        opt.step()
    second = FakeModel()
    second.p.value = first.p.value.copy()
    opt2 = Adam(second)
    opt2.load_state_dict(opt.state_dict())
    for g in grads[3:]:
        second.p.grad = g
        opt2.step()
    np.testing.assert_array_equal(second.p.value, straight.p.value)


def test_clip_by_norm():
    np.testing.assert_allclose(ClipByNorm(1.0)(None, np.array([3.0, -4.0])), [0.6, -0.8])


def test_schedules():
    assert [PiecewiseConstantDecay([1, 3], [3, 2, 1])(s) for s in range(5)] == [3, 3, 2, 2, 1]
    cosine = CosineDecay(1.0, decay_steps=10, alpha=0.1)
    assert cosine(0) == 1.0 and cosine(5) == pytest.approx(0.55) and cosine(20) == pytest.approx(0.1)
