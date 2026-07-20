# Extending EBANA

Most experiments I have run with EBANA needed something the framework didn't
have: a memristor instead of a resistor, a different update rule, a way to model
how the devices are programmed. The framework is split so that each of these is
a small class of your own, passed to the parts that already exist.

## Devices

A device is an element with a trainable value. It knows how to add itself to the
netlist, what to measure after a simulation, and how two measurements give the
gradient. This is the default weight device, a resistor, without the rounding of
the real one:

```python
from ebana.devices import Device, voltage_across

class Resistor(Device):
    def stamp(self, block, id, a, b, value):
        return block.resistor(id, a, b, 1 / value)

    def measure(self, sim, elements):
        return voltage_across(sim, elements)

    def gradient(self, a, b):
        return (b**2 - a**2) / 2
```

`gradient(a, b)` receives the measurements of the two states the algorithm
compares: free and nudged for EP, the negative and positive nudges for the
symmetric version. Pass your device to any layer that takes one, `Dense("d1",
10, device=Memristor(...))`, and the rest follows.

The `block` passed to `stamp` can add most SPICE elements: sources, controlled
sources, diodes, MOSFETs, `.model` cards, includes, and Verilog-A devices, which
are compiled with OpenVAF when the netlist is loaded. It returns a handle that
the simulator later uses to change or read the element.

Elements without a trainable value, like the diodes of the `Diode` layer, are
`ElementModel`s: they only implement `stamp`.

## Update rules

If you only want to change how the gradient is computed from the measurements,
you don't need a new device. Every layer with weights takes a `grad_func`:

```python
class PowerRule:
    """For a device that responds to its voltage like |V|^p."""

    def __init__(self, p):
        self.p = p

    def __call__(self, a, b):
        return (np.abs(b)**self.p - np.abs(a)**self.p) / self.p

Dense("d1", 10, grad_func=PowerRule(2.8), ...)
```

It is a class and not a lambda because it is sent to the worker processes.
`update_func` does the same for the update itself.

## Layers

A layer implements `build`, which creates its parameters and returns its output
nets, and `stamp`, which adds its elements to the netlist. The netlist is built
once per batch and reused for every sample, so a layer whose elements depend on
the data (like `Input`) also implements `set_sample`, which alters them in the
loaded circuit. The built-in layers in `ebana/layers` are short and are the best
reference.

## Learning algorithms

EP is one `LearningAlgorithm`. An algorithm gets the circuit of one sample and
decides which simulations to run and how their measurements become gradients:

```python
from ebana.algorithms import LearningAlgorithm, StepResult

class MyAlgorithm(LearningAlgorithm):
    def train_step(self, circuit, targets, loss):
        circuit.solve("free")
        outputs, free = circuit.outputs(), circuit.measure()
        ...
        return StepResult(grads=..., losses=..., outputs=outputs)
```

`circuit.nudge(forces)` injects currents into the outputs, and
`circuit.set(parameter, value)` changes a parameter in the loaded circuit
without touching its value, which is what a perturbation-based method needs. The
framework undoes both before the next sample.

## Callbacks and probes

A callback is a class with `on_batch_end`, `on_epoch_end` and so on. If it keeps
state that matters for resuming, like a random generator, it returns it from
`state_dict` and takes it back in `load_state_dict`, and checkpoints then
include it.

A probe is anything a `Recorder` can measure: a callable `probe(model, sim)`
returning an array. `Voltages` and `Measurement` cover the common cases.

## Checking gradients

When you write a device or an algorithm, check it against the simulator:

```python
from ebana import gradient_check

result = gradient_check(model, dataset, loss, algorithm, elements=5)
result["d1/weight"]["estimate"] / result["d1/weight"]["numerical"]
```

The ratio should be 1, or `1/gain^2` for layers before an amplifier. Anything
else means the gradient rule doesn't match the circuit.
