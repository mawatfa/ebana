# Building Models

A model is a circuit. Every layer adds its own elements to one flat netlist, and
layers are connected with the functional style (similar to Keras): `y =
Layer(...)(x)`. The model is everything connected to the outputs you give it.

```python
from ebana import Model
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.constraints import Clip, MaxRelativeChange
from ebana.layers import Amplifier, Bias, Concatenate, CurrentOutput, Dense, Differential, Diode, Input

init = GlorotUniform(low=1e-7, high=1e-5, seed=0)
window = [MaxRelativeChange(0.5), Clip(1e-7, 1e-5)]

xp = Input("xp", 4)
xn = Input("xn", 4)
bias = Bias("bias", [0.9, 0.3])
rows = Concatenate("rows")([xp, xn, bias])

d1 = Dense("d1", 10, initializer=init, constraint=window, lr=1e-8)(rows)
clamp_hi = Diode("clamp_hi", "down", model=IdealDiode(vth=0.15), bias_voltage=0.6)(d1)
clamp_lo = Diode("clamp_lo", "up", model=IdealDiode(vth=0.15), bias_voltage=0.6)(clamp_hi)
amp = Amplifier("amp", gain=4, device=BidirectionalAmplifier(vcm=0.6))(clamp_lo)

d2 = Dense("d2", 6, initializer=init, lr=1e-9)(amp)
out = CurrentOutput("out", readout=Differential())(d2)

model = Model(outputs=[out])
```

This is the Iris network from the examples, with the whole circuit working
around a common-mode voltage of 0.6 V.

## Inputs and biases

`Input` is a set of voltage sources set by the dataset input with the same name.
Conductances can't be negative, so the usual trick is to apply every input twice
with opposite signs (`xp = x`, `xn = -x`) and let the crossbar pick whichever it
needs. `Bias` is a set of fixed sources; they give the next crossbar a constant
row, like the bias of a neuron.

Both take a `device`, which is a `VoltageSource` by default. Pass
`CurrentSource()` to drive the nets with currents instead.

## Weights

`Dense(name, units)` is a crossbar from every input net to every output net,
with one element per weight. The weight is the conductance, so the initializer
and constraint work in siemens. `Clip` keeps the weights inside the range the
device can actually reach, and `MaxRelativeChange` limits how far a single
update can move them.

`LocallyConnected2D` is the convolution-like version: every output position has
its own small crossbar over its patch of the input, since a physical array has
no weight sharing. `AveragePooling2D` ties each output to its window through
equal conductances.

## Nonlinearities

A `Diode` layer puts one diode per net to its own bias voltage. With `"down"` it
conducts when the net rises above the bias, with `"up"` when it falls below, so
a pair of them clamps the signal. The diode itself is an element model:
`IdealDiode` is piecewise linear, and `SpiceDiode` uses a real `.model` card,
either from parameters or from a library file.

An `Amplifier` gives every net its own `BidirectionalAmplifier` stage. It
amplifies forwards and passes the current back divided by the gain, which is
what lets the nudge travel from the outputs back to the hidden layers.

## Outputs

`CurrentOutput` reads its input nets and, during training, injects the nudge
currents into them. With `Differential()` each output is the difference of two
nets, which gives signed outputs from positive conductances; `Direct()` reads
the nets as they are.

## Wiring

`Reshape`, `Concatenate` and `Stack` only rearrange nets. `Join` ties the nets
of several layers together, and `Ammeter` puts a 0 V source in series with
every net so you can read the currents through it.

## Looking at the circuit

`model.write_netlist(dataset, "circuit.sp")` writes the netlist for one sample.
It is worth doing once for every new model, mostly to check that the circuit is
what you meant. The model also gives you its layers and parameters by name:

```python
model.layer("d1")
model.parameter("d1/weight").value
model.save_parameters("parameters.npz")
```
