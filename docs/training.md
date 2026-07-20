# Training

Training with Equilibrium Propagation runs two simulations per sample. The free
phase lets the circuit settle with the outputs untouched. The nudged phase
injects a small current into every output, proportional to the gradient of the
loss, and lets it settle again. The difference between what each device measured
in the two phases gives its gradient. EBANA runs these simulations in ngspice,
in parallel over the samples of a batch.

## Data

Inputs and outputs are given by layer name:

```python
from ebana import Dataset, DataLoader

train = Dataset(inputs={"xp": x, "xn": -x}, outputs={"out": y})
loader = DataLoader(train, batch_size=35, shuffle=True, seed=0)
```

Outputs are always `(samples, units)`. The shuffle depends only on the seed and
the epoch, which is what makes resuming exact (see below).

## Fitting

```python
from ebana.algorithms import EquilibriumPropagation
from ebana.losses import CrossEntropy
from ebana.metrics import CategoricalAccuracy
from ebana.optimizers import SGD

history = model.fit(
    train_data=loader,
    validation_data=test,
    epochs=40,
    optimizer=SGD(model),
    algorithm=EquilibriumPropagation(beta=2e-9),
    loss=CrossEntropy(),
    metrics=[CategoricalAccuracy()],
    workers=8,
)
```

`beta` is the nudge strength. It is a conductance, because the injected current
is `-beta * dL/dy`: small enough to keep the nudge a perturbation, large enough
to stay above the simulator's precision. `SymmetricEquilibriumPropagation`
nudges in both directions and removes the first-order error of the estimate,
at the cost of one more simulation per sample.

`workers` is the number of processes simulating samples in parallel. Each one
keeps its own ngspice and loads the netlist once per batch. Because the model is
sent to the workers, functions you pass to it (a `grad_func`, for example) must
be defined at module level, and scripts need an `if __name__ == "__main__":`
guard.

Progress is logged to the `"ebana"` logger, one line per epoch:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

## Losses, metrics and optimizers

The losses are `MSE`, `CrossEntropy` (softmax, on the raw outputs) and
`BinaryCrossEntropy`, and the metrics `BinaryAccuracy` and
`CategoricalAccuracy`. With several outputs, give a dict per output name.

The optimizers are `SGD`, `SGDMomentum`, `Adam` and `Adadelta`. Gradients can be
transformed before the update, for example with `Adam(model,
transforms=[ClipByNorm(1.0)])`.

Learning rates belong to the layers, since each crossbar usually needs its own.
A learning rate can also be a schedule from `ebana.schedules`, such as
`CosineDecay`.

One thing that surprises people: the amplifier passes the nudge back divided by
its gain, so the gradients of every layer before an amplifier come out scaled
by `1/gain^2`. The direction is right, so it just means those layers need a
larger learning rate. `ebana.gradient_check` compares the gradients with finite
differences through the simulator if you want to see it.

## History

`fit` returns a `History` with one record per batch and one per epoch, keyed
like `loss/out`, `categorical_accuracy/out` and, for validation, `val_loss/out`:

```python
history.epoch_values("val_categorical_accuracy/out")
history.save("history.npz")
```

`model.evaluate(dataset, loss, metrics)` and `model.predict(dataset)` use the
free phase only.

## Callbacks

Callbacks run at the end of every batch and epoch. `EarlyStopping` stops when a
value stops improving, optionally going back to the best parameters. `Recorder`
keeps what you ask for about every sample of the batches it records: voltages,
device measurements in every phase, outputs, forces.

```python
from ebana.callbacks import EarlyStopping, Recorder
from ebana.probes import Measurement, Voltages

recorder = Recorder({"hidden": Voltages("amp"), "drops": Measurement("d1/weight")}, every=10)
model.fit(..., callbacks=[recorder, EarlyStopping("val_loss", patience=5)])
recorder.get("drops/nudge")    # (records, batch, 10, 10)
```

A callback that changes parameter values at the end of a batch is how I model
programming the devices: quantized levels, write noise and so on. The next batch
is simulated with whatever values it leaves.

## Checkpoints

Long runs get interrupted. With `checkpoint="run.ckpt"` the whole run is saved
after every epoch (and every `checkpoint_every` batches), and
`resume="run.ckpt"` continues from exactly where it stopped: same batches, same
optimizer state, same history, same callback state. Give `epochs` a larger value
to train further.
