# EBANA: Energy-Based Analog Neural Network Framework

EBANA is a deep learning framework for training analog neural networks with
algorithms that require only local signal access. Initially introduced in
[Frontiers in
Neuroscience](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1114651/full)
, EBANA remains under active development. It currently supports Equilibrium
Propagation but can accommodate any algorithm that needs only local data
access. Inspired by Keras, EBANA seeks to make machine learning and analog
electronics more accessible through an easy-to-use API. Users can experiment
with different architectures and explore tradeoffs in the analog design space.

For more details on Equilibrium Propagation, see [this
paper](https://arxiv.org/abs/1602.05179) .

## Installation

EBANA is written in Python and relies on SPICE simulation for accurate
simulation of circuit dynamics.

**Steps:**

1. **Install PySpice**

```bash
git clone https://github.com/medwatt/PySpice
cd PySpice
pip install .
```

2. **Clone the EBANA framework**

```bash
git clone https://github.com/mawatfa/ebana
```

3. **Install ngspice** (operating system dependent). For example, on Ubuntu:

```bash
sudo apt install ngspice libngspice0-dev
```

## Usage

### Building The Model

Creating and training models in EBANA resembles the Keras functional API. Below
is an example of a model that learns the MNIST dataset.

```python
xp = DCVoltageLayer(units=input_units, name="xp")
rp = ReshapeLayer(name="rxp", shape=(28, 28))(xp)

xn = DCVoltageLayer(units=input_units, name="xn")
rn = ReshapeLayer(name="rxn", shape=(28, 28))(xn)

j1 = StackLayer(name="j1", axis=-1)([rp, rn])

d1 = LocallyConnected2DLayer(
        name="d1",
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding="valid",
        filters=n_filters,
        initializer=weight_initialzier,
        lr=lr1,
        trainable=True,
        grad_func=grad_func,
)(j1)

a1_1 = DiodeLayer(name="act1_1", direction="down", bias_voltage=diode_bias_down, kind="behavioral", param=behaviorial_diode_param, lr=1e-1, trainable=False)(d1)
a1_2 = DiodeLayer(name="act1_2", direction="up", bias_voltage=diode_bias_up, kind="behavioral", param=behaviorial_diode_param, lr=1e-1, trainable=False)(a1_1)
g1 = AmplificationLayer(name="amp1", param=amp_param)(a1_2)

r2 = ReshapeLayer(name="reshape_g1", shape=(-1,))(g1)

d_out = DenseLayer(units=2*output_units, lr=lr2, name="d_out", initializer=weight_initialzier, trainable=True grad_func=grad_func)(r2)

c_out = CurrentNudgingLayer(name="c1", fold=True, activation=activation)(d_out)

model = Model(
        inputs=[xp, xn],
        outputs=[c_out],
        n_processes_train=ARGS['n_processes_train'],
        n_processes_test=ARGS['n_processes_test'],
)
```

- Inputs arrive as voltages through two sources: `xp` and `xn`.

- In analog circuits, resistors (or memristors) store weights and cannot
  be negative. To capture negative connections, a second input with opposite
  polarity (`xn`) is added.

- The inputs are shaped and stacked to form `(28, 28, 2)` before feeding
  them to `LocallyConnected2DLayer`.

- A diode pair provides nonlinearity, and an amplification layer restores
  the dynamic range.

- The output is doubled so each class prediction is given by the voltage
  difference of two nodes.

- Current sources at the output nodes inject gradients during training.

### Training

Training follows Keras-like conventions:

```python
loss_fn = losses.CrossEntropyLoss(beta=beta)
optimizer = optimizers.Adam(model, beta=beta)
# optimizer.load_state("optimizer_state_path")

metrics = Metrics(
    model,
    verbose=ARGS["verbose"],
    save_phase_data=False,
    save_batch_data=False,
    save_injected_currents=False,
    validate_every={"epoch_num": 1},
)

model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    loss_fn=loss_fn,
    optimizer=optimizer,
    test_dataloader=test_dataloader,
    metrics=metrics,
)

# model.evaluate(test_dataset, loss_fn=loss_fn)
# model.predict(test_dataset, loss_fn=loss_fn)
```

Three key components:

- **Loss function**  (e.g., `CrossEntropyLoss`, `MSE`)

- **Optimizer**  (e.g., `Adam`, `SGD`), which can load previous states

- **Metrics**  to track data during training

### Saving History and Model State

After training, save states and logs:

```python
optimizer.save_state(save_name + "_optimizer.pickle")
model.save_history(save_name + "_history.pickle")
```

### Creating Custom Layers

Custom layers follow this structure:

```python
class MyCustomLayer(BaseLayer):
    def __init__(self, name:str, units:int, trainable:bool=False, save_sim_data:bool=True, grad_func=None, weight_upata_func=None):
        super().__init__(
            name,
            units,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_upata_func,
        )

    def _define_shapes(self) -> None:
        """Define the input and output shape."""

    def _define_internal_nets(self) -> None:
        """Net names for the SPICE subcircuit."""

    def _define_external_nets(self) -> None:
        """Net names for connecting with other layers."""

    def _define_internal_branches(self) -> None:
        """Branch names for measuring currents, if needed."""

    def _build(self) -> None:
        """Initialization for training variables, if needed."""

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index) -> None:
        """Generate a subcircuit instance."""

    def get_phase_data_spec(self) -> dict:
        """Dimensions of data to be saved at the end of a phase."""

    def get_batch_data_spec(self) -> dict:
        """Dimensions of data to be saved at the end of a batch."""

    def store_phase_data(self, sim_obj, phase):
        """Store simulation results for a specific phase."""

    def get_phase_data(self):
        """Return stored phase data."""

    def get_batch_data(self):
        """Return stored batch data."""

    def get_variables(self) -> dict:
        """Return the state dict to be saved in the optimizer."""

    def set_variables(self, optimizer_state):
        """Restore internal variables from optimizer state."""

    def get_training_data(self, phase):
        """Return data for gradient computation."""

    def default_grad_func(self, free, nudge):
        """Calculate and return gradients."""

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        """Update internal weight variable using gradient."""
```

See `src/analog_layers` for concrete examples.
