import numpy as np

from ebana import Dataset, DataLoader


def ids(batch):
    return batch.inputs["x"][:, 0].astype(int).tolist()


def data(n=10):
    x = np.arange(n, dtype=float).reshape(-1, 1)
    return Dataset({"x": x}, {"y": 2 * x})


def test_batches():
    assert [ids(b) for _, b in DataLoader(data(), 3).batches()] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for indices, batch in DataLoader(data(), 3, shuffle=True, seed=0).batches(4):
        assert indices.tolist() == ids(batch)
        np.testing.assert_array_equal(batch.outputs["y"], 2 * batch.inputs["x"])


def test_shuffle_depends_on_seed_and_epoch_only():
    loader = DataLoader(data(), 5, shuffle=True, seed=0)
    epochs = [[ids(b) for _, b in loader.batches(e)] for e in range(3)]
    assert epochs[0] != epochs[1]
    assert [ids(b) for _, b in DataLoader(data(), 5, shuffle=True, seed=0).batches(2)] == epochs[2]


def test_split():
    first, second = data().split(0.7, seed=0)
    assert len(first) == 7 and sorted(ids(first) + ids(second)) == list(range(10))
