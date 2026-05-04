<p align="center">
  <img src="figures/logo.svg" width="300"/>
</p>

# LearnTorch

A deep learning library written in C++ with Python bindings via [pybind11](https://github.com/pybind/pybind11).

## Requirements

- Python 3.10+
- C++17 compiler (g++ or clang++)
- pybind11

## Installation

### Option 1 — pip install (recommended)

```bash
pip install .
```

### Option 2 — build in place

```bash
pip install pybind11
python setup.py build_ext --inplace
```

Then add the repo root to your Python path before importing.

## Quick start

XOR problem with a small MLP:

```python
import learntorch as lt

model = lt.Serial([
    lt.Linear(2, 4),
    lt.ReLU(),
    lt.Linear(4, 1),
    lt.Sigmoid()
])

X = lt.Tensor([4, 2], [0.0, 0.0,
                        0.0, 1.0,
                        1.0, 0.0,
                        1.0, 1.0])
Y = lt.Tensor([4, 1], [0.0, 1.0, 1.0, 0.0])

optimizer = lt.Adam(model.parameters(), lr=0.01)
trainer   = lt.Trainer(model, optimizer, lt.MSELoss())
trainer.fit(X, Y, epochs=500, batch_size=4, print_every=100)
```

## Layers

| Layer | Signature |
|---|---|
| `Linear` | `Linear(in_features, out_features)` |
| `Conv2D` | `Conv2D(in_channels, out_channels, kernel_size, stride, padding)` |
| `ReLU` | `ReLU()` |
| `Sigmoid` | `Sigmoid()` |
| `Tanh` | `Tanh()` |
| `Softmax` | `Softmax()` |
| `Flatten` | `Flatten()` |
| `Identity` | `Identity()` |
| `Embedding` | `Embedding(vocab_size, vector_size)` |
| `LayerNorm` | `LayerNorm(embed_dim, epsilon=1e-5)` |
| `SelfAttention` | `SelfAttention(dim)` |
| `Serial` | `Serial([block, ...])` |
| `Parallel` | `Parallel([block, ...])` |
| `Join` | `Join(mode=JoinMode.SUM)` — modes: `SUM`, `CONCAT` |

## Optimizers

| Optimizer | Signature |
|---|---|
| `SGD` | `SGD(params, lr)` |
| `Adam` | `Adam(params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)` |

## Loss functions

- `MSELoss()`
- `CrossEntropy()`

## Trainer

```python
trainer = lt.Trainer(model, optimizer, loss)
trainer.fit(x_train, y_train, epochs, batch_size, print_every=1)
```

## Tensor operations

```python
lt.matmul(a, b)
lt.sum(tensor, dim=0)
lt.max(tensor, dim=0)
lt.argmax(tensor, dim=0)
lt.transpose(tensor)
lt.relu(tensor)
lt.sigmoid(tensor)
lt.tanh(tensor)
lt.exp(tensor)
lt.log(tensor)
lt.sqrt(tensor)

lt.Tensor.zeros(shape)
lt.Tensor.ones(shape)
lt.Tensor.random(shape, min_val=-1.0, max_val=1.0)
```

## Backends

```python
lt.set_backend(lt.CPU)            # default
lt.set_backend(lt.CPU_OPTIMIZED)  # omp
lt.set_backend(lt.GEMM_OPTIMIZED) # AVX2
```

## Save / load weights

```python
model.save_weights("my_weights")
model.load_weights("my_weights")
```
