<p align="center">
  <img src="figures/logo.svg" width="300"/>
</p>

# LearnTorch

A deep learning framework written from scratch in C++ with Python bindings via [pybind11](https://github.com/pybind/pybind11). It includes a reverse-mode automatic differentiation engine, a modular layer system, multiple math backends (including AVX2/GEMM), and a training loop — all without external dependencies beyond pybind11.

## Requirements

- Python 3.10+
- C++17 compiler (g++ or clang++)
- pybind11

## Installation

### Option 1 — install from GitHub (recommended)

```bash
pip install git+https://github.com/JoseLu2121/LearnTorch.git
```

### Option 2 — pip install (local clone)

```bash
pip install .
```

### Option 3 — editable build (development)

```bash
pip install -e . --no-build-isolation
```

## Quick start

XOR problem with a small MLP:

```python
import learntorch as lt

lt.set_backend(lt.BackendType.GEMM_OPTIMIZED)

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

optimizer = lt.Adam(model.parameters(), 0.01, 0.9, 0.999, 1e-8)
trainer   = lt.Trainer(model, optimizer, lt.MSELoss())
trainer.fit(X, Y, X, Y, epochs=500, batch_size=4, print_every=100)
```

---

## Backends

LearnTorch ships three interchangeable math backends. Set one before building your model:

```python
lt.set_backend(lt.BackendType.CPU)            # scalar, no optimization (default)
lt.set_backend(lt.BackendType.CPU_OPTIMIZED)  # vectorized element-wise ops
lt.set_backend(lt.BackendType.GEMM_OPTIMIZED) # tiled GEMM + AVX2/FMA (recommended)
```

The `GEMM_OPTIMIZED` backend implements the algorithm described in *Anatomy of High-Performance Matrix Multiplication* (Goto & Van de Geijn, 2008) with cache-level tiling and AVX2 FMA intrinsics, achieving near-PyTorch throughput on matrix multiplications.

---

## Layers

### Basic layers

| Layer | Signature | Description |
|---|---|---|
| `Linear` | `Linear(in_features, out_features)` | Fully connected layer. `Y = X @ W.T + B` |
| `ReLU` | `ReLU()` | Rectified linear unit. `Y = max(0, X)` |
| `Sigmoid` | `Sigmoid()` | `Y = 1 / (1 + exp(-X))` |
| `Tanh` | `Tanh()` | `Y = tanh(X)` |
| `Softmax` | `Softmax()` | Numerically stable softmax over last dimension |
| `Flatten` | `Flatten()` | Collapses all dimensions except batch into one |
| `Identity` | `Identity()` | Passes input through unchanged (useful in residual branches) |

### Convolutional layers

| Layer | Signature | Description |
|---|---|---|
| `Conv2D` | `Conv2D(in_channels, out_channels, kernel_size, stride, padding)` | 2D convolution via im2col + GEMM |

### Sequence / NLP layers

| Layer | Signature | Description |
|---|---|---|
| `Embedding` | `Embedding(vocab_size, vector_size)` | Lookup table mapping token indices to dense vectors |
| `LayerNorm` | `LayerNorm(embed_dim, epsilon=1e-5)` | Normalizes over the last dimension |
| `SelfAttention` | `SelfAttention(dim)` | Scaled dot-product self-attention with causal mask |

### Structural blocks

| Block | Signature | Description |
|---|---|---|
| `Serial` | `Serial([block, ...])` | Runs blocks sequentially, passing output of each as input to the next |
| `Parallel` | `Parallel([block, ...])` | Runs all blocks on the same input simultaneously, returning a list of outputs |
| `Join` | `Join(mode=JoinMode.SUM)` | Merges a list of tensors. Modes: `JoinMode.SUM`, `JoinMode.CONCAT` |

`Serial`, `Parallel` and `Join` together allow expressing any network topology — including residual connections — declaratively without custom forward methods:

```python
# Residual block
block = lt.Serial([
    lt.Parallel([lt.Identity(), lt.Serial([lt.LayerNorm(d), lt.SelfAttention(d)])]),
    lt.Join(lt.JoinMode.SUM),
])
```

---

## Optimizers

| Optimizer | Signature |
|---|---|
| `SGD` | `SGD(params, lr)` |
| `Adam` | `Adam(params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8)` |

```python
optimizer = lt.Adam(model.parameters(), 0.001, 0.9, 0.999, 1e-8)
optimizer.zero_grad()
optimizer.step()
```

---

## Loss functions

| Loss | Description |
|---|---|
| `MSELoss()` | Mean squared error. Suitable for regression and binary classification with sigmoid output |
| `CrossEntropy()` | Cross-entropy loss. Applies softmax internally, expects one-hot targets |

```python
criterion = lt.CrossEntropy()
loss = criterion.forward(prediction, target)
loss.backward()
```

---

## Trainer

```python
trainer = lt.Trainer(model, optimizer, criterion)
trainer.fit(
    x_train, y_train,   # training data
    x_val,   y_val,     # validation data
    epochs     = 100,
    batch_size = 32,
    print_every = 10    # print loss/accuracy every N epochs
)
```

---

## Tensor operations

```python
# Arithmetic (supports broadcasting)
c = a + b
c = a - b
c = a * b
c = a / b
c = a @ b          # matrix multiply (calls lt.matmul internally)

# Functions
lt.matmul(a, b, transpose_b=False)
lt.relu(tensor)
lt.sigmoid(tensor)
lt.tanh(tensor)
lt.exp(tensor)
lt.log(tensor)
lt.sqrt(tensor)
lt.sum(tensor, dim=0)
lt.max(tensor, dim=0)
lt.argmax(tensor, dim=0)
lt.transpose(tensor)   # returns a transposed view (no copy)

# Construction
lt.Tensor(shape, flat_data)
lt.Tensor.zeros(shape)
lt.Tensor.ones(shape)
lt.Tensor.random(shape, min_val=-1.0, max_val=1.0)

# Inspection
tensor.shape     # list of ints
tensor.strides   # list of ints
tensor.data()    # flat list of floats
tensor.grad      # gradient tensor (after backward)
tensor.item()    # scalar value

# Autograd
tensor.backward()
```

---

## Save / load weights

```python
model.save_weights("my_weights")   # writes binary file
model.load_weights("my_weights")   # restores all parameters in order
```

---

## Transformer example (decoder-only GPT-style)

```python
import learntorch as lt

lt.set_backend(lt.BackendType.GEMM_OPTIMIZED)

embed_dim  = 128
vocab_size = 53

attention_branch = lt.Serial([lt.SelfAttention(embed_dim)])
ff_branch = lt.Serial([
    lt.Linear(embed_dim, 4 * embed_dim),
    lt.ReLU(),
    lt.Linear(4 * embed_dim, embed_dim),
])

model = lt.Serial([
    lt.Embedding(vocab_size, embed_dim),
    lt.Parallel([lt.Identity(), lt.Serial([lt.LayerNorm(embed_dim), attention_branch])]),
    lt.Join(lt.JoinMode.SUM),
    lt.Parallel([lt.Identity(), lt.Serial([lt.LayerNorm(embed_dim), ff_branch])]),
    lt.Join(lt.JoinMode.SUM),
    lt.Linear(embed_dim, vocab_size),
])

optimizer = lt.Adam(model.parameters(), 0.001, 0.9, 0.999, 1e-8)
trainer   = lt.Trainer(model, optimizer, lt.CrossEntropy())
trainer.fit(x_train, y_train, x_val, y_val, epochs=500, batch_size=64, print_every=10)
```
