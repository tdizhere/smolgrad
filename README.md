# **smolgrad**
an smol version of autograd but more features than karpathy micrograd
smolgrad is a lightweight, pure-Python autograd engine that implements a scalar-valued backpropagation framework from scratch. Heavily inspired by Andrej Karpathy's micrograd, smolgrad goes several steps further by introducing structured deep learning layers, modern optimization algorithms, and multiple activation functions.

It builds a dynamic computational graph on the fly, allowing you to track gradients through standard mathematical operations and train deep neural networks without external tensor libraries.
# "Learn the concepts with same syntax before PyTorch."
## Supported Example Problems

The framework can currently solve:

- XOR classification
- Binary classification using BCE loss
- Two-moons classification
- Circle classification
- Blob classification
- Linear regression
- Polynomial regression
- Sine function approximation

Example:

```python
model = Sequential(
    Linear(2,16),
    Tanh(),
    Linear(16,1),
    Sigmoid()
)

optimizer = Adam(model.parameters(), lr=0.01)
criterion = BCELoss()

```

## Feature Comparison

| Feature | Karpathy Micrograd | Our SmolGrad |
|---|---|---|
| Scalar Autograd Engine | ✅ | ✅ |
| Dynamic Computation Graph | ✅ | ✅ |
| Automatic Backpropagation | ✅ | ✅ |
| Reverse Mode Autodiff | ✅ | ✅ |
| Value Object | ✅ | ✅ Extended |
| Sine (sin) | ❌ | ✅ |
| Cosine (cos) | ❌ | ✅ |
| Absolute Value (abs) | ❌ | ✅ |
| Leaky ReLU | ❌ | ✅ |
| Sigmoid Activation | ❌ | ✅ |
| Custom Activations Support | ❌ | ✅ |
| Gradient Storage | ✅ | ✅ |
| Neuron Abstraction | ✅ | ✅ |
| Linear Layer | ❌ | ✅ |
| Sequential Models | ❌ | ✅ |
| MLP Builder | ✅ | ✅ Extended |
| Parameter Count | ❌ | ✅ |
| Mean Squared Error (MSE) | ✅ | ✅ |
| L1 Loss | ❌ | ✅ |
| Binary Cross Entropy (BCE) | ❌ | ✅ |
| Optimizers | Manual update | ✅ |
| SGD | Manual | ✅ |
| RMSProp | ❌ | ✅ |
| Adam | ❌ | ✅ |
| Dataset Utilities | ❌ | ✅ |
| XOR Dataset | ❌ | ✅ |
| Linear Regression Dataset | ❌ | ✅ |
| Quadratic Dataset | ❌ | ✅(breaks sometime) |
| Sine Dataset | ❌ | ✅ |
| Moons Dataset | ❌ | ✅ |
| Circles Dataset | ❌ | ✅ |
| Blob Dataset | ❌ | ✅ |
| Regression Dataset Generator | ❌ | ✅ |
| Classification Dataset Generator | ❌ | ✅ |
| NumPy/Scikit-learn Dataset Support | ❌ | ✅ |
| Value Conversion Utilities | ❌ | ✅ |
| Model Saving/Loading | ❌ | Planned |
| Computation Graph Visualization | ✅ (example notebook) | Planned |
| Gradient Checking | ❌ | Planned |
