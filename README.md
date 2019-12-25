# CplxModule

A lightweight extension for `pytorch.nn` that adds layers and activations,
which respect algebraic operations over the field of complex numbers.

The implementation is based on the ICLR 2018 parer on Deep Complex Networks
[1]_ and borrows ideas from their [implementation](https://github.com/ChihebTrabelsi/deep_complex_networks).


# Installation

Just run to install with `pip` from git
```bash
pip install --upgrade git+https://github.com/ivannz/cplxmodule.git
```
or a developer install (editable) from the root of the local repo
```bash
pip install -e .
```
.


# Example

Basically the module is designed in such a way as to be ready for plugging
into the existing `torch.nn` sequential models.

Importing the building blocks.
```python
import torch
import torch.nn

# complex valeud tensor class
from cplxmodule import cplx

# converters
from cplxmodule.nn import RealToCplx, CplxToReal

# layers of encapsulating other complex valued layers
from cplxmodule.nn.sequential import CplxSequential

# common layers
from cplxmodule.nn.layers import CplxConv1d, CplxLinear

# activation layers
from cplxmodule.nn.activation import CplxModReLU, CplxActivation
```

After `RealToCplx` layer the intermediate inputs are Cplx objects, which are abstractions
for complex valued tensors, represented by real and imaginary parts, and which obey complex
arithmetic (currently no support for mixed-type arithmetic like `torch.Tensor +/-* Cplx`).
```python
n_features, n_channels = 16, 4
z = torch.randn(3, n_features*2)

cplx = RealToCplx()(z)
print(cplx)
```

Stacking and constructing linear pipelines:
```python
n_features, n_channels = 16, 4
z = torch.randn(256, n_features*2)

# gain network works on the modulus of the complex input
modulus_gain = torch.nn.Sequential(
    torch.nn.Linear(n_features, n_channels * n_features),
    torch.nn.Sigmoid(),
)

# purely complex-to-complex sequential container
complex_model = CplxSequential(
    CplxLinear(n_features, n_features, bias=True),

    # complex: batch x n_channels x n_features
    CplxConv1d(n_channels, 3 * n_channels, kernel_size=4, stride=1, bias=False),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxModReLU(threshold=0.15),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxActivation(torch.flatten, start_dim=-2),
)

# branching into complex within a real-to-real model
real_input_model = torch.nn.Sequential(
    # real: batch x (n_features * 2)
    torch.nn.Linear(n_features * 2, n_features * 2),

    # real: batch x (n_features * 2)
    RealToCplx(),

    # complex: batch x n_features
    complex_model,

    # complex: batch x (3 * n_channels * (n_features - (4-1)))
    CplxToReal(),

    # real: batch x ((3 * n_channels * (n_features - (4-1))) * 2)
)

print(real_input_model(z).shape)
# >>> torch.Size([256, 312])
```

# References

.. [1] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian,
       S., Santos, J. F., ... & Pal, C. J. (2017). Deep complex networks.
       arXiv preprint arXiv:1705.09792
