# Complex-valued layers

## Implementation

### Real-Complex Conversion layers

* ConcatenatedRealToCplx
* CplxToConcatenatedReal
* InterleavedRealToCplx
* CplxToInterleavedReal
* AsTypeCplx
* CplxReal
* CplxImag

### Base building blocks

* CplxLinear
* CplxConv1d
* CplxConv2d
* CplxBilinear

### Miscellaneous layers

* CplxDropout
* CplxAvgPool1d
* CplxPhaseShift

### Complex-valued parameter representation

`CplxParameter` is the complex-valued version of `torch.nn.Parameter` that enables seamless access to `.real` and `.imag` parts of a Cplx tensor.

### Subclassing CplxToCplx

The base class for complex-valued layers is `CplxToCplx`. It has not `__init__` and thus can be placed anywhere in the base class list, however it is preferable to keep it as right as possible.

### Promoting real layers

It is possible to promote an existing real-valued module to complex-valued module, which is shared between the real and imaginary parts and acts on them independently. For example the typical use case is to convert an activation to split complex-valued acitvation:

```python
from cplxmodule import cplx
from cplxmodule.nn import CplxToCplx

CplxELU = CplxToCplx[torch.nn.ELU]

z = cplx.Cplx(0 - 1j)
CplxELU()(z)

# Warning this creates a shared layer applied to real-imaginary parts
# This is not equivalent to `nn.CplxLinear`!
CplxSharedLinear = CplxToCplx[torch.nn.Linear]

z = cplx.Cplx(torch.ones(1, 1), - torch.ones(1, 1))
CplxSharedLinear(1, 3, bias=False)(z)
```

## Initialization

Functions in `nn.init` implement various random initialization strategies suitable for complex-valued layers.

## BatchNorm layers

Whitening-based batch normalization layers proposed in [1]_ are provided by `nn.batchnorm`.

* CplxBatchNorm1d
* CplxBatchNorm2d
* CplxBatchNorm3d

## Usage

Basically the module is designed in such a way as to be ready for plugging into the existing `torch.nn` models.

Importing the building blocks.
```python
import torch
import torch.nn

# complex valued tensor class
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

After `RealToCplx` layer the intermediate inputs are `Cplx` objects, which are abstractions for complex valued tensors, represented by real and imaginary parts, and which obey complex arithmetic (currently no support for mixed-type arithmetic like `torch.Tensor +/-* Cplx`).
```python
n_features, n_channels = 16, 4
z = torch.randn(3, n_features*2)

cplx = RealToCplx()(z)
print(cplx)
```

Stacking and constructing purely complex-to-complex pipelines with troch.nn.Sequential:
```python
n_features, n_channels = 16, 4
z = torch.randn(256, n_features*2)

complex_model = CplxSequential(
    CplxLinear(n_features, n_features, bias=True),

    # complex: batch x n_channels x n_features
    CplxConv1d(n_channels, 3 * n_channels, kernel_size=4, stride=1, bias=False),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxModReLU(threshold=0.15),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxToCplx[torch.nn.Flatten](start_dim=-2),

    CplxActivation(torch.tanh),
)
```

Switching to complex-valued layers within a real-to-real model:

```python
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
