# Complex-valued layers

## Implementation

### Real-Complex Conversion layers

* ConcatenatedRealToCplx, CplxToConcatenatedReal
* InterleavedRealToCplx (RealToCplx), CplxToInterleavedReal (CplxToReal)
* AsTypeCplx

### Basic building blocks

* CplxReal, CplxImag
* CplxIdentity, CplxLinear, CplxBilinear
* CplxConv1d, CplxConv2d, CplxConv3d
* CplxSequential

### Complex activation layers

* CplxModulus, CplxAngle
* CplxModReLU, CplxAdaptiveModReLU

### Complex batch normalization

Batch normalization layers, based on 2d vector whitening proposed in _[1]_, are provided by `nn.modules.batchnorm`.

* CplxBatchNorm1d, CplxBatchNorm2d, CplxBatchNorm3d

### Miscellaneous layers

* CplxDropout
* CplxPhaseShift

### Complex-valued parameter representation

`CplxParameter` is the complex-valued version of `torch.nn.Parameter` that enables seamless access to `.real` and `.imag` parts of a Cplx tensor.

### Subclassing CplxToCplx

The base class for complex-valued layers is `CplxToCplx`. It does not have `__init__` and thus can be placed anywhere in the base class list, however it is preferable to keep it as right as possible, but preceding `torch.nn.Module`.

### Promoting real layers

It is possible to promote an existing real-valued module to complex-valued module, which is *shared* between the real and imaginary parts and acts on them independently, i.e. the same layer is applied twice. For example, the typical use case is to convert a real-valued activation to split complex-valued acitvation:

```python
import torch

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

It is also possible to promote a unary and not-inplace real-valued function from `torch.` to a complex-valued split activation of tranformation, i.e.
```python
CplxSplitSin = CplxToCplx[torch.sin]

CplxSplitSin()(z)
```

## Initialization

Functions in `nn.init` implement various random initialization strategies suitable for complex-valued layers, that were researched in _[1]_.

## Usage

Basically the module is designed in such a way as to be ready for plugging into the existing `torch.nn` models.

Importing the building blocks.
```python
import torch

# complex valued tensor class
from cplxmodule import cplx

# converters
from cplxmodule.nn import RealToCplx, CplxToReal

# layers of encapsulating other complex valued layers
from cplxmodule.nn import CplxSequential

# common layers
from cplxmodule.nn import CplxConv1d, CplxLinear

# activation layers
from cplxmodule.nn import CplxModReLU
```

After `RealToCplx` layer the intermediate inputs are `Cplx` objects, which are abstractions for complex valued tensors, represented by real and imaginary parts, and which obey complex arithmetic (currently no support for mixed-type arithmetic like `torch.Tensor +/-* Cplx`).
```python
n_features, n_channels = 16, 4
z = torch.randn(3, n_features*2)

cplx = RealToCplx()(z)
print(cplx)
```

Stacking and constructing purely complex-to-complex pipelines with `torch.nn.Sequential`:
```python
n_features, n_channels = 16, 4
z = torch.randn(256, n_channels, n_features * 2)

complex_model = CplxSequential(
    CplxLinear(n_features, n_features, bias=True),

    # complex: batch x n_channels x n_features
    CplxConv1d(n_channels, 3 * n_channels, kernel_size=4, stride=1, bias=False),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxModReLU(threshold=0.15),

    # complex: batch x (3 * n_channels) x (n_features - (4-1))
    CplxToCplx[torch.nn.Flatten](start_dim=-2),

    CplxToCplx[torch.tanh](),
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
# >>> torch.Size([256, 4, 32])
```

# References

.. [1] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N, Bengio, Y. & Pal, C. J. (2018). Deep complex networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H1T2hmZAb.
