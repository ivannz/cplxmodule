# Real- and Complex- valued Masked Layers

This submodule implements maskable real- and complex- valued layers, e.g. for fine-tuning sparsified models.

## Usage

The typical use case is covered in the README of `nn.relevance`, since it is reasonable to use maskable layers after sparsification methods did their job. Nevertheless, below is a simple illustration of the interface for standalone use:

```python
from cplxmodule.nn.masked import LinearMasked

lin = LinearMasked(23, 32, bias=True)

lin.weight  # stored non sparse weight

lin.weight.shape  # (32, 23)

lin.mask = torch.randint(2, (32, 23))  # unstructured sparsity
lin.mask = torch.randint(2, (1, 23))   # structured input feature masking
lin.mask = torch.randint(2, (32, 1))   # structured output masking

lin.weight_masked  # read-only masked weight for forward pass
```

### Setting and collecting masks

All subclasses of `BaseMasked`, unless explicitly redefined, have a readable and writable property `.mask`, and an alternative method interface `.mask_(tensor)`. Setting `.mask` to a tensor automatically does the necessary dtype conversion, device placement and broadcasting. The right-hand side value must be bradcastible to the real shape of the layer's weight.

Setting `.mask` to `None` removes the currently set mask (if present, and has not effect otherwise). Without a set `.mask` the layer refuses to function and raises a `RuntimeError` exception upon forward pass. This constraint was deliberately added to alert the user about absent masks, since the use of maskable layer implies that something should be masked. If masking is not required it is recommended to use the equivalent layer from `torch.nn`. If however a mask is dynamic, and adapts to the learning process, then it is possible to effectively disable masking by assigning `.mask = torch.tensor(1.)`.

Masked layers fully support serializing and deserializing through the pytorch's standard `.state_dict()` and `.load_state_dict()` methods. `nn.masked` also provides `deploy_masks()` procedure, which accepts a properly keyed state dict, and sets masks through overwriting `.mask` of every compatible layer.

The `nn.masked` submodule has a function `named_masks()`, which returns a generator of name-mask pairs, which traverses the module's architecture in exactly the same DFS manner as `.named_modules()` of `torch.nn.Module`.

### Cleaning masks and weights

It is entirely possible for the masks to not be binary, and instead take any real value. Although layers with such masks would operate normally, their parameter values might be differently scaled due to non-binary masks. To address this possibility the `nn.masked` submodule provides `binarize_masks()` function, which takes in a `state_dict` of model's weights and compatible `masks` dictionary, and forces masks to `0-1` values by pre-multiplying and overwriting the corresponding weight in the `state_dict`. See the README in `nn.relevance` for a use case and a code snippet.

## Implementation

At the time of the original design and implementation (around may of 2019) sparse tensor were not very well supported in pytorch, and if they were, their throughput left a lot to be desired. Thus a decision was made to use explicit dense sparsity masks and multiply the associated parameters by them on forward pass. This introduces some overhead, but ensures that the zeroed out weights' values are kept, but have no effect, and do not get updated, this also makes it possible to hot-swap the masks during training or inference, which is useful for magnitude pruning methods. Optimization methods which use running statistics based on gradient information could potentially affect the masked weights due to the accumulated momentum, although the gradients for the masked are explicit zeros.


### Modules

The modules in `nn.masked` implement both `real`- and `complex` valued maskable layers.

* (real) LinearMasked, Conv1dMasked, Conv2dMasked, Conv3dMasked, BilinearMasked
* (complex) CplxLinearMasked, CplxConv1dMasked, CplxConv2dMasked, CplxConv3dMasked, CplxBilinearMasked

### Subclassing Modules

The `BaseMasked` base class has a non-default `__init__`, which sets up the `mask` buffer necessary for proper and pytorch-friendly dtype conversion and device placement. Therefore it should be placed as right as possible in the base classes list of the class definition, but before the `torch.nn.Module` class. This properly sets up the method resolution order and allows proper `__init__` and access to `.mask`.

### Compatibility

Compatible with `.load_state_dict()`, `.state_dict()` and interchangeable with `torch.nn` and `nn.relevance`.
