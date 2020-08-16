# Version 2020.08.17

* FIX: Fixed shape mismatch in `nn.init.cplx_trabelsi_independent_`, which prevented it from working properly [# 11](https://github.com/ivannz/cplxmodule/issues/11)
* ENH: [Hendrik Schröter](https://github.com/Rikorose) implemented Complex Transposed Convolutions [# 8](https://github.com/ivannz/cplxmodule/pull/8), squeeze/unsqueeze methods for `Cplx` [# 7](https://github.com/ivannz/cplxmodule/pull/7), and added support for `view` and `view_as` methods for `Cplx` [# 6](https://github.com/ivannz/cplxmodule/pull/6)
* ENH: Introduce converters for special torch format of complex tensors (last dim is exactly 2) see [torch.fft](https://pytorch.org/docs/stable/generated/torch.fft.html#torch.fft)
* ENH: `Cplx` now also has `.size()` method, which mimics `torch.Tensor.size()`
* DOC: Improved documentation of `nn.casting` modules


# Version 2020.08

* structure of the `.nn.relevance` was simplified
    - importing from `ard` has been deprecated, and ARD layers have been moved
    to `.real` or `.complex` depending on their type
* changed relevance layers class hierarchy in `.relevance.real` and `.relevance.complex`:
    - factored out Gaussian Local Reparameterization into pure `*Gaussian` layers,
    that reside in `real.base` and `complex.base`
    - subclassed Variational Dropout layers (`*VD`) from `*Gaussian` with improper prior KL mixin
    - subclassed ARD layers (`*ARD`) from Variational Dropout layers `*VD` with ARD Gaussian prior KL mixin


# Version 2020.03

## Major changes in `.nn`
* The structure of the `.nn` sub-module now more closely resembles that of `torch`
    _ `.base` : `CplxToCplx` and parameter type `CplxParameter`
    _ `.casting` : real-Cplx tensor conversion layers
    _ `.linear`, `.conv`, `.activation` : essential layers and activations
    _ `.container` : sequential container which explicitly checks types of internal layers
    _ `.extra` : 1-dim Bernoulli Dropout for complex-valued tensors (Cplx)
* `CplxToCplx` can now promote torch's univariate functions to split-complex activations, e.g. use `CplxToCplx[AvgPoool1d]` instead of `CplxAvgPool1d`
* Niche complex-valued containers were removed, dropped dedicated activations, like `CplxLog` and `CplxExp`


## Major changes in `.nn.relevance`
* misnamed Bayesian layers in `.nn.relevance` were moved around and corrected
    - layers in `.real` and `.complex` were renamed to Var Dropout, with deprecation warnings for old names
    - `.ard` implements the Bayesian Dropout methods with Automatic Relevance Determination priors
* `.extensions` submodule contains relaxations, approximations, and related but non-Bayesian layers
    - `\ell_0` stochastic regularization layer was moved to `.real`
    - `Lasso` was kept to illustrate extensibility, but similarly moved to `.real`
    - Variational Dropout approximations and speeds ups were moved to `.complex`


## Enhancements
* `CplxParameter` now supports real-to-complex promotion during `load_state_dict`
* added submodule-specific readme's, explaining typical use cases and peculiarities

# Prior to 2020.03
Prior version used different version numbering and although the layers are backwards compatible, their location within the library was much different.