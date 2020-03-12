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