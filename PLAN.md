# Plan

## Plans for the 2020.09 release

- [ ] Extend the implementation
  - [ ] (Bernoulli Dropout) need 1d (exists), 2d and 3d
  - [x] (Convolutions) implement 3d convolutions and 3d Variational Dropout convolutions both real and complex
  - [x] (Transposed Convolutions) figure out the math and implement var dropout for transposed convos
- [ ] update the complex layer initialization from Kaiming to independent by default (check Trabelsi et al. 2018)
  - [ ] this may break older experiments in third-party repos, so need to issue a warning and a patch
- [x] deal with the `torch.nonzero(..., as_tuple=True)` deprecation warning in `utils.spectrum`
- [x] figure out the issues with ONNX support

## Plans for the first release

- [x] make `.load_state_dict` respect components of CplxParameter and allow promoting real-tensors to complex-tensors provided the state dict has no `.real` or `.imag`, but a correct key referring to the parameter.
- [x] fix the incorrect naming of Bayesain methods in `nn.relevance`
  - rename `*ARD` named layers in `.real` and `.complex` to `*VD` layers, since they use log-uniform prior and thus are in fact Variational Dropout layers
  - start deprecating importing `*ARD` named layers from `.real` and `.complex`
  - fix aliases of imported layers in `.extensions`
  - expose all base VD/ARD layers in `\_\_init\_\_.py` and require importing modifications from `.extensions`
  - fix the text in `nn/relevance/README.md`
- [x] fix the names for `L0` regularized layer which in fact performs *probabilistic sparsification*, and is not related to Variational inference
- [x] check if `setup.py` has correct requirements and specify them explicitly
  - `requires` is not a keyword, use `install_requires` and `tests_require`
- [x] investigate reordering base classes in `LinearMasked(MaskedWeightMixin, Linear, _BaseRealMixin)` and similar in `nn.masked`.
  - could moving it further into the bases result in a slower property lookup? It seems no:
    - from python descriptors [doc](https://docs.python.org/3/howto/descriptor.html)
    > The implementation works through a precedence chain that gives data descriptors priority over instance variables, instance variables priority over non-data descriptors, and assigns lowest priority to \_\_getattr\_\_
    - lookup order is thus by \_\_getattribute\_\_: descriptors (aka @property), instance \_\_dict\_\_, class attributes \_\_dict\_\_, and lastly \_\_getattr\_\_.
  - moved MaskedWeightMixin into \_BaseMixin
- [x] get rid of `torch_module` from `.utils` and declare `activations` explicitly
- [x] clean up the `nn` module itself
  - remove crap from `.sequential`: `CplxResidualBottleneck`, `CplxResidualSequential` and CplxBusResidualSequential must go, and move CplxSequential to base layers
  - split `.layers`, `.activation`, and `.sequential`
    - `.modules.base` : base classes (`CplxToCplx`, `BaseRealToCplx`, `BaseCplxToReal`), and parameter type (`CplxParameter`, `CplxParameterAccessor`)
    - `.modules.casting` : converting real tensors in various formats to and from `Cplx` (`InterleavedRealToCplx`, `ConcatenatedRealToCplx`, `CplxToInterleavedReal`, `CplxToConcatenatedReal`, `AsTypeCplx`)
    - `.modules.linear` : `Linear`, `Bilinear`, `Identity`, `PhaseShift`
    - `.modules.conv` : everything convolutional
    - `.modules.activation` : activations (`CplxModReLU`, `CplxAdaptiveModReLU`, `CplxModulus`, `CplxAngle`) and layers (`CplxReal`, `CplxImag`)
    - `.modules.container` : `CplxSequential`
    - `.modules.extra` : `Dropout`, `AvgPool1d`
  - move `.batchnorm` to modules, keep `.init` in `.nn`
  - fix imports from adjacent modules: `nn.masked` and `nn.relevance`.
- [x] in `nn.relevance.complex` : drop `Cplx(*map(torch.randn_like, (s2, s2)))` and write `Cplx(torch.randn_like(s2), torch.randn_like(s2))` explicitly
  - implemented `cplx.randn` and `cplx.randn_like`
- [x] residual clean up in `nn` module
  - `.activation` : `CplxActivation` is the same as `CplxToCplx[...]`
    - `CplxActivation` promotes classic (real) torch functions to split activations, so yes.
    - See if it is possible to implement function promotion through `CplxToCplx[...]`
      - it is possible: just reuse `CplxActivation`
    - Currently `CplxToCplx` promotes layers and real functions to independently applied layers/functions (split)
      - how should we proceed with `Cplx` trig. functions? a wrapper, or hardcoded activations?
        - the latter seems more natural, as the trig. functions are vendored by this module
        - since torch is the base, and implements a great number of univariate tensor functions and could potentially be extended, it is more natural to use a wrapper (rationale behind `CplxToCplx[...]`).
  - `.modules.extra`: this needs thorough cleaning
    - drop `CplxResidualBottleneck`, `CplxResidualSequential` and `CplxBusResidualSequential`
    - abandon `torch_module` and code the trig. activations by hand.
    - remove alias `CplxDropout1d`: use torch.nn names as much as possible
    - deprecate `CplxAvgPool1d`: it can be created in runtime with `CplxToCplx\[torch.nn.AvgPool1d\]`
- [x] documentation for Bayesian and maskable layers
  - in `nn.relevance.base`, making it like in `nn.masked`
  - classes in `nn.relevance`  `.real` and `.complex` should be also documented properly, the same goes for `.extensions`
- [x] restructure the extensions and non-Bayesian layers
  - new folder structure
    - take ARD-related declarations and move them to `relevance/ard.py`, everything else to a submodule
    - `.extensions` submodule:
      - `complex` for `Cplx`-specific extended layers: bogus penalties, approximations and other stuff, -- not directly related to Variational Dropout or automatic relevance determination
      - `real` for supplementary real-valued layers
  - decide the fate of `lasso` class in `nn.relevance`:
    - it is irrelevant to Bayesian methods: move it to `extensions/real`
- [x] documentation
  - go through README-s in each submodule to make sure that info there is correct and typical use cases are described
  - `nn.init` : document the initializations according to Trabelsi et al. (2018)
    - seems to be automatically documented using `functools.wraps` from the original `torch.nn.init` procedures.
- [x] add missing tests to the unit test suite
  - tests for `*state_dict` api compliance of `nn.masked` and `nn.base.CplxParameter`
    - implementing these test helped figure out and fix edge cases and fix them, so yay for TDD!

## Abandoned plans

The following ideas were at some point planned out, but ultimately were not pursued due to various reasons.

- [ ] begin migration to `complex` tensors in `pytorch>=1.6`
  - #20220608 consider the discussion in [issue #21](https://github.com/ivannz/cplxmodule/issues/21)
  - For C->R real-valued loss functions [grad.conj()](https://pytorch.org/docs/master/complex_numbers.html#autograd) gives a descent direction.
  - complex [autograd](https://pytorch.org/docs/master/notes/autograd.html#complex-autograd-doc)
- [ ] Consider replacing `Real` with `Tensor` in format-conversion layers, like `RealToCplx`, `CplxToReal`
  - the term `Real` has connotations with real numbers, making it very *unintuitive* to convert between `Cplx`, which is perceived as a complex number, to a torch `Tensor`, which serves merely as a storage format.
  - need a deprecation cycle for these and related functions
    - in `cplx`: `from_interleaved_real`, `from_concatenated_real`, `to_interleaved_real`, `to_concatenated_real`, aliases `from_real` and `to_real` (affects `__init__.py`)
    - `nn.modules.casting`: `InterleavedRealToCplx`, `ConcatenatedRealToCplx`, `CplxToInterleavedReal`, `CplxToConcatenatedReal`, also base classes `BaseRealToCplx` and `BaseCplxToReal`.
    - three basic types?
      - Tensor -- aka Storage
      - Real -- real-valued tensor
      - Cplx -- complex-valued tensor
- [ ] Implement scheduled mag-pruning of [Zhu and Gupta (2017)](http://arxiv.org/abs/1710.01878) or thresholded of [Wu et al. (2019)](https://arxiv.org/abs/1903.02358).
  - use `nn.masked` as a `backend` -- this will automatically support real and `Cplx` layers!!!!
  - implement as either wrapper around optimizer (bad), or as a separate entity (better)
    - settings of the target sparsity per eligible layer (`dict`)
    - method `.step()` which updates the masks according to the schedule and the current sorted magnitudes of the parameters
