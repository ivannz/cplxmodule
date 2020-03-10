# Plan for the first release

[+] make load_state_dict respect components of CplxParameter and allow promoting real-tensors to complex-tensors provided the state dict has no .real or .imag, but a correct key referring to the parameter.

[+] fix the incorrect naming of bayesain methods in `nn.relevance`
* rename `*ARD` named layers in `.real` and `.complex` to `*VD` layers, since they use log-uniform prior and thus are in fact Variational Dropout layers
* start deprecating importing `*ARD` named layers from `.real` and `.complex`
* fix aliases of imported layers in `.extensions`
* expose all base VD/ARD layers in `\_\_init\_\_.py` and require importing modifications from `.extensions`
* fix the text in nn/relevance/README.md

[+] fix the names for L0 regularized layer whith in fact performs probabilistic sparsification, and is not related to Variational inference

[+] check if `setup.py` has correct requirements and specifiy them explicitly
* `requires` is not a keyword, use `install_requires` and `tests_require`

[+] investigate reordering base classes in `LinearMasked(MaskedWeightMixin, Linear, _BaseRealMixin)` and similar in `nn.masked`.
* could moving it further into the bases result in a slower property lookup? It seems no:
  * from python decsriptors [doc](https://docs.python.org/3/howto/descriptor.html)
  > The implementation works through a precedence chain that gives data descriptors priority over instance variables, instance variables priority over non-data descriptors, and assigns lowest priority to \_\_getattr\_\_
  * lookup order is thus by \_\_getattribute\_\_: descriptors (aka @property), instance \_\_dict\_\_, class attributes \_\_dict\_\_, and lastly \_\_getattr\_\_.
* moved MaskedWeightMixin into \_BaseMixin

[+] get rid of `torch_module` from `.utils` and declare `activations` explicitly

[+] clean up the `nn` module itself
* remove crap from `.sequential`: `CplxResidualBottleneck`, `CplxResidualSequential` and CplxBusResidualSequential must go, and move CplxSequential to base layers
* split `.layers`, `.activation`, and `.sequential`
  * `.modules.base` : base classes (CplxToCplx, BaseRealToCplx, BaseCplxToReal), and parameter type (CplxParameter, CplxParameterAccessor)
  * `.modules.casting` : converting real tensors in various formats to and from Cplx (InterleavedRealToCplx, ConcatenatedRealToCplx, CplxToInterleavedReal, CplxToConcatenatedReal, AsTypeCplx)
  * `.modules.linear` : Linear, Bilinear, Identity, PhaseShift
  * `.modules.conv` : everything convolutional
  * `.modules.activation` : activations (CplxModReLU, CplxAdaptiveModReLU, CplxModulus, CplxAngle) and layers (CplxReal, CplxImag)
  * `.modules.container` : CplxSequential
  * `.modules.extra` : Dropout, AvgPool1d
* move `.batchnorm` to modules, keep `.init` in `.nn`
* fix imports from adjacent modules: `nn.masked` and `nn.relevance`.

[+] in `nn.relevance.complex` : drop `Cplx(*map(torch.randn_like, (s2, s2)))` and write `Cplx(torch.randn_like(s2), torch.randn_like(s2))` explicitly
* implemented `cplx.randn` and `cplx.randn_like`

[ ] residual clean up in `nn` module
* `.activation` : `CplxActivation` is the same as CplxToCplx[...]
  * CplxActivation promotes classic (real) torch functions to split activations, so yes.
  * See if it is possible to implement function promotion through CplxToCplx[...]
* `.modules.extra` : this needs thorough cleaning

[ ] documentation
* in `nn.relevance.base`, making it like in `nn.masked`
* classes in `nn.relevance`  `.real` and `.complex` should be also documented properly
* `nn.init` : document the initializations according to Trabelsi et al. (2018)
* go through README-s in each submodule to make sure that info there is correct and typical use cases described

[ ] decide the fate of `lasso` class in `nn.relevance`:
* it is irrelevant to Bayesian methods

[ ] add missing tests to the unit test suite

