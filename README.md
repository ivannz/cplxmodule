# CplxModule

A lightweight extension for `torch.nn` that adds layers and activations, which respect algebraic operations over the field of complex numbers, and implements real- and complex-valued Variational Dropout methods for weight sparsification.


## Documentation

Please refer to README files located in `cplxmodule.nn`, `cplxmodule.nn.relevance`, and `cplxmodule.nn.masked` for a high-level description of the implementation, functionality and useful code patterns.


## Implementation

The core implementation of the complex-valued arithmetic and layers is based on careful tracking of transformations of real and imaginary parts of complex-valued tensors, and leverages differentiable computations of the real-valued pytorch backend.

The batch normalization and weight initialization layers are based on the ICLR 2018 paper by [Chiheb Trabelsi et al. (2018)](https://openreview.net/forum?id=H1T2hmZAb) on Deep Complex Networks [[1]](#user-content-ref1) and borrow ideas from their [implementation](https://github.com/ChihebTrabelsi/deep_complex_networks) (`nn.init`, `nn.modules.batchnorm`). The complex-valued magnitude-based Max pooling is based on the idea by [Zhang et al. (2017)](https://ieeexplore.ieee.org/document/8039431) [[6]](#user-content-ref6).

The implementations of the real-valued Variational Dropout and Automatic Relevance Determination are based on the profound works by [Diederik Kingma et al. (2015)](https://proceedings.neurips.cc/paper/2015/hash/bc7316929fe1545bf0b98d114ee3ecb8-Abstract.html) [[2]](#user-content-ref2), [Dmitry Molchanov et al. (2017)](http://proceedings.mlr.press/v70/molchanov17a.html) [[3]](#user-content-ref3), and [Valery Kharitonov et al. (2018)](http://arxiv.org/abs/1811.00596) [[4]](#user-content-ref4).

Complex-valued Bayesian sparsification layers are based on the research by [Nazarov and Burnaev (2020)](http://proceedings.mlr.press/v119/nazarov20a.html) [[5]](#user-content-ref5).


## Installation

You can install this package with `pip`:
```bash
pip install cplxmodule
```
or from the git repo to get the latest version:
```bash
pip install --upgrade git+https://github.com/ivannz/cplxmodule.git
```
If you prefer a developer install (editable), then run the following from the root of the locally cloned repo
```bash
pip install -e .
```


## Citation

The proper citation for the real-valued Bayesian Sparsification layers from `cplxmodule.nn.relevance.real` is either [[3]](#user-content-ref3) (VD) or [[4]](#user-content-ref4) (ARD). If you find the complex-valued Bayesian Sparsification layers from `cplxmodule.nn.relevance.complex` useful in your research, please consider citing the following paper [[5]](#user-content-ref5):

```bibtex
@inproceedings{nazarov_bayesian_2020,
    title = {Bayesian {Sparsification} of {Deep} {C}-valued {Networks}},
    volume = {119},
    url = {http://proceedings.mlr.press/v119/nazarov20a.html},
    language = {en},
    urldate = {2021-08-02},
    booktitle = {International {Conference} on {Machine} {Learning}},
    publisher = {PMLR},
    author = {Nazarov, Ivan and Burnaev, Evgeny},
    month = nov,
    year = {2020},
    note = {ISSN: 2640-3498},
    pages = {7230--7242}
}
```


## References

<a id="user-content-ref1">[1]</a>
Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N, Bengio, Y. & Pal, C. J. (2018). Deep complex networks. In International Conference on Learning Representations, 2018.

<a id="user-content-ref2">[2]</a>
Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local reparameterization trick. In Advances in neural information processing systems (pp. 2575-2583).

<a id="user-content-ref3">[3]</a>
Molchanov, D., Ashukha, A., & Vetrov, D. (2017, August). Variational dropout sparsifies deep neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 2498-2507). JMLR.org

<a id="user-content-ref4">[4]</a>
Kharitonov, V., Molchanov, D., & Vetrov, D. (2018). Variational Dropout via Empirical Bayes. arXiv preprint arXiv:1811.00596.

<a id="user-content-ref5">[5]</a>
Nazarov, I., & Burnaev, E. (2020, November). Bayesian Sparsification of Deep C-valued Networks. In International Conference on Machine Learning (pp. 7230-7242). PMLR.

<a id="user-content-ref6">[6]</a>
Zhang, Z., Wang, H., Xu, F., & Jin, Y. Q. (2017). Complex-valued convolutional neural network and its application in polarimetric SAR image classification. IEEE Transactions on Geoscience and Remote Sensing, 55(12), 7177-7188.
