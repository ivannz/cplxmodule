# CplxModule

A lightweight extension for `torch.nn` that adds layers and activations, which respect algebraic operations over the field of complex numbers.

The core implementation of the complex-valued batch normalization and weight initialization layers is based on the ICLR 2018 parer by Chiheb Trabelsi et al. on Deep Complex Networks _[1]_ and borrows ideas from their [implementation](https://github.com/ChihebTrabelsi/deep_complex_networks) (`nn.init`, `nn.modules.batchnorm`). Real-valued variational dropout and automatic relevance determination are original implementations based on the profound works by Diederik Kingma et al. (2015) _[2]_, Dmitry Molchanov et al. (2017) _[3]_, and Valery Kharitonov et al. (2018) _[4]_. Complex-valued Bayesian sparsification layers are based on original research _[5]_.

# Installation

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

# Documentation

Please refer to README files located in `cplxmodule.nn`, `cplxmodule.nn.relevance`, and `cplxmodule.nn.masked` for a high-level description of the implementation, functionality and useful code patterns.


# References

.. [1] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N, Bengio, Y. & Pal, C. J. (2018). Deep complex networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H1T2hmZAb.

.. [2] Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local reparameterization trick. In Advances in neural information processing systems (pp. 2575-2583).

.. [3] Molchanov, D., Ashukha, A., & Vetrov, D. (2017, August). Variational dropout sparsifies deep neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 2498-2507). JMLR.org

.. [4] Kharitonov, V., Molchanov, D., & Vetrov, D. (2018). Variational Dropout via Empirical Bayes. arXiv preprint arXiv:1811.00596.

.. [5] Nazarov, I., Burnaev, E. (2020, March). Bayesian Sparsification Methods for Deep Complex-valued Networks. arXiv preprint arXiv:2003.11413.
