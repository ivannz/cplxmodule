# CplxModule

A lightweight extension for `pytorch.nn` that adds layers and activations, which respect algebraic operations over the field of complex numbers.

The core implementation of the complex-valued batch normalization and weight initialization layers is inspired by the ICLR 2018 parer on Deep Complex Networks _[1]_ and borrows ideas from the [implementation](https://github.com/ChihebTrabelsi/deep_complex_networks). Real-valued variational dropout and automatic relevance determination are based on the profound works by Kingma _[2]_ and Molchanov _[3]_. Complex-valued Bayesian sparsification layers are based on original research.

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

# Documentation

Please refer to README files located in `cplxmodule.nn`, `cplxmodule.nn.relevance`, and `cplxmodule.nn.masked` for the high-level description of the implementation, functionality and useful code patterns.


# References

.. [1] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N, Bengio, Y. & Pal, C. J. (2018). Deep complex networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H1T2hmZAb.

.. [2] Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local reparameterization trick. In Advances in neural information processing systems (pp. 2575-2583).

.. [3] Molchanov, D., Ashukha, A., & Vetrov, D. (2017, August). Variational dropout sparsifies deep neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 2498-2507). JMLR. org.
