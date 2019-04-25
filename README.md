# CplxModule

A lightweight extension for `pytorch.nn` that adds layers and activations,
which respect algebraic operations over the field of complex numbers.

The implementation is based on the ICLR 2018 parer on Deep Complex Networks
[1]_ and borrows ideas from their [implementation](https://github.com/ChihebTrabelsi/deep_complex_networks).


# Installation

Just run to imnstall with `pip`
```bash
pip install --upgrade git+https://github.com/ivannz/cplxmodule.git
```
or
```bash
python setup.py install
```
to install from thie root of thie repo.


# Example

Basically the module is designed in such a way as to be ready for plugging
into the existing `torch.nn` sequential models.

```python
import torch
import torch.nn

pass
```


# References

.. [1] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian,
       S., Santos, J. F., ... & Pal, C. J. (2017). Deep complex networks.
       arXiv preprint arXiv:1705.09792