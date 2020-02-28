import warnings

warnings.warn("Importing sparsity helper function from this submodule"
              " is deprecated, and `.stats` is to be removed in the"
              " release version of `cplxmodule`. Please import from"
              " `cplxmodule.nn.utils.sparsity`",
              FutureWarning)

from ..nn.utils.sparsity import SparsityStats

from ..nn.utils.sparsity import named_sparsity
from ..nn.utils.sparsity import sparsity
