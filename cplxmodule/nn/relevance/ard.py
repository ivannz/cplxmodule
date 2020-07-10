import warnings

warnings.warn("Importing Automatic Relevance Determination layers from"
              " `cplxmodule.nn.relevance.ard` has been deprecated since"
              " version `2020.8` and will be removed in a later version."
              " Please, import from `cplxmodule.nn.relevance.real.ard` or"
              " `cplxmodule.nn.relevance.complex.ard`.",
              DeprecationWarning)

from .real import LinearARD, BilinearARD
from .real import Conv1dARD, Conv2dARD, Conv3dARD

from .complex import CplxLinearARD, CplxBilinearARD
from .complex import CplxConv1dARD, CplxConv2dARD, CplxConv3dARD
