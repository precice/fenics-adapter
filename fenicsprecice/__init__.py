import warnings

try:
    from fenics import *
except ModuleNotFoundError:
    warnings.warn("No FEniCS installation found on system. Please check whether it is found correctly. "
                  "The FEniCS adapter might not work as expected.\n\n")

from .fenicsprecice import Adapter
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
