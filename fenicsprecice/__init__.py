import warnings

try:
    from fenics import *
except ModuleNotFoundError:
    warnings.warn("No FEniCS installation found on system. Please check whether it is found correctly. "
                  "The FEniCS adapter might not work as expected.\n\n")

from .fenicsprecice import Adapter
from . import _version
__version__ = _version.get_versions()['version']
