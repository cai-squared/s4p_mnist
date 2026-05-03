"""S4P_MNIST.

machine learning for MNIST
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("S4P_MNIST")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Cindy Cai"
__email__ = "ccai5@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
