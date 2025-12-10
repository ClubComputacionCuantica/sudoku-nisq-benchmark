"""Quantum computing providers for the BackendManager."""

from .base import QuantumProvider
from .aer import AerProvider

# Always available providers
__all__ = ["QuantumProvider", "AerProvider"]

# Optional providers (require additional dependencies)
try:
    from .ibm import IBMProvider
    __all__.append("IBMProvider")
except ImportError:
    IBMProvider = None  # type: ignore

try:
    from .quantinuum import QuantinuumProvider
    __all__.append("QuantinuumProvider")
except ImportError:
    QuantinuumProvider = None  # type: ignore

try:
    from .aws import AWSProvider
    __all__.append("AWSProvider")
except ImportError:
    AWSProvider = None  # type: ignore