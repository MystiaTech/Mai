"""Resource management system for Mai.

This module provides intelligent resource detection, tier classification, and
adaptive scaling to enable Mai to run gracefully across different hardware
configurations from low-end systems to high-end workstations.

Key components:
- HardwareTierDetector: Classifies system capabilities into performance tiers
- ProactiveScaler: Monitors resources and requests scaling when needed
- ResourcePersonality: Communicates resource status in Mai's personality voice
"""

from .tiers import HardwareTierDetector

__all__ = [
    "HardwareTierDetector",
]
