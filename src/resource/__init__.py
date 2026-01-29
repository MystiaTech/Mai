"""Resource management system for Mai.

This module provides intelligent resource detection, tier classification, and
adaptive scaling to enable Mai to run gracefully across different hardware
configurations from low-end systems to high-end workstations.

Key components:
- HardwareTierDetector: Classifies system capabilities into performance tiers
- ProactiveScaler: Monitors resources and requests scaling when needed
- ResourcePersonality: Communicates resource status in Mai's personality voice
"""

# Import resource components safely to avoid circular imports
try:
    from .tiers import HardwareTierDetector
    from .scaling import ProactiveScaler, ScalingDecision
    from .personality import ResourcePersonality, ResourceType

    __all__ = [
        "HardwareTierDetector",
        "ProactiveScaler",
        "ScalingDecision",
        "ResourcePersonality",
        "ResourceType",
    ]
except ImportError as e:
    print(f"Warning: Could not import resource components: {e}")
    __all__ = []
