"""
Personality learning module for Mai.

This module provides pattern extraction, personality layer management,
and adaptive personality learning from conversation data.
"""

from .pattern_extractor import PatternExtractor
from .layer_manager import LayerManager
from .adaptation import PersonalityAdaptation

__all__ = [
    "PatternExtractor",
    "LayerManager",
    "PersonalityAdaptation",
]
