"""Mai - Autonomous Conversational AI Agent

A local-first AI agent that can improve her own code through
safe, reviewed modifications.
"""

__version__ = "0.1.0"
__author__ = "Mai Project"

from .models import LMStudioAdapter, ResourceMonitor

__all__ = ["LMStudioAdapter", "ResourceMonitor"]
