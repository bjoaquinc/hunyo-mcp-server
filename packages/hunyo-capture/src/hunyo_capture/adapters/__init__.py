"""
Execution context adapters for hunyo-capture.

This module provides abstract context handling for different notebook
environments, extracted from the existing Marimo context logic.
"""

from .base import ExecutionContextAdapter
from .marimo_adapter import MarimoContextAdapter

__all__ = ["ExecutionContextAdapter", "MarimoContextAdapter"]
