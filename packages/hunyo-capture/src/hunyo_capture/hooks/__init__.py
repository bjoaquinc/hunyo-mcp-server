"""
Hook system abstraction for hunyo-capture.

This module provides abstract hook management for different notebook
environments, extracted from the existing Marimo hook implementation.
"""

from .base import NotebookHooks
from .marimo_hooks import MarimoHooks

__all__ = ["MarimoHooks", "NotebookHooks"]
