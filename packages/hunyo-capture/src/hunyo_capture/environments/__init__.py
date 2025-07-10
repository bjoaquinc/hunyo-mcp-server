"""
Environment detection and management for hunyo-capture.

This module provides environment detection capabilities for different
notebook environments (Marimo, Jupyter) and deployment modes (development, production).
"""

from .base import EnvironmentDetector, NotebookEnvironment

__all__ = ["EnvironmentDetector", "NotebookEnvironment"]
