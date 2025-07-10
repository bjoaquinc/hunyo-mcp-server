"""
Abstract hook system for notebook environments.

Extracted from existing UnifiedMarimoInterceptor._install_marimo_hooks() logic
to provide a generic interface for different notebook environments.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class NotebookHooks(ABC):
    """
    Abstract base class for notebook hook systems.

    Extracted from the existing Marimo hook implementation in
    UnifiedMarimoInterceptor._install_marimo_hooks() and uninstall() methods.

    Provides standard interface for:
    - Pre-execution hooks (called before cell execution)
    - Post-execution hooks (called after cell execution)
    - Finish hooks (called at session end)
    - Hook uninstallation and cleanup
    """

    def __init__(self):
        """Initialize hook manager with empty installed hooks list."""
        # Extracted from existing logic: self.installed_hooks = []
        self.installed_hooks: list[tuple[str, Callable]] = []

    @abstractmethod
    def install_pre_execution_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install pre-execution hook.

        Extracted from existing logic:
        PRE_EXECUTION_HOOKS.append(hook_func)
        self.installed_hooks.append(("PRE_EXECUTION_HOOKS", hook_func))

        Args:
            hook_func: Hook function to install, typically created by
                      _create_pre_execution_hook() in the interceptor
        """
        pass

    @abstractmethod
    def install_post_execution_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install post-execution hook.

        Extracted from existing logic:
        POST_EXECUTION_HOOKS.append(hook_func)
        self.installed_hooks.append(("POST_EXECUTION_HOOKS", hook_func))

        Args:
            hook_func: Hook function to install, typically created by
                      _create_post_execution_hook() in the interceptor
        """
        pass

    @abstractmethod
    def install_finish_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install session finish hook.

        Extracted from existing logic:
        ON_FINISH_HOOKS.append(hook_func)
        self.installed_hooks.append(("ON_FINISH_HOOKS", hook_func))

        Args:
            hook_func: Hook function to install, typically created by
                      _create_finish_hook() in the interceptor
        """
        pass

    @abstractmethod
    def uninstall_hooks(self) -> None:
        """
        Uninstall all registered hooks.

        Extracted from existing uninstall() logic:
        for hook_list_name, hook_func in self.installed_hooks:
            if hook_list_name == "PRE_EXECUTION_HOOKS":
                if hook_func in PRE_EXECUTION_HOOKS:
                    PRE_EXECUTION_HOOKS.remove(hook_func)
            # ... etc for other hook types

        Should clear self.installed_hooks after removing hooks.
        """
        pass

    def get_installed_hooks_count(self) -> int:
        """
        Get number of installed hooks.

        Extracted from existing get_session_summary() logic:
        "hooks_installed": len(self.installed_hooks)

        Returns:
            int: Number of currently installed hooks
        """
        return len(self.installed_hooks)

    def get_installed_hooks_summary(self) -> list[str]:
        """
        Get summary of installed hook types.

        Returns:
            List[str]: List of installed hook type names
        """
        return [hook_type for hook_type, _ in self.installed_hooks]
