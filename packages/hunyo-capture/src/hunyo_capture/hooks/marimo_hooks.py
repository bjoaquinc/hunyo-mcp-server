"""
Marimo-specific hook implementation.

Extracted from existing UnifiedMarimoInterceptor._install_marimo_hooks()
and uninstall() methods to preserve exact behavior.
"""

from collections.abc import Callable
from typing import Any

from hunyo_capture.logger import get_logger

from .base import NotebookHooks

unified_logger = get_logger("hunyo_capture.hooks.marimo")


class MarimoHooks(NotebookHooks):
    """
    Marimo-specific hook implementation.

    Extracted from existing UnifiedMarimoInterceptor._install_marimo_hooks()
    and uninstall() methods to preserve exact production behavior.
    """

    def install_pre_execution_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install Marimo pre-execution hook.

        Extracted from existing _install_marimo_hooks() logic:
        PRE_EXECUTION_HOOKS.append(pre_hook)
        self.installed_hooks.append(("PRE_EXECUTION_HOOKS", pre_hook))
        """
        try:
            from marimo._runtime.runner.hooks import PRE_EXECUTION_HOOKS

            PRE_EXECUTION_HOOKS.append(hook_func)
            self.installed_hooks.append(("PRE_EXECUTION_HOOKS", hook_func))

        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
            raise

    def install_post_execution_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install Marimo post-execution hook.

        Extracted from existing _install_marimo_hooks() logic:
        POST_EXECUTION_HOOKS.append(post_hook)
        self.installed_hooks.append(("POST_EXECUTION_HOOKS", post_hook))
        """
        try:
            from marimo._runtime.runner.hooks import POST_EXECUTION_HOOKS

            POST_EXECUTION_HOOKS.append(hook_func)
            self.installed_hooks.append(("POST_EXECUTION_HOOKS", hook_func))

        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
            raise

    def install_finish_hook(self, hook_func: Callable[..., Any]) -> None:
        """
        Install Marimo finish hook.

        Extracted from existing _install_marimo_hooks() logic:
        ON_FINISH_HOOKS.append(finish_hook)
        self.installed_hooks.append(("ON_FINISH_HOOKS", finish_hook))
        """
        try:
            from marimo._runtime.runner.hooks import ON_FINISH_HOOKS

            ON_FINISH_HOOKS.append(hook_func)
            self.installed_hooks.append(("ON_FINISH_HOOKS", hook_func))

        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
            raise

    def uninstall_hooks(self) -> None:
        """
        Uninstall all Marimo hooks.

        Extracted from existing uninstall() method logic:
        for hook_list_name, hook_func in self.installed_hooks:
            if hook_list_name == "PRE_EXECUTION_HOOKS":
                if hook_func in PRE_EXECUTION_HOOKS:
                    PRE_EXECUTION_HOOKS.remove(hook_func)
            # ... etc for other hook types
        """
        try:
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )

            unified_logger.info("[UNINSTALL] Removing marimo hooks...")

            # Remove all installed hooks - exact logic from existing uninstall() method
            for hook_list_name, hook_func in self.installed_hooks:
                if hook_list_name == "PRE_EXECUTION_HOOKS":
                    if hook_func in PRE_EXECUTION_HOOKS:
                        PRE_EXECUTION_HOOKS.remove(hook_func)
                elif hook_list_name == "POST_EXECUTION_HOOKS":
                    if hook_func in POST_EXECUTION_HOOKS:
                        POST_EXECUTION_HOOKS.remove(hook_func)
                elif hook_list_name == "ON_FINISH_HOOKS":
                    if hook_func in ON_FINISH_HOOKS:
                        ON_FINISH_HOOKS.remove(hook_func)

            # Clear installed hooks list
            self.installed_hooks.clear()

            unified_logger.info("[OK] Marimo hooks removed")

        except ImportError:
            # Graceful degradation if Marimo not available - matches existing behavior
            unified_logger.warning("[WARN] Marimo hooks not available for removal")
        except Exception as e:
            unified_logger.warning(f"[WARN] Error during marimo hook removal: {e}")

    def install_all_hooks(
        self, pre_hook: Callable, post_hook: Callable, finish_hook: Callable
    ) -> None:
        """
        Install all hooks at once - extracted from existing _install_marimo_hooks() method.

        This method replicates the exact behavior of the original _install_marimo_hooks()
        method for backward compatibility and testing purposes.
        """
        try:
            from marimo._runtime.runner.hooks import (
                ON_FINISH_HOOKS,
                POST_EXECUTION_HOOKS,
                PRE_EXECUTION_HOOKS,
            )
        except ImportError as e:
            unified_logger.error(f"[ERROR] Failed to import marimo hooks: {e}")
            raise

        unified_logger.info("[INSTALL] Installing marimo native hooks...")

        # Install pre-execution hook
        PRE_EXECUTION_HOOKS.append(pre_hook)
        self.installed_hooks.append(("PRE_EXECUTION_HOOKS", pre_hook))

        # Install post-execution hook
        POST_EXECUTION_HOOKS.append(post_hook)
        self.installed_hooks.append(("POST_EXECUTION_HOOKS", post_hook))

        # Install finish hook (for cleanup)
        ON_FINISH_HOOKS.append(finish_hook)
        self.installed_hooks.append(("ON_FINISH_HOOKS", finish_hook))
