#!/usr/bin/env python3
"""
Component factory system for hunyo-capture.

Creates environment-specific components (hooks, adapters) with proper validation
and error handling. Designed for easy extension to new notebook environments.
"""

from typing import ClassVar

from .adapters.base import ExecutionContextAdapter
from .adapters.marimo_adapter import MarimoContextAdapter
from .environments.base import EnvironmentDetector, NotebookEnvironment
from .hooks.base import NotebookHooks
from .hooks.marimo_hooks import MarimoHooks
from .logger import get_logger

factory_logger = get_logger("hunyo_capture.factory")


class ComponentFactory:
    """
    Factory for creating environment-specific components.

    Provides centralized component creation with environment validation,
    error handling, and extensibility for new notebook environments.
    """

    # Component registries for different environments
    _HOOK_REGISTRY: ClassVar[dict[NotebookEnvironment, type[NotebookHooks]]] = {
        NotebookEnvironment.MARIMO: MarimoHooks,
        # Future environments will be added here:
        # NotebookEnvironment.JUPYTER: JupyterHooks,
    }

    _ADAPTER_REGISTRY: ClassVar[
        dict[NotebookEnvironment, type[ExecutionContextAdapter]]
    ] = {
        NotebookEnvironment.MARIMO: MarimoContextAdapter,
        # Future environments will be added here:
        # NotebookEnvironment.JUPYTER: JupyterContextAdapter,
    }

    @classmethod
    def _resolve_and_validate_environment(
        cls,
        environment: NotebookEnvironment | None,
        registry: dict[NotebookEnvironment, type],
        component_type: str,
    ) -> NotebookEnvironment:
        """
        Shared environment resolution and validation logic.

        This method consolidates the environment detection, validation, and error handling
        that was previously duplicated across create_hook_manager(), create_context_adapter(),
        and create_components() methods.

        Args:
            environment: Target notebook environment. If None, auto-detects.
            registry: Component registry to validate against
            component_type: Type of component for error messages

        Returns:
            NotebookEnvironment: Validated environment

        Raises:
            EnvironmentNotSupportedError: If environment is not supported
            EnvironmentNotAvailableError: If environment is not available
        """
        # Auto-detect environment if not provided
        if environment is None:
            environment = EnvironmentDetector.detect_environment()
            factory_logger.info(
                f"[FACTORY] Auto-detected environment: {environment.value}"
            )

        # Validate environment is supported
        if environment not in registry:
            msg = (
                f"{component_type} not available for environment: {environment.value}. "
                f"Supported environments: {list(registry.keys())}"
            )
            raise EnvironmentNotSupportedError(msg)

        # Validate environment is available
        if not EnvironmentDetector.validate_environment(environment):
            msg = (
                f"Environment {environment.value} is not available. "
                "Please install required dependencies or check environment setup."
            )
            raise EnvironmentNotAvailableError(msg)

        return environment

    @classmethod
    def create_hook_manager(
        cls, environment: NotebookEnvironment | None = None
    ) -> NotebookHooks:
        """
        Create environment-specific hook manager.

        Args:
            environment: Target notebook environment. If None, auto-detects.

        Returns:
            NotebookHooks: Environment-specific hook manager instance

        Raises:
            EnvironmentNotSupportedError: If environment is not supported
            EnvironmentNotAvailableError: If environment is not available
        """
        # Use shared validation logic
        environment = cls._resolve_and_validate_environment(
            environment, cls._HOOK_REGISTRY, "Hook manager"
        )

        # Create hook manager
        hook_class = cls._HOOK_REGISTRY[environment]
        hook_manager = hook_class()

        factory_logger.info(f"[FACTORY] Created hook manager: {hook_class.__name__}")
        return hook_manager

    @classmethod
    def create_context_adapter(
        cls, environment: NotebookEnvironment | None = None
    ) -> ExecutionContextAdapter:
        """
        Create environment-specific context adapter.

        Args:
            environment: Target notebook environment. If None, auto-detects.

        Returns:
            ExecutionContextAdapter: Environment-specific context adapter instance

        Raises:
            EnvironmentNotSupportedError: If environment is not supported
            EnvironmentNotAvailableError: If environment is not available
        """
        # Use shared validation logic
        environment = cls._resolve_and_validate_environment(
            environment, cls._ADAPTER_REGISTRY, "Context adapter"
        )

        # Create context adapter
        adapter_class = cls._ADAPTER_REGISTRY[environment]
        context_adapter = adapter_class()

        factory_logger.info(
            f"[FACTORY] Created context adapter: {adapter_class.__name__}"
        )
        return context_adapter

    @classmethod
    def create_components(
        cls, environment: NotebookEnvironment | None = None
    ) -> tuple[NotebookHooks, ExecutionContextAdapter]:
        """
        Create both hook manager and context adapter for an environment.

        Convenience method that creates both components with consistent
        environment detection and validation.

        Args:
            environment: Target notebook environment. If None, auto-detects.

        Returns:
            Tuple[NotebookHooks, ExecutionContextAdapter]: Hook manager and context adapter

        Raises:
            EnvironmentNotSupportedError: If environment is not supported
            EnvironmentNotAvailableError: If environment is not available
        """
        # Auto-detect environment if not provided (do this once)
        if environment is None:
            environment = EnvironmentDetector.detect_environment()
            factory_logger.info(
                f"[FACTORY] Auto-detected environment: {environment.value}"
            )

        # Create both components using the same environment (no need to re-validate)
        hook_manager = cls.create_hook_manager(environment)
        context_adapter = cls.create_context_adapter(environment)

        factory_logger.info(f"[FACTORY] Created component pair for {environment.value}")
        return hook_manager, context_adapter

    @classmethod
    def get_supported_environments(cls) -> list[NotebookEnvironment]:
        """
        Get list of supported notebook environments.

        Returns:
            List[NotebookEnvironment]: List of supported environments
        """
        # Return intersection of hook and adapter supported environments
        hook_environments = set(cls._HOOK_REGISTRY.keys())
        adapter_environments = set(cls._ADAPTER_REGISTRY.keys())
        supported = hook_environments & adapter_environments

        return list(supported)

    @classmethod
    def register_hook_manager(
        cls, environment: NotebookEnvironment, hook_class: type[NotebookHooks]
    ) -> None:
        """
        Register a new hook manager for an environment.

        Args:
            environment: Target notebook environment
            hook_class: Hook manager class to register
        """
        cls._HOOK_REGISTRY[environment] = hook_class
        factory_logger.info(
            f"[FACTORY] Registered hook manager: {hook_class.__name__} for {environment.value}"
        )

    @classmethod
    def register_context_adapter(
        cls,
        environment: NotebookEnvironment,
        adapter_class: type[ExecutionContextAdapter],
    ) -> None:
        """
        Register a new context adapter for an environment.

        Args:
            environment: Target notebook environment
            adapter_class: Context adapter class to register
        """
        cls._ADAPTER_REGISTRY[environment] = adapter_class
        factory_logger.info(
            f"[FACTORY] Registered context adapter: {adapter_class.__name__} for {environment.value}"
        )


class EnvironmentNotSupportedError(ValueError):
    """Raised when requested environment is not supported by the factory."""

    pass


class EnvironmentNotAvailableError(RuntimeError):
    """Raised when requested environment is not available in current context."""

    pass


# Convenience functions for direct component creation
def create_hook_manager(
    environment: NotebookEnvironment | None = None,
) -> NotebookHooks:
    """
    Create environment-specific hook manager.

    Convenience function that delegates to ComponentFactory.create_hook_manager().

    Args:
        environment: Target notebook environment. If None, auto-detects.

    Returns:
        NotebookHooks: Environment-specific hook manager instance
    """
    return ComponentFactory.create_hook_manager(environment)


def create_context_adapter(
    environment: NotebookEnvironment | None = None,
) -> ExecutionContextAdapter:
    """
    Create environment-specific context adapter.

    Convenience function that delegates to ComponentFactory.create_context_adapter().

    Args:
        environment: Target notebook environment. If None, auto-detects.

    Returns:
        ExecutionContextAdapter: Environment-specific context adapter instance
    """
    return ComponentFactory.create_context_adapter(environment)


def create_components(
    environment: NotebookEnvironment | None = None,
) -> tuple[NotebookHooks, ExecutionContextAdapter]:
    """
    Create both hook manager and context adapter for an environment.

    Convenience function that delegates to ComponentFactory.create_components().

    Args:
        environment: Target notebook environment. If None, auto-detects.

    Returns:
        Tuple[NotebookHooks, ExecutionContextAdapter]: Hook manager and context adapter
    """
    return ComponentFactory.create_components(environment)
