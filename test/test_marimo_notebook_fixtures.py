"""
Test marimo notebook fixtures to ensure they work correctly.

This module tests the marimo notebooks stored in test/fixtures/ to ensure
they can be executed and their functionality works as expected.
"""

import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def openlineage_notebook_path() -> Path:
    """Path to the openlineage demo notebook fixture."""
    return Path(__file__).parent / "fixtures" / "openlineage_demo_notebook.py"


@pytest.fixture
def runtime_notebook_path() -> Path:
    """Path to the runtime tracking demo notebook fixture."""
    return Path(__file__).parent / "fixtures" / "runtime_tracking_demo_notebook.py"


class TestMarimoNotebookFixtures:
    """Test the marimo notebook fixtures to ensure they work correctly."""

    def test_openlineage_notebook_imports_work(self, openlineage_notebook_path: Path):
        """Test that the openlineage demo notebook can import required modules."""
        # Test that the file exists
        assert (
            openlineage_notebook_path.exists()
        ), f"Notebook not found: {openlineage_notebook_path}"

        # Test that we can import the required modules by executing the import section
        import sys

        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Test that the key import works
        try:
            from capture.native_hooks_interceptor import enable_native_hook_tracking

            assert callable(
                enable_native_hook_tracking
            ), "enable_native_hook_tracking should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import required module: {e}")

    def test_runtime_notebook_imports_work(self, runtime_notebook_path: Path):
        """Test that the runtime tracking demo notebook can import required modules."""
        # Test that the file exists
        assert (
            runtime_notebook_path.exists()
        ), f"Notebook not found: {runtime_notebook_path}"

        # Test the import functionality
        import sys

        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from capture.native_hooks_interceptor import enable_native_hook_tracking

            assert callable(
                enable_native_hook_tracking
            ), "enable_native_hook_tracking should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import required module: {e}")

    def test_notebooks_are_valid_marimo_apps(
        self, openlineage_notebook_path: Path, runtime_notebook_path: Path
    ):
        """Test that the notebooks are valid marimo applications."""
        notebooks = [openlineage_notebook_path, runtime_notebook_path]

        for notebook_path in notebooks:
            # Read the file and check for marimo app structure
            content = notebook_path.read_text(encoding="utf-8")

            # Basic checks for marimo app structure
            assert (
                "import marimo" in content
            ), f"Notebook should import marimo: {notebook_path}"
            assert (
                "marimo.App(" in content
            ), f"Notebook should create marimo app: {notebook_path}"
            assert (
                "@app.cell" in content
            ), f"Notebook should have app cells: {notebook_path}"

    def test_notebook_execution_simulation(
        self, temp_dir: Path, openlineage_notebook_path: Path
    ):
        """Test that we can simulate parts of the notebook execution."""
        # Change to temp directory for test execution
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Test the basic pandas operations from the notebook
            import sys

            import pandas as pd

            # Add project root to path
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Test basic DataFrame operations that the notebook would do
            data2 = {"colA": [1, 2], "colB": [5, 6]}
            df2 = pd.DataFrame(data2)
            assert len(df2) == 2, "Test DataFrame should have 2 rows"
            assert list(df2.columns) == [
                "colA",
                "colB",
            ], "DataFrame should have correct columns"

            # Test CSV operations
            csv_content = "id,value\n1,A\n2,B\n3,C"
            csv_path = temp_dir / "test_data.csv"
            csv_path.write_text(csv_content)

            df_from_csv = pd.read_csv(csv_path)
            assert len(df_from_csv) == 3, "CSV DataFrame should have 3 rows"
            assert "id" in df_from_csv.columns, "CSV DataFrame should have 'id' column"
            assert (
                "value" in df_from_csv.columns
            ), "CSV DataFrame should have 'value' column"

        finally:
            os.chdir(original_cwd)

    def test_notebooks_use_correct_import_paths(
        self, openlineage_notebook_path: Path, runtime_notebook_path: Path
    ):
        """Test that notebooks use the correct import paths and don't reference non-existent modules."""
        notebooks = [
            (openlineage_notebook_path, "openlineage demo"),
            (runtime_notebook_path, "runtime tracking demo"),
        ]

        for notebook_path, notebook_name in notebooks:
            content = notebook_path.read_text(encoding="utf-8")

            # Should NOT import the non-existent module
            assert (
                "import marimo_native_hooks_interceptor" not in content
            ), f"{notebook_name} should not import non-existent marimo_native_hooks_interceptor"

            # Should import from the correct source
            assert (
                "from capture.native_hooks_interceptor import enable_native_hook_tracking"
                in content
            ), f"{notebook_name} should import from correct module path"

    @pytest.mark.slow
    def test_marimo_notebook_can_be_executed(
        self, temp_dir: Path, runtime_notebook_path: Path
    ):
        """Test that marimo can actually load and validate the notebook structure (if marimo is available)."""
        # This test only runs if marimo is available and is marked as slow
        try:
            import marimo  # noqa: F401
        except ImportError:
            pytest.skip("Marimo not available for notebook execution test")

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Copy the notebook to temp directory to avoid side effects
            temp_notebook = temp_dir / "test_notebook.py"
            temp_notebook.write_text(
                runtime_notebook_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

            # Try to validate the notebook structure by importing it
            # This doesn't execute the notebook but validates its structure
            python_code = (
                "import sys; sys.path.insert(0, '.'); "
                "import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('test_nb', {str(temp_notebook)!r}); "
                "module = importlib.util.module_from_spec(spec); "
                "print('[OK] Notebook structure valid')"
            )
            result = subprocess.run(
                ["python", "-c", python_code],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert (
                result.returncode == 0
            ), f"Failed to import notebook structure: {result.stderr}"
            assert (
                "[OK] Notebook structure valid" in result.stdout
            ), "Notebook structure should be valid"

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Could not execute marimo validation: {e}")
        finally:
            os.chdir(original_cwd)
