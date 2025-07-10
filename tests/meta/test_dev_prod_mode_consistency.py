#!/usr/bin/env python3
"""
Cross-Package Dev/Prod Mode Consistency Tests

Critical tests ensuring hunyo-capture and hunyo-mcp-server use consistent
data directories and file paths in both development and production modes.

This addresses the critical issue where events captured by hunyo-capture
must be discoverable by hunyo-mcp-server in both dev and prod environments.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestDevProdModeConsistency:
    """Test dev/prod mode consistency between hunyo-capture and hunyo-mcp-server"""

    def test_development_mode_data_directory_consistency(self):
        """Test both packages use same data directory in development mode"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_user_data_dir as capture_get_data_dir

        from hunyo_mcp_server.config import get_hunyo_data_dir as mcp_get_data_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create development markers in temporary directory
            project_root = Path(temp_dir) / "project"
            project_root.mkdir()
            (project_root / ".git").mkdir()
            (project_root / "pyproject.toml").touch()
            (project_root / "packages").mkdir()

            # Both packages should detect development mode and use same directory
            with patch("pathlib.Path.cwd", return_value=project_root):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    capture_data_dir = capture_get_data_dir()
                    mcp_data_dir = mcp_get_data_dir()

                    # Both should point to same directory
                    assert capture_data_dir == str(mcp_data_dir)

                    # Both should be using project root/.hunyo
                    expected_path = str(project_root / ".hunyo")
                    assert capture_data_dir == expected_path
                    assert str(mcp_data_dir) == expected_path

    def test_production_mode_data_directory_consistency(self):
        """Test both packages use same data directory in production mode"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_user_data_dir as capture_get_data_dir

        from hunyo_mcp_server.config import get_hunyo_data_dir as mcp_get_data_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate production environment (no dev markers)
            fake_home = Path(temp_dir) / "fake_home"
            fake_home.mkdir()

            with patch("pathlib.Path.home", return_value=fake_home):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "0"}):
                    capture_data_dir = capture_get_data_dir()
                    mcp_data_dir = mcp_get_data_dir()

                    # Both should point to same directory
                    assert capture_data_dir == str(mcp_data_dir)

                    # Both should be using user home/.hunyo
                    expected_path = str(fake_home / ".hunyo")
                    assert capture_data_dir == expected_path
                    assert str(mcp_data_dir) == expected_path

    def test_event_file_path_consistency(self):
        """Test both packages generate identical event file paths"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_event_filenames, get_user_data_dir

        from hunyo_mcp_server.config import get_event_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test notebook
            notebook_path = Path(temp_dir) / "test_notebook.py"
            notebook_path.write_text("import pandas as pd\ndf = pd.DataFrame()")

            # Test with development mode
            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    # Get file paths from both packages
                    capture_data_dir = get_user_data_dir()
                    capture_files = get_event_filenames(
                        str(notebook_path), capture_data_dir
                    )

                    mcp_files = [
                        get_event_file_path("runtime", str(notebook_path)),
                        get_event_file_path("lineage", str(notebook_path)),
                        get_event_file_path("dataframe_lineage", str(notebook_path)),
                    ]

                    # All file paths should match exactly
                    assert len(capture_files) == len(mcp_files)
                    for capture_file, mcp_file in zip(
                        capture_files, mcp_files, strict=False
                    ):
                        assert str(capture_file) == str(
                            mcp_file
                        ), f"Mismatch: {capture_file} != {mcp_file}"

    def test_hash_generation_consistency(self):
        """Test both packages generate identical notebook hashes"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_notebook_file_hash as capture_hash

        from hunyo_mcp_server.config import get_notebook_file_hash as mcp_hash

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test notebook
            notebook_path = Path(temp_dir) / "test_notebook.py"
            notebook_path.write_text(
                "import pandas as pd\ndf = pd.DataFrame({'test': [1, 2, 3]})"
            )

            # Both packages should generate identical hashes
            capture_hash_result = capture_hash(str(notebook_path))
            mcp_hash_result = mcp_hash(str(notebook_path))

            assert capture_hash_result == mcp_hash_result
            assert len(capture_hash_result) == 8  # Should be 8-character hex
            assert all(
                c in "0123456789abcdef" for c in capture_hash_result
            )  # Should be hex

    def test_directory_structure_consistency(self):
        """Test both packages create identical directory structures"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_user_data_dir as capture_get_data_dir

        from hunyo_mcp_server.config import (
            get_database_path,
            get_dataframe_lineage_events_dir,
            get_event_directories,
        )
        from hunyo_mcp_server.config import get_hunyo_data_dir as mcp_get_data_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            project_root.mkdir()
            (project_root / ".git").mkdir()

            with patch("pathlib.Path.cwd", return_value=project_root):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    # Initialize both packages (creates directory structures)
                    capture_data_dir = Path(capture_get_data_dir())
                    mcp_data_dir = Path(mcp_get_data_dir())

                    # Verify base directories match
                    assert capture_data_dir == mcp_data_dir

                    # Verify MCP server directory structure
                    runtime_dir, lineage_dir = get_event_directories()
                    dataframe_lineage_dir = get_dataframe_lineage_events_dir()
                    database_path = get_database_path()

                    # All MCP server directories should be under same base
                    assert (
                        Path(runtime_dir).parent.parent.resolve()
                        == mcp_data_dir.resolve()
                    )
                    assert (
                        Path(lineage_dir).parent.parent.resolve()
                        == mcp_data_dir.resolve()
                    )
                    assert (
                        Path(dataframe_lineage_dir).parent.parent.resolve()
                        == mcp_data_dir.resolve()
                    )
                    assert (
                        Path(database_path).parent.parent.resolve()
                        == mcp_data_dir.resolve()
                    )  # Database file is in database/ dir

                    # Expected directory structure should exist
                    expected_dirs = [
                        capture_data_dir / "events" / "runtime",
                        capture_data_dir / "events" / "lineage",
                        capture_data_dir / "events" / "dataframe_lineage",
                        capture_data_dir / "database",
                        capture_data_dir / "config",
                    ]

                    for expected_dir in expected_dirs:
                        assert (
                            expected_dir.exists()
                        ), f"Missing directory: {expected_dir}"

    def test_environment_variable_consistency(self):
        """Test both packages respect same environment variables"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_user_data_dir as capture_get_data_dir

        from hunyo_mcp_server.config import get_hunyo_data_dir as mcp_get_data_dir
        from hunyo_mcp_server.config import is_development_mode

        # Test HUNYO_DEV_MODE environment variable
        test_cases = [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_home = Path(temp_dir) / "fake_home"
            fake_home.mkdir()

            for env_value, expected_dev_mode in test_cases:
                with patch("pathlib.Path.home", return_value=fake_home):
                    with patch.dict(os.environ, {"HUNYO_DEV_MODE": env_value}):
                        # MCP server should detect mode correctly
                        assert is_development_mode() == expected_dev_mode

                        # Both packages should use consistent directories
                        capture_data_dir = capture_get_data_dir()
                        mcp_data_dir = mcp_get_data_dir()

                        assert capture_data_dir == str(mcp_data_dir)

    def test_real_world_scenario_compatibility(self):
        """Test full compatibility scenario with real notebook processing"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_event_filenames, get_user_data_dir

        from hunyo_mcp_server.config import get_event_file_path, get_hunyo_data_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic notebook structure
            project_root = Path(temp_dir) / "my_project"
            project_root.mkdir()
            (project_root / ".git").mkdir()
            (project_root / "pyproject.toml").touch()

            notebooks_dir = project_root / "notebooks"
            notebooks_dir.mkdir()
            notebook_path = notebooks_dir / "data_analysis.py"
            notebook_path.write_text(
                """
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Analysis
result = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count']
}).round(2)

print("Analysis complete")
"""
            )

            # Simulate development workflow
            with patch("pathlib.Path.cwd", return_value=project_root):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    # 1. hunyo-capture creates events
                    capture_data_dir = get_user_data_dir()
                    capture_files = get_event_filenames(
                        str(notebook_path), capture_data_dir
                    )

                    # 2. hunyo-mcp-server should find same files
                    mcp_data_dir = get_hunyo_data_dir()
                    mcp_files = [
                        get_event_file_path("runtime", str(notebook_path)),
                        get_event_file_path("lineage", str(notebook_path)),
                        get_event_file_path("dataframe_lineage", str(notebook_path)),
                    ]

                    # 3. Verify complete compatibility
                    assert capture_data_dir == str(mcp_data_dir)
                    assert len(capture_files) == len(mcp_files)
                    for capture_file, mcp_file in zip(
                        capture_files, mcp_files, strict=False
                    ):
                        assert str(capture_file) == str(mcp_file)

                    # 4. Create mock event files to test discovery
                    for file_path in capture_files:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(file_path).write_text('{"event": "test"}')

                    # 5. MCP server should find all files
                    for mcp_file in mcp_files:
                        assert Path(
                            mcp_file
                        ).exists(), f"MCP server cannot find: {mcp_file}"


class TestCriticalFailureScenarios:
    """Test scenarios that would cause critical failures in production"""

    def test_different_data_directories_detection(self):
        """Test that catches if packages use different data directories"""
        # This test should FAIL if the packages become inconsistent
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_user_data_dir as capture_get_data_dir

        from hunyo_mcp_server.config import get_hunyo_data_dir as mcp_get_data_dir

        # Test both development and production modes
        test_scenarios = [
            ("dev", {"HUNYO_DEV_MODE": "1"}),
            ("prod", {"HUNYO_DEV_MODE": "0"}),
        ]

        for scenario_name, env_vars in test_scenarios:
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_home = Path(temp_dir) / "fake_home"
                fake_home.mkdir()

                with patch("pathlib.Path.home", return_value=fake_home):
                    with patch.dict(os.environ, env_vars):
                        capture_data_dir = capture_get_data_dir()
                        mcp_data_dir = mcp_get_data_dir()

                        # CRITICAL: These must be identical
                        assert capture_data_dir == str(mcp_data_dir), (
                            f"CRITICAL FAILURE in {scenario_name} mode: "
                            f"hunyo-capture uses {capture_data_dir} but "
                            f"hunyo-mcp-server uses {mcp_data_dir}. "
                            f"Events will be lost!"
                        )

    def test_missing_event_file_discovery(self):
        """Test that catches if MCP server can't find capture events"""
        # Import both packages
        import sys

        sys.path.insert(0, "packages/hunyo-capture/src")
        sys.path.insert(0, "packages/hunyo-mcp-server/src")

        from hunyo_capture import get_event_filenames, get_user_data_dir

        from hunyo_mcp_server.config import get_event_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test scenario
            notebook_path = Path(temp_dir) / "test.py"
            notebook_path.write_text("import pandas as pd")

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    # hunyo-capture creates event files
                    capture_data_dir = get_user_data_dir()
                    capture_files = get_event_filenames(
                        str(notebook_path), capture_data_dir
                    )

                    # Simulate event file creation
                    for file_path in capture_files:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(file_path).write_text('{"event": "test"}')

                    # MCP server must find ALL event files
                    event_types = ["runtime", "lineage", "dataframe_lineage"]
                    for event_type in event_types:
                        mcp_file = get_event_file_path(event_type, str(notebook_path))
                        assert Path(mcp_file).exists(), (
                            f"CRITICAL FAILURE: MCP server cannot find {event_type} "
                            f"events at {mcp_file}. Events created by hunyo-capture "
                            f"at {capture_files} are not discoverable!"
                        )
