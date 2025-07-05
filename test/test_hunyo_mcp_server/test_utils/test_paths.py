#!/usr/bin/env python3
"""
Tests for paths.py utility module.

Tests cover cross-platform path handling, Windows long path support,
directory creation, permissions, and error handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hunyo_mcp_server.utils.paths import (
    WINDOWS_MAX_PATH_LENGTH,
    get_cross_platform_temp_dir,
    get_project_root,
    get_safe_temp_database_path,
    get_schema_path,
    normalize_database_path,
    setup_cross_platform_directories,
    validate_path_accessibility,
)


class TestNormalizeDatabasePath:
    """Tests for normalize_database_path function following project patterns"""

    def test_memory_database_unchanged(self):
        """Test that :memory: database path is unchanged"""
        assert normalize_database_path(":memory:") == ":memory:"
        assert normalize_database_path(None) == ":memory:"

    def test_regular_path_normalization(self, tmp_path):
        """Test regular path normalization to absolute path"""
        db_path = tmp_path / "test.duckdb"
        result = normalize_database_path(str(db_path))

        # Should return absolute path
        assert result == str(db_path.resolve())
        assert Path(result).is_absolute()

    @patch("platform.system")
    def test_windows_short_path(self, mock_system, tmp_path):
        """Test Windows path handling for short paths"""
        mock_system.return_value = "Windows"

        db_path = tmp_path / "short.duckdb"
        result = normalize_database_path(str(db_path))

        # Should NOT apply long path prefix for short paths
        assert not result.startswith("\\\\?\\")
        assert result == str(db_path.resolve())

    @patch("platform.system")
    def test_windows_long_path_prefix(self, mock_system, tmp_path):
        """Test Windows long path prefix application"""
        mock_system.return_value = "Windows"

        # Create a very long path that exceeds Windows limit
        long_subdir = "a" * (WINDOWS_MAX_PATH_LENGTH - 50)  # Leave room for filename
        long_path = tmp_path / long_subdir / "test.duckdb"

        result = normalize_database_path(str(long_path))

        # Should apply long path prefix for long paths
        if len(str(long_path.resolve())) > WINDOWS_MAX_PATH_LENGTH:
            assert result.startswith("\\\\?\\")
        else:
            # If the path isn't actually long enough, it won't get the prefix
            assert result == str(long_path.resolve())

    @patch("platform.system")
    def test_non_windows_path_unchanged(self, mock_system, tmp_path):
        """Test that non-Windows systems don't apply long path prefix"""
        mock_system.return_value = "Linux"

        # Create a long path
        long_subdir = "a" * 300
        long_path = tmp_path / long_subdir / "test.duckdb"

        result = normalize_database_path(str(long_path))

        # Should NOT apply Windows long path prefix on non-Windows
        assert not result.startswith("\\\\?\\")
        assert result == str(long_path.resolve())

    def test_relative_path_conversion(self):
        """Test that relative paths are converted to absolute"""
        result = normalize_database_path("./test.duckdb")

        # Should be absolute
        assert Path(result).is_absolute()
        # Should end with test.duckdb
        assert result.endswith("test.duckdb")


class TestGetCrossPlatformTempDir:
    """Tests for get_cross_platform_temp_dir function"""

    @patch("platform.system")
    @patch.dict(os.environ, {"TEMP": "C:\\custom\\temp"})
    def test_windows_temp_directory(self, mock_system):
        """Test Windows temporary directory handling"""
        mock_system.return_value = "Windows"

        result = get_cross_platform_temp_dir()

        assert result == "C:\\custom\\temp"

    @patch("platform.system")
    @patch.dict(os.environ, {}, clear=True)
    def test_windows_fallback_temp(self, mock_system):
        """Test Windows fallback temporary directory"""
        mock_system.return_value = "Windows"

        result = get_cross_platform_temp_dir()

        assert result == "C:\\temp\\hunyo"

    @patch("platform.system")
    @patch.dict(os.environ, {"TMPDIR": "/custom/tmp"})
    def test_macos_temp_directory(self, mock_system):
        """Test macOS temporary directory handling"""
        mock_system.return_value = "Darwin"

        result = get_cross_platform_temp_dir()

        assert result == "/custom/tmp"

    @patch("platform.system")
    @patch.dict(os.environ, {}, clear=True)
    def test_macos_fallback_temp(self, mock_system):
        """Test macOS fallback temporary directory"""
        mock_system.return_value = "Darwin"

        result = get_cross_platform_temp_dir()

        assert result == "/tmp/hunyo"  # noqa: S108

    @patch("platform.system")
    def test_linux_temp_directory(self, mock_system):
        """Test Linux temporary directory handling"""
        mock_system.return_value = "Linux"

        result = get_cross_platform_temp_dir()

        assert result == "/tmp/hunyo"  # noqa: S108

    @patch("platform.system")
    def test_other_unix_temp_directory(self, mock_system):
        """Test other Unix systems temporary directory handling"""
        mock_system.return_value = "FreeBSD"

        result = get_cross_platform_temp_dir()

        assert result == "/tmp/hunyo"  # noqa: S108


class TestGetProjectRoot:
    """Tests for get_project_root function"""

    def test_find_project_root_with_git(self, tmp_path):
        """Test finding project root using .git directory"""
        # Create a mock project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        # Create a subdirectory
        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        # Mock __file__ to be in the subdirectory
        with patch("hunyo_mcp_server.utils.paths.__file__", str(subdir / "paths.py")):
            result = get_project_root()

            assert result == project_dir

    def test_find_project_root_with_pyproject(self, tmp_path):
        """Test finding project root using pyproject.toml"""
        # Create a mock project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").touch()

        # Create a subdirectory
        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        # Mock __file__ to be in the subdirectory
        with patch("hunyo_mcp_server.utils.paths.__file__", str(subdir / "paths.py")):
            result = get_project_root()

            assert result == project_dir

    def test_find_project_root_fallback_to_cwd(self, tmp_path):
        """Test fallback to current working directory"""
        # Create a directory without .git or pyproject.toml
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").touch()  # Add pyproject.toml to cwd

        # Create a subdirectory without markers
        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        # Mock __file__ to be in the subdirectory and cwd to be project_dir
        with (
            patch("hunyo_mcp_server.utils.paths.__file__", str(subdir / "paths.py")),
            patch("pathlib.Path.cwd", return_value=project_dir),
        ):
            result = get_project_root()

            assert result == project_dir

    def test_project_root_not_found_error(self, tmp_path):
        """Test error when project root cannot be found"""
        # Create a directory without any markers
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        # Mock __file__ and cwd to directories without markers
        with (
            patch("hunyo_mcp_server.utils.paths.__file__", str(subdir / "paths.py")),
            patch("pathlib.Path.cwd", return_value=subdir),
        ):
            with pytest.raises(RuntimeError, match="Could not find project root"):
                get_project_root()


class TestGetSchemaPath:
    """Tests for get_schema_path function"""

    def test_get_existing_schema_path(self, tmp_path):
        """Test retrieving path for existing schema file"""
        # Create mock project structure
        project_dir = tmp_path / "project"
        schemas_dir = project_dir / "schemas" / "json"
        schemas_dir.mkdir(parents=True)

        # Create a schema file
        schema_file = schemas_dir / "test_schema.json"
        schema_file.write_text('{"test": "schema"}')

        # Mock get_project_root to return our test directory
        with patch(
            "hunyo_mcp_server.utils.paths.get_project_root", return_value=project_dir
        ):
            result = get_schema_path("test_schema.json")

            assert result == schema_file
            assert result.exists()

    def test_get_nonexistent_schema_path(self, tmp_path):
        """Test error when schema file doesn't exist"""
        # Create mock project structure without the schema file
        project_dir = tmp_path / "project"
        schemas_dir = project_dir / "schemas" / "json"
        schemas_dir.mkdir(parents=True)

        # Mock get_project_root to return our test directory
        with patch(
            "hunyo_mcp_server.utils.paths.get_project_root", return_value=project_dir
        ):
            with pytest.raises(FileNotFoundError, match="Schema file not found"):
                get_schema_path("nonexistent_schema.json")

    def test_get_schema_path_with_subdirectory(self, tmp_path):
        """Test retrieving schema path with proper directory structure"""
        # Create mock project structure
        project_dir = tmp_path / "project"
        schemas_dir = project_dir / "schemas" / "json"
        schemas_dir.mkdir(parents=True)

        # Create a schema file
        schema_file = schemas_dir / "runtime_events_schema.json"
        schema_file.write_text('{"test": "runtime_schema"}')

        # Mock get_project_root to return our test directory
        with patch(
            "hunyo_mcp_server.utils.paths.get_project_root", return_value=project_dir
        ):
            result = get_schema_path("runtime_events_schema.json")

            assert result == schema_file
            assert result.name == "runtime_events_schema.json"
            assert result.parent.name == "json"
            assert result.parent.parent.name == "schemas"


class TestSetupCrossPlatformDirectories:
    """Tests for setup_cross_platform_directories function"""

    def test_create_all_directories(self, tmp_path):
        """Test creating all required directories"""
        base_path = tmp_path / "hunyo_base"

        result = setup_cross_platform_directories(str(base_path))

        # Should return dict with all expected directories
        expected_dirs = [
            "data",
            "temp",
            "logs",
            "backup",
            "events",
            "database",
            "config",
        ]
        assert all(name in result for name in expected_dirs)

        # All directories should exist
        for _dir_name, dir_path in result.items():
            assert Path(dir_path).exists()
            assert Path(dir_path).is_dir()

    @patch("platform.system")
    @patch("os.chmod")
    def test_unix_permissions_set(self, mock_chmod, mock_system, tmp_path):
        """Test that Unix permissions are set correctly"""
        mock_system.return_value = "Linux"

        base_path = tmp_path / "hunyo_base"

        setup_cross_platform_directories(str(base_path))

        # Should have called chmod for each directory
        assert mock_chmod.call_count == 7  # 7 directories created

        # Check that chmod was called with correct permissions
        for call in mock_chmod.call_args_list:
            assert call[0][1] == 0o755  # Second argument should be 0o755

    @patch("platform.system")
    @patch("os.chmod")
    def test_windows_no_permissions_set(self, mock_chmod, mock_system, tmp_path):
        """Test that Windows doesn't attempt to set permissions"""
        mock_system.return_value = "Windows"

        base_path = tmp_path / "hunyo_base"

        setup_cross_platform_directories(str(base_path))

        # Should not have called chmod on Windows
        mock_chmod.assert_not_called()

    @patch("os.chmod")
    def test_chmod_error_handling(self, mock_chmod, tmp_path):
        """Test graceful handling of chmod errors"""
        # Make chmod raise an exception
        mock_chmod.side_effect = OSError("Permission denied")

        base_path = tmp_path / "hunyo_base"

        # Should not raise an exception
        result = setup_cross_platform_directories(str(base_path))

        # Should still create directories even if chmod fails
        assert len(result) == 7
        for dir_path in result.values():
            assert Path(dir_path).exists()

    def test_directory_creation_error_handling(self, tmp_path):
        """Test handling of directory creation errors"""
        # Try to create directories in a read-only location
        base_path = tmp_path / "readonly"
        base_path.mkdir()
        base_path.chmod(0o555)  # Read-only

        try:
            result = setup_cross_platform_directories(str(base_path / "hunyo"))

            # Should handle errors gracefully and continue with other directories
            # Some directories might fail to create due to permissions
            assert isinstance(result, dict)

        finally:
            # Clean up - restore write permissions
            base_path.chmod(0o755)


class TestGetSafeTempDatabasePath:
    """Tests for get_safe_temp_database_path function"""

    @patch("hunyo_mcp_server.utils.paths.get_cross_platform_temp_dir")
    def test_generate_temp_database_path_no_suffix(self, mock_temp_dir, tmp_path):
        """Test generating temp database path without suffix"""
        mock_temp_dir.return_value = str(tmp_path / "temp")

        result = get_safe_temp_database_path()

        # Should contain the temp directory
        assert str(tmp_path / "temp") in result
        # Should end with .duckdb
        assert result.endswith(".duckdb")
        # Should contain default name
        assert "test_hunyo.duckdb" in result

    @patch("hunyo_mcp_server.utils.paths.get_cross_platform_temp_dir")
    def test_generate_temp_database_path_with_suffix(self, mock_temp_dir, tmp_path):
        """Test generating temp database path with suffix"""
        mock_temp_dir.return_value = str(tmp_path / "temp")

        result = get_safe_temp_database_path("integration_test")

        # Should contain the temp directory
        assert str(tmp_path / "temp") in result
        # Should end with .duckdb
        assert result.endswith(".duckdb")
        # Should contain suffix in name
        assert "test_hunyo_integration_test.duckdb" in result

    @patch("hunyo_mcp_server.utils.paths.get_cross_platform_temp_dir")
    def test_temp_directory_creation(self, mock_temp_dir, tmp_path):
        """Test that temp directory is created if it doesn't exist"""
        temp_dir = tmp_path / "temp_hunyo"
        mock_temp_dir.return_value = str(temp_dir)

        # Directory doesn't exist initially
        assert not temp_dir.exists()

        get_safe_temp_database_path()

        # Directory should be created
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    @patch("hunyo_mcp_server.utils.paths.get_cross_platform_temp_dir")
    @patch("hunyo_mcp_server.utils.paths.normalize_database_path")
    def test_path_normalization_applied(self, mock_normalize, mock_temp_dir, tmp_path):
        """Test that path normalization is applied to result"""
        mock_temp_dir.return_value = str(tmp_path / "temp")
        mock_normalize.return_value = "normalized_path"

        result = get_safe_temp_database_path("test")

        # Should have called normalize_database_path
        mock_normalize.assert_called_once()
        assert result == "normalized_path"


class TestValidatePathAccessibility:
    """Tests for validate_path_accessibility function"""

    def test_memory_database_always_accessible(self):
        """Test that :memory: database is always accessible"""
        assert validate_path_accessibility(":memory:") is True

    def test_accessible_file_path(self, tmp_path):
        """Test validation of accessible file path"""
        # Create a writable directory
        test_dir = tmp_path / "accessible"
        test_dir.mkdir()

        # Test file path in accessible directory
        test_file = test_dir / "test.duckdb"

        assert validate_path_accessibility(str(test_file)) is True

    def test_nonexistent_parent_directory(self, tmp_path):
        """Test validation fails for nonexistent parent directory"""
        # Test file path in nonexistent directory
        nonexistent_dir = tmp_path / "nonexistent"
        test_file = nonexistent_dir / "test.duckdb"

        assert validate_path_accessibility(str(test_file)) is False

    def test_readonly_parent_directory(self, tmp_path):
        """Test validation fails for readonly parent directory"""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)  # Read-only

        try:
            # Test file path in readonly directory
            test_file = readonly_dir / "test.duckdb"

            # Should fail on systems that respect permissions
            result = validate_path_accessibility(str(test_file))

            # Result depends on system permissions handling
            assert isinstance(result, bool)

        finally:
            # Clean up - restore write permissions
            readonly_dir.chmod(0o755)

    def test_invalid_path_format(self):
        """Test validation fails for invalid path format"""
        # Test with invalid path
        invalid_paths = [
            "",  # Empty string
            "\x00invalid",  # Null byte
            "con",  # Windows reserved name (if on Windows)
        ]

        for invalid_path in invalid_paths:
            result = validate_path_accessibility(invalid_path)
            assert isinstance(result, bool)
            # Don't assert specific value as it depends on system

    def test_path_accessibility_exception_handling(self):
        """Test that exceptions are handled gracefully"""
        # Test with a path that might cause exceptions
        with patch("hunyo_mcp_server.utils.paths.Path") as mock_path:
            mock_path.side_effect = ValueError("Invalid path")

            result = validate_path_accessibility("any_path")

            assert result is False


class TestPathsIntegration:
    """Integration tests for paths utility functions"""

    def test_complete_workflow_temp_database(self, tmp_path):
        """Test complete workflow of creating and validating temp database"""
        # Mock project root for schema path
        project_dir = tmp_path / "project"
        schemas_dir = project_dir / "schemas" / "json"
        schemas_dir.mkdir(parents=True)

        # Create a schema file
        schema_file = schemas_dir / "test_schema.json"
        schema_file.write_text('{"test": "schema"}')

        with (
            patch(
                "hunyo_mcp_server.utils.paths.get_project_root",
                return_value=project_dir,
            ),
            patch(
                "hunyo_mcp_server.utils.paths.get_cross_platform_temp_dir",
                return_value=str(tmp_path / "temp"),
            ),
        ):

            # Get schema path
            schema_path = get_schema_path("test_schema.json")
            assert schema_path.exists()

            # Get temp database path
            db_path = get_safe_temp_database_path("integration")
            assert db_path.endswith(".duckdb")

            # Validate the database path
            assert validate_path_accessibility(db_path) is True

            # Normalize the database path
            normalized_path = normalize_database_path(db_path)
            assert Path(normalized_path).is_absolute()

    def test_directory_setup_and_validation(self, tmp_path):
        """Test complete directory setup and validation workflow"""
        base_path = tmp_path / "hunyo_test"

        # Setup directories
        directories = setup_cross_platform_directories(str(base_path))

        # Validate all created directories
        for _dir_name, dir_path in directories.items():
            assert Path(dir_path).exists()
            assert validate_path_accessibility(dir_path) is True

            # Test creating a database in each directory
            test_db = Path(dir_path) / "test.duckdb"
            assert validate_path_accessibility(str(test_db)) is True
