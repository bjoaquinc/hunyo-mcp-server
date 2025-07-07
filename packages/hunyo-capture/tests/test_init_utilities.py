#!/usr/bin/env python3
"""
Test suite for the capture.__init__ utility functions
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from hunyo_capture import (
    capture_logger,
    get_event_filenames,
    get_notebook_file_hash,
    get_notebook_name,
    get_user_data_dir,
    hooks_logger,
    lineage_logger,
    runtime_logger,
)
from hunyo_capture.logger import HunyoLogger


class TestNotebookFileHash:
    """Test get_notebook_file_hash function"""

    def test_hash_generation_basic(self):
        """Test basic hash generation for simple paths"""
        # Test with absolute path
        hash1 = get_notebook_file_hash("/path/to/notebook.py")
        assert isinstance(hash1, str)
        assert len(hash1) == 8  # Should be 8-character hex

        # Test with relative path
        hash2 = get_notebook_file_hash("./notebook.py")
        assert isinstance(hash2, str)
        assert len(hash2) == 8

    def test_hash_consistency(self):
        """Test that same path produces same hash"""
        path = "/home/user/notebooks/analysis.py"
        hash1 = get_notebook_file_hash(path)
        hash2 = get_notebook_file_hash(path)
        assert hash1 == hash2

    def test_hash_uniqueness(self):
        """Test that different paths produce different hashes"""
        path1 = "/home/user/notebook1.py"
        path2 = "/home/user/notebook2.py"
        hash1 = get_notebook_file_hash(path1)
        hash2 = get_notebook_file_hash(path2)
        assert hash1 != hash2

    def test_hash_path_resolution(self):
        """Test that relative and absolute paths to same file produce same hash"""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "test_notebook.py"
            notebook_path.write_text("# Test notebook")

            # Create relative and absolute paths to the same file
            abs_path = str(notebook_path.resolve())

            # Change to temp directory and use relative path
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                rel_path = "test_notebook.py"

                hash_abs = get_notebook_file_hash(abs_path)
                hash_rel = get_notebook_file_hash(rel_path)

                # Should produce same hash since they resolve to same file
                assert hash_abs == hash_rel
            finally:
                os.chdir(original_cwd)

    def test_hash_with_special_characters(self):
        """Test hash generation with paths containing special characters"""
        special_paths = [
            "/path/with spaces/notebook.py",
            "/path/with-dashes/notebook.py",
            "/path/with_underscores/notebook.py",
            "/path/with.dots/notebook.py",
            "/path/with(parentheses)/notebook.py",
        ]

        hashes = []
        for path in special_paths:
            hash_val = get_notebook_file_hash(path)
            assert isinstance(hash_val, str)
            assert len(hash_val) == 8
            hashes.append(hash_val)

        # All hashes should be unique
        assert len(set(hashes)) == len(hashes)

    def test_hash_format_validation(self):
        """Test that hash is valid hexadecimal"""
        hash_val = get_notebook_file_hash("/test/notebook.py")
        # Should be valid hex string
        int(hash_val, 16)  # Will raise ValueError if not valid hex


class TestNotebookName:
    """Test get_notebook_name function"""

    def test_basic_name_extraction(self):
        """Test basic filename extraction without extension"""
        assert get_notebook_name("/path/to/notebook.py") == "notebook"
        assert get_notebook_name("simple.py") == "simple"
        assert (
            get_notebook_name("/complex/path/analysis_script.py") == "analysis_script"
        )

    def test_name_without_extension(self):
        """Test filename extraction for files without extension"""
        assert get_notebook_name("/path/to/notebook") == "notebook"
        assert get_notebook_name("simple") == "simple"

    def test_filesystem_safe_characters(self):
        """Test that problematic characters are replaced with underscores"""
        import platform

        test_cases = [
            ("note book.py", "note_book"),  # Space
            ("note-book.py", "note-book"),  # Dash (allowed)
            ("note_book.py", "note_book"),  # Underscore (allowed)
            ("note@book.py", "note_book"),  # Special character
            ("note#book$.py", "note_book_"),  # Multiple special characters
            ("note/book.py", "book"),  # Slash - only takes filename
            ("note:book.py", "note_book"),  # Colon
            ("note?book.py", "note_book"),  # Question mark
            ("note*book.py", "note_book"),  # Asterisk
            ("note|book.py", "note_book"),  # Pipe
            ("note<book>.py", "note_book_"),  # Angle brackets
        ]

        # Platform-specific test case for backslash
        if platform.system() == "Windows":
            # On Windows, backslash is path separator, so note\book.py -> book
            test_cases.append(("note\\book.py", "book"))
        else:
            # On Unix systems, backslash is treated as character
            test_cases.append(("note\\book.py", "note_book"))

        for input_path, expected_name in test_cases:
            actual_name = get_notebook_name(input_path)
            assert (
                actual_name == expected_name
            ), f"Input: {input_path}, Expected: {expected_name}, Got: {actual_name}"

    def test_alphanumeric_preservation(self):
        """Test that alphanumeric characters are preserved"""
        assert get_notebook_name("notebook123.py") == "notebook123"
        assert get_notebook_name("ABC123def.py") == "ABC123def"
        assert get_notebook_name("test_notebook_v2.py") == "test_notebook_v2"

    def test_edge_cases(self):
        """Test edge cases for name extraction"""
        assert get_notebook_name("") == ""
        assert (
            get_notebook_name(".py") == "_py"
        )  # .py becomes _py after character replacement
        assert get_notebook_name("/.py") == "_py"  # Same result
        assert get_notebook_name("/path/to/.hidden.py") == "_hidden"
        assert (
            get_notebook_name("file.with.multiple.dots.py") == "file_with_multiple_dots"
        )

    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        # Unicode characters are actually preserved (not replaced with underscores)
        assert get_notebook_name("café.py") == "café"
        assert get_notebook_name("naïve_approach.py") == "naïve_approach"


class TestEventFilenames:
    """Test get_event_filenames function"""

    def test_basic_filename_generation(self):
        """Test basic event filename generation"""
        notebook_path = "/path/to/test_notebook.py"
        data_dir = "/tmp/data"  # noqa: S108 # Test fixture using hardcoded path

        runtime_file, lineage_file, dataframe_lineage_file = get_event_filenames(
            notebook_path, data_dir
        )

        # Should return tuple of strings
        assert isinstance(runtime_file, str)
        assert isinstance(lineage_file, str)
        assert isinstance(dataframe_lineage_file, str)

        # Files should be in correct subdirectories (platform-aware)
        expected_runtime_path = Path(data_dir) / "events" / "runtime"
        expected_lineage_path = Path(data_dir) / "events" / "lineage"
        expected_dataframe_lineage_path = (
            Path(data_dir) / "events" / "dataframe_lineage"
        )

        assert str(expected_runtime_path) in runtime_file
        assert str(expected_lineage_path) in lineage_file
        assert str(expected_dataframe_lineage_path) in dataframe_lineage_file

        # Files should contain hash and notebook name
        hash_val = get_notebook_file_hash(notebook_path)
        notebook_name = get_notebook_name(notebook_path)

        assert hash_val in runtime_file
        assert notebook_name in runtime_file
        assert hash_val in lineage_file
        assert notebook_name in lineage_file
        assert hash_val in dataframe_lineage_file
        assert notebook_name in dataframe_lineage_file

        # Files should have correct extensions
        assert runtime_file.endswith("_runtime_events.jsonl")
        assert lineage_file.endswith("_lineage_events.jsonl")
        assert dataframe_lineage_file.endswith("_dataframe_lineage_events.jsonl")

    def test_filename_uniqueness(self):
        """Test that different notebooks get different filenames"""
        data_dir = "/tmp/data"  # noqa: S108 # Test fixture using hardcoded path

        runtime1, lineage1, dataframe_lineage1 = get_event_filenames(
            "/path/notebook1.py", data_dir
        )
        runtime2, lineage2, dataframe_lineage2 = get_event_filenames(
            "/path/notebook2.py", data_dir
        )

        # Different notebooks should have different filenames
        assert runtime1 != runtime2
        assert lineage1 != lineage2
        assert dataframe_lineage1 != dataframe_lineage2

    def test_filename_consistency(self):
        """Test that same notebook path consistently produces same filenames"""
        notebook_path = "/home/user/analysis.py"
        data_dir = "/data"

        runtime1, lineage1, dataframe_lineage1 = get_event_filenames(
            notebook_path, data_dir
        )
        runtime2, lineage2, dataframe_lineage2 = get_event_filenames(
            notebook_path, data_dir
        )

        assert runtime1 == runtime2
        assert lineage1 == lineage2
        assert dataframe_lineage1 == dataframe_lineage2

    def test_different_data_directories(self):
        """Test filename generation with different data directories"""
        notebook_path = "/notebooks/test.py"

        runtime1, lineage1, dataframe_lineage1 = get_event_filenames(
            notebook_path, "/data1"
        )
        runtime2, lineage2, dataframe_lineage2 = get_event_filenames(
            notebook_path, "/data2"
        )

        # Base filenames should be same, but paths should differ
        filename1 = Path(runtime1).name
        filename2 = Path(runtime2).name
        assert filename1 == filename2

        # Platform-aware path checking
        expected_path1 = str(Path("/data1") / "events" / "runtime")
        expected_path2 = str(Path("/data2") / "events" / "runtime")
        assert expected_path1 in runtime1
        assert expected_path2 in runtime2

        # Check dataframe lineage paths as well
        expected_df_path1 = str(Path("/data1") / "events" / "dataframe_lineage")
        expected_df_path2 = str(Path("/data2") / "events" / "dataframe_lineage")
        assert expected_df_path1 in dataframe_lineage1
        assert expected_df_path2 in dataframe_lineage2

    def test_path_object_handling(self):
        """Test that function works with Path objects"""
        notebook_path = Path("/notebooks/test.py")
        data_dir = Path("/data")

        runtime_file, lineage_file, dataframe_lineage_file = get_event_filenames(
            str(notebook_path), str(data_dir)
        )

        assert isinstance(runtime_file, str)
        assert isinstance(lineage_file, str)
        assert isinstance(dataframe_lineage_file, str)
        assert "test" in runtime_file
        assert "test" in lineage_file
        assert "test" in dataframe_lineage_file


class TestUserDataDir:
    """Test get_user_data_dir function"""

    def setUp(self):
        """Set up clean environment for each test"""
        # Store original environment
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_development_mode_env_variable(self):
        """Test development mode detection via environment variable"""
        with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                    data_dir = get_user_data_dir()
                    assert str(Path(temp_dir) / ".hunyo") == data_dir

    def test_production_mode_env_variable(self):
        """Test production mode enforcement via environment variable"""
        with patch.dict(os.environ, {"HUNYO_DEV_MODE": "0"}):
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path("/fake/home")
                with patch("pathlib.Path.mkdir"):
                    data_dir = get_user_data_dir()
                    expected_path = str(Path("/fake/home") / ".hunyo")
                    assert expected_path == data_dir

    def test_development_mode_git_detection(self):
        """Test development mode detection via .git directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .git directory to simulate git repo
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=True):  # Clear HUNYO_DEV_MODE
                    data_dir = get_user_data_dir()
                    assert str(Path(temp_dir) / ".hunyo") == data_dir

    def test_development_mode_pyproject_detection(self):
        """Test development mode detection via pyproject.toml"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pyproject.toml to simulate Python project
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text("[tool.poetry]")

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=True):
                    data_dir = get_user_data_dir()
                    assert str(Path(temp_dir) / ".hunyo") == data_dir

    def test_production_mode_no_dev_markers(self):
        """Test production mode when no development markers found"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory with no development markers
            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path("/fake/home")
                    with patch.dict(os.environ, {}, clear=True):
                        with patch("pathlib.Path.mkdir"):
                            data_dir = get_user_data_dir()
                            expected_path = str(Path("/fake/home") / ".hunyo")
                            assert expected_path == data_dir

    def test_directory_structure_creation(self):
        """Test that required directory structure is created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a subdirectory that doesn't exist yet
            test_data_dir = Path(temp_dir) / "test_hunyo_data"

            with patch("pathlib.Path.home", return_value=test_data_dir.parent):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "0"}):
                    data_dir = get_user_data_dir()

                    # Check that all required directories are created
                    data_path = Path(data_dir)
                    assert data_path.exists()
                    assert (data_path / "events" / "runtime").exists()
                    assert (data_path / "events" / "lineage").exists()
                    assert (data_path / "events" / "dataframe_lineage").exists()
                    assert (data_path / "database").exists()
                    assert (data_path / "config").exists()

    def test_existing_directory_handling(self):
        """Test behavior when .hunyo directory already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Pre-create .hunyo directory with some content
            hunyo_dir = Path(temp_dir) / ".hunyo"
            hunyo_dir.mkdir()
            existing_file = hunyo_dir / "existing.txt"
            existing_file.write_text("existing content")

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"HUNYO_DEV_MODE": "1"}):
                    data_dir = get_user_data_dir()

                    # Should return existing directory and preserve existing content
                    assert str(hunyo_dir) == data_dir
                    assert existing_file.exists()
                    assert existing_file.read_text() == "existing content"

                    # Should still create missing subdirectories
                    assert (hunyo_dir / "events" / "runtime").exists()
                    assert (hunyo_dir / "events" / "lineage").exists()
                    assert (hunyo_dir / "events" / "dataframe_lineage").exists()

    def test_parent_directory_search(self):
        """Test searching parent directories for development markers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            project_root = Path(temp_dir) / "project"
            sub_dir = project_root / "subdir" / "deep"
            sub_dir.mkdir(parents=True)

            # Put development marker in project root
            (project_root / "pyproject.toml").write_text("[tool.poetry]")

            with patch("pathlib.Path.cwd", return_value=sub_dir):
                with patch.dict(os.environ, {}, clear=True):
                    data_dir = get_user_data_dir()
                    # Should use project root, not deep subdirectory
                    assert str(project_root / ".hunyo") == data_dir


class TestLoggerInstances:
    """Test pre-created logger instances"""

    def test_logger_instances_exist(self):
        """Test that all expected logger instances are created"""
        assert isinstance(capture_logger, HunyoLogger)
        assert isinstance(runtime_logger, HunyoLogger)
        assert isinstance(lineage_logger, HunyoLogger)
        assert isinstance(hooks_logger, HunyoLogger)

    def test_logger_names(self):
        """Test that logger instances have correct names"""
        assert capture_logger.logger.name == "hunyo.capture"
        assert runtime_logger.logger.name == "hunyo.runtime"
        assert lineage_logger.logger.name == "hunyo.lineage"
        assert hooks_logger.logger.name == "hunyo.hooks"

    def test_logger_instances_are_configured(self):
        """Test that logger instances are properly configured"""
        for logger in [capture_logger, runtime_logger, lineage_logger, hooks_logger]:
            assert len(logger.logger.handlers) >= 1
            assert logger.logger.level == 20  # INFO level


class TestIntegrationScenarios:
    """Integration tests for utility functions working together"""

    def test_complete_workflow(self):
        """Test complete workflow from notebook path to event files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test notebook
            notebook_path = Path(temp_dir) / "test_analysis.py"
            notebook_path.write_text("import pandas as pd\ndf = pd.DataFrame()")

            # Set up data directory
            data_dir = Path(temp_dir) / "data"

            # Test complete workflow
            hash_val = get_notebook_file_hash(str(notebook_path))
            name = get_notebook_name(str(notebook_path))
            runtime_file, lineage_file, dataframe_lineage_file = get_event_filenames(
                str(notebook_path), str(data_dir)
            )

            # Verify consistency
            assert hash_val in runtime_file
            assert name in runtime_file
            assert hash_val in lineage_file
            assert name in lineage_file
            assert hash_val in dataframe_lineage_file
            assert name in dataframe_lineage_file

            # Verify paths
            assert str(data_dir / "events" / "runtime") in runtime_file
            assert str(data_dir / "events" / "lineage") in lineage_file
            assert (
                str(data_dir / "events" / "dataframe_lineage") in dataframe_lineage_file
            )

    def test_cross_platform_compatibility(self):
        """Test functions work on different path formats"""
        # Test with different path separators
        unix_path = "/home/user/notebook.py"

        # Both should work without errors
        unix_name = get_notebook_name(unix_path)

        # Unix path should extract "notebook"
        assert unix_name == "notebook"

        # Test different format that preserves filename
        simple_path = "notebook.py"
        simple_name = get_notebook_name(simple_path)
        assert simple_name == "notebook"

        # Test that paths with same filename produce same name (but different hashes)
        path1 = "/path1/notebook.py"
        path2 = "/path2/notebook.py"

        hash1 = get_notebook_file_hash(path1)
        hash2 = get_notebook_file_hash(path2)
        name1 = get_notebook_name(path1)
        name2 = get_notebook_name(path2)

        # Same filename should produce same name
        assert name1 == name2 == "notebook"

        # Different paths should produce different hashes
        assert hash1 != hash2

    def test_concurrent_access_safety(self):
        """Test that functions are safe for concurrent access"""
        import threading
        import time

        def worker(results, worker_id):
            for i in range(10):
                hash_val = get_notebook_file_hash(f"/path/notebook_{worker_id}_{i}.py")
                name = get_notebook_name(f"notebook_{worker_id}_{i}.py")
                results.append((hash_val, name))
                time.sleep(0.001)  # Small delay to encourage race conditions

        # Run multiple workers concurrently
        results = []
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(results, worker_id))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 30 results (3 workers × 10 iterations)
        assert len(results) == 30

        # All results should be valid
        for hash_val, name in results:
            assert isinstance(hash_val, str)
            assert len(hash_val) == 8
            assert isinstance(name, str)
            assert len(name) > 0
