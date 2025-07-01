from __future__ import annotations

import re
from pathlib import Path

import pytest


class TestRoadmapValidation:
    """Tests for validating ROADMAP.md updates and research phase completion"""

    @pytest.fixture
    def roadmap_path(self) -> Path:
        """Returns path to ROADMAP.md file"""
        return Path(__file__).parent.parent / "ROADMAP.md"

    @pytest.fixture
    def roadmap_content(self, roadmap_path: Path) -> str:
        """Returns content of ROADMAP.md file"""
        try:
            # Try UTF-8 first (most common)
            return roadmap_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to system default encoding
            try:
                return roadmap_path.read_text(encoding='utf-8', errors='replace')
            except Exception:
                # Last resort - binary mode and decode with error handling
                return roadmap_path.read_bytes().decode('utf-8', errors='replace')

    def test_roadmap_file_exists(self, roadmap_path: Path):
        """Test that ROADMAP.md exists and is readable"""
        assert roadmap_path.exists(), "ROADMAP.md file should exist"
        assert roadmap_path.is_file(), "ROADMAP.md should be a file"
        assert roadmap_path.stat().st_size > 0, "ROADMAP.md should not be empty"

    def test_research_phase_marked_complete(self, roadmap_content: str):
        """Test that research phase is properly marked as complete"""
        # Look for the research phase completion markers
        assert "✅ **Research Phase Complete**" in roadmap_content
        assert "Comprehensive analysis of MCP implementation patterns" in roadmap_content
        assert "✅ **Architecture Research**" in roadmap_content
        assert "FastMCP framework patterns, CLI design, tool registration" in roadmap_content

    def test_integration_analysis_documented(self, roadmap_content: str):
        """Test that integration analysis is documented"""
        assert "✅ **Integration Analysis**" in roadmap_content
        assert "Capture layer architecture, event generation, DuckDB schema" in roadmap_content
        assert "✅ **Component Mapping**" in roadmap_content
        assert "Configuration system, data paths, testing infrastructure" in roadmap_content

    def test_next_steps_updated(self, roadmap_content: str):
        """Test that next steps section is properly updated"""
        # Check that research step is marked complete
        assert "✅ **Research `server.py` architecture**" in roadmap_content
        assert "MCP patterns, CLI design, component integration" in roadmap_content

        # Check that implementation step is present
        assert "**Implement `server.py`**" in roadmap_content
        assert "CLI that accepts --notebook parameter" in roadmap_content

    def test_phase_progress_updated(self, roadmap_content: str):
        """Test that Phase 1 progress tracking is updated"""
        # Should still show 3/4 complete but with updated current task
        assert "Phase 1 Progress: ✅ **3/4 items complete**" in roadmap_content
        assert "server.py research complete ✅, beginning CLI implementation" in roadmap_content

    def test_roadmap_structure_preserved(self, roadmap_content: str):
        """Test that overall roadmap structure is preserved"""
        # Key sections should still exist
        assert "🎯 Project Purpose" in roadmap_content
        assert "🏗️ Overarching Architecture" in roadmap_content
        assert "📊 Current Implementation Status" in roadmap_content
        assert "🗺️ Target Project Structure" in roadmap_content
        assert "🚀 Implementation Roadmap" in roadmap_content
        assert "📋 Next Immediate Steps" in roadmap_content

    def test_testing_status_maintained(self, roadmap_content: str):
        """Test that excellent testing status is maintained"""
        assert "✅ **70/70 tests passing** (100% success rate)" in roadmap_content
        assert "✅ **EXCELLENT QUALITY**" in roadmap_content

    @pytest.mark.parametrize("phase_item", [
        "✅ **Create `pyproject.toml`**",
        "✅ **Implement `config.py`**",
        "✅ **Create basic project structure**",
        "🚧 **IN PROGRESS - Create `src/hunyo_mcp_server/server.py`**"
    ])
    def test_phase_1_items_properly_marked(self, roadmap_content: str, phase_item: str):
        """Test that Phase 1 items have correct completion status"""
        if phase_item not in roadmap_content:
            # Better error message for debugging
            pytest.fail(f"Phase item '{phase_item}' not found in roadmap content. "
                       f"First 500 chars of content:\n{roadmap_content[:500]}")
        assert phase_item in roadmap_content

    def test_last_updated_date_present(self, roadmap_content: str):
        """Test that last updated date is present"""
        # Should have a date pattern like "Last Updated: 2024-12-29"
        date_pattern = r"Last Updated: \d{4}-\d{2}-\d{2}"
        assert re.search(date_pattern, roadmap_content), "Should have last updated date"


class TestServerImplementationPreparation:
    """Tests to validate readiness for server.py implementation"""

    def test_project_structure_ready_for_server(self):
        """Test that project structure is ready for server.py implementation"""
        src_dir = Path(__file__).parent.parent / "src" / "hunyo_mcp_server"

        # Core directories should exist
        assert src_dir.exists(), "hunyo_mcp_server package should exist"
        assert (src_dir / "__init__.py").exists(), "Package should be importable"
        assert (src_dir / "config.py").exists(), "Config module should exist"

        # Subdirectories should exist for server integration
        assert (src_dir / "ingestion").exists(), "Ingestion package should exist"
        assert (src_dir / "tools").exists(), "Tools package should exist"

    def test_capture_layer_importable(self):
        """Test that capture layer is properly importable for server integration"""
        import sys
        from pathlib import Path

        # Add src to path temporarily
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # These imports should work for server integration
            from capture.lightweight_runtime_tracker import LightweightRuntimeTracker
            from capture.live_lineage_interceptor import MarimoLiveInterceptor
            from capture.native_hooks_interceptor import MarimoNativeHooksInterceptor
            from capture.websocket_interceptor import MarimoWebSocketProxy

            # Test that classes are properly defined
            assert MarimoLiveInterceptor is not None
            assert LightweightRuntimeTracker is not None
            assert MarimoNativeHooksInterceptor is not None
            assert MarimoWebSocketProxy is not None

        except ImportError as e:
            pytest.fail(f"Capture layer should be importable for server integration: {e}")
        finally:
            # Clean up path
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_config_system_ready(self):
        """Test that configuration system is ready for server integration"""
        import sys
        from pathlib import Path

        # Add src to path temporarily
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from hunyo_mcp_server.config import HunyoConfig, get_hunyo_data_dir

            # Test config functions work
            data_dir = get_hunyo_data_dir()
            assert data_dir is not None
            assert isinstance(data_dir, Path)

            # Test config class is available
            assert HunyoConfig is not None

        except ImportError as e:
            pytest.fail(f"Config system should be ready for server integration: {e}")
        finally:
            # Clean up path
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_pyproject_entry_point_configured(self):
        """Test that pyproject.toml has correct CLI entry point for server"""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        content = pyproject_path.read_text()

        # Should have the CLI entry point configured
        assert "hunyo-mcp-server" in content
        assert "hunyo_mcp_server.server:main" in content

    def test_dependencies_available_for_server(self):
        """Test that required dependencies for server implementation are available"""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        # Key dependencies for MCP server should be listed
        required_deps = [
            "mcp",
            "click",
            "duckdb",
            "pandas",
            "watchdog"  # for file watching
        ]

        for dep in required_deps:
            assert dep in content, f"Required dependency {dep} should be in pyproject.toml"

    def test_schemas_available_for_server(self):
        """Test that database schemas are available for server initialization"""
        schemas_dir = Path(__file__).parent.parent / "schemas" / "sql"

        # Key schema files should exist
        assert (schemas_dir / "init_database.sql").exists()
        assert (schemas_dir / "runtime_events_table.sql").exists()
        assert (schemas_dir / "openlineage_events_table.sql").exists()

        # Views directory should exist
        assert (schemas_dir / "views").exists()
        assert (schemas_dir / "views" / "lineage_summary.sql").exists()
        assert (schemas_dir / "views" / "performance_metrics.sql").exists()
