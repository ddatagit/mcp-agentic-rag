"""Contract tests for project structure migration."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.contract
@pytest.mark.migration
class TestMigrationToolContract:
    """Contract tests for reorganize_project_structure_tool."""

    def test_migration_tool_exists(self):
        """Test that migration tool exists in CLI module."""
        try:
            from cli.validate import reorganize_project_structure

            import inspect
            sig = inspect.signature(reorganize_project_structure)
            assert sig.return_annotation == dict, "Tool must return dict"

        except ImportError:
            pytest.fail("reorganize_project_structure not found - implement in cli/validate.py")

    def test_project_structure_validation(self):
        """Test that project structure validation works."""
        try:
            from cli.validate import validate_structure

            # This should work after migration
            result = validate_structure()
            assert isinstance(result, dict)
            assert "status" in result

        except ImportError:
            pytest.fail("validate_structure not found")

    def test_src_directory_exists(self):
        """Test that src directory structure is created correctly."""
        src_path = Path(__file__).parent.parent.parent / "src"

        # These directories should exist after migration
        expected_dirs = [
            src_path / "mcp_agentic_rag",
            src_path / "mcp_agentic_rag" / "models",
            src_path / "mcp_agentic_rag" / "services",
            src_path / "mcp_agentic_rag" / "server",
            src_path / "mcp_agentic_rag" / "config",
            src_path / "cli"
        ]

        for dir_path in expected_dirs:
            assert dir_path.exists(), f"Directory {dir_path} should exist after migration"
            assert dir_path.is_dir(), f"{dir_path} should be a directory"

    def test_init_files_exist(self):
        """Test that __init__.py files are created in all packages."""
        src_path = Path(__file__).parent.parent.parent / "src"

        expected_init_files = [
            src_path / "mcp_agentic_rag" / "__init__.py",
            src_path / "mcp_agentic_rag" / "models" / "__init__.py",
            src_path / "mcp_agentic_rag" / "services" / "__init__.py",
            src_path / "mcp_agentic_rag" / "server" / "__init__.py",
            src_path / "mcp_agentic_rag" / "config" / "__init__.py",
            src_path / "cli" / "__init__.py"
        ]

        for init_file in expected_init_files:
            assert init_file.exists(), f"__init__.py file {init_file} should exist"
            assert init_file.is_file(), f"{init_file} should be a file"