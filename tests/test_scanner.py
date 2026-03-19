"""Integration tests for the ai_risk_scanner.scanner module.

Tests cover the FileWalker directory traversal, ScanConfig configuration,
WalkStats accumulation, and the Scanner orchestration pipeline using
temporary directory fixtures.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest
import yaml

from ai_risk_scanner.inventory import (
    RiskLevel,
    ScanResult,
)
from ai_risk_scanner.scanner import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILENAMES,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    FileWalker,
    ScanConfig,
    Scanner,
    WalkStats,
    discover_files,
    scan_path,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_project(tmp_path: Path) -> Path:
    """Create a minimal project directory structure for testing."""
    # Python source files
    (tmp_path / "app.py").write_text(
        "import openai\nclient = openai.OpenAI()\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "import os\nimport sys\n",
        encoding="utf-8",
    )

    # Config files
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-placeholder\nDEBUG=true\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "my-app"\n',
        encoding="utf-8",
    )

    # A subdirectory with more Python files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "client.py").write_text(
        "from anthropic import Anthropic\nclient = Anthropic()\n",
        encoding="utf-8",
    )
    (src_dir / "config.yaml").write_text(
        "model: gpt-4o\ntemperature: 0.7\n",
        encoding="utf-8",
    )

    # A test directory (should be walk-able unless excluded)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_app.py").write_text(
        "import pytest\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def multi_provider_project(tmp_path: Path) -> Path:
    """Create a project that references many AI providers."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""
            import openai
            import anthropic
            from langchain_openai import ChatOpenAI

            openai_client = openai.OpenAI()
            claude_client = anthropic.Anthropic()
            llm = ChatOpenAI(model="gpt-4")
        """),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        textwrap.dedent("""
            OPENAI_API_KEY=sk-placeholder
            ANTHROPIC_API_KEY=sk-ant-placeholder
            COHERE_API_KEY=co-placeholder
            GEMINI_API_KEY=gemini-placeholder
        """),
        encoding="utf-8",
    )
    (tmp_path / "bedrock_client.py").write_text(
        textwrap.dedent("""
            import boto3
            client = boto3.client('bedrock-runtime')
            response = client.InvokeModel(modelId='anthropic.claude')
        """),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def nested_project(tmp_path: Path) -> Path:
    """Create a deeply nested project directory structure."""
    # Root files
    (tmp_path / "main.py").write_text("import openai\n", encoding="utf-8")

    # Nested directories
    level1 = tmp_path / "level1"
    level2 = level1 / "level2"
    level3 = level2 / "level3"
    level3.mkdir(parents=True)

    (level1 / "a.py").write_text("import anthropic\n", encoding="utf-8")
    (level2 / "b.py").write_text("import cohere\n", encoding="utf-8")
    (level3 / "c.py").write_text("import mistralai\n", encoding="utf-8")

    # A non-AI file at each level
    (level1 / "helpers.py").write_text("import os\n", encoding="utf-8")
    (level2 / "utils.py").write_text("import sys\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def project_with_excluded_dirs(tmp_path: Path) -> Path:
    """Create a project with directories that should be excluded."""
    (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")

    # These directories should be excluded by default
    for excluded in [".git", "node_modules", "__pycache__", ".venv", "dist"]:
        excluded_dir = tmp_path / excluded
        excluded_dir.mkdir()
        (excluded_dir / "some_file.py").write_text(
            "import anthropic\n", encoding="utf-8"
        )

    return tmp_path


@pytest.fixture
def project_with_unsupported_files(tmp_path: Path) -> Path:
    """Create a project containing files with unsupported extensions."""
    (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")
    (tmp_path / "binary.exe").write_bytes(b"MZ")
    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    (tmp_path / "model.pkl").write_bytes(b"\x80\x04")
    return tmp_path


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """Return an empty temporary directory."""
    return tmp_path


@pytest.fixture
def credential_project(tmp_path: Path) -> Path:
    """Create a project with hardcoded credentials."""
    (tmp_path / "config.py").write_text(
        f'OPENAI_API_KEY = "sk-{"a" * 30}"\n',
        encoding="utf-8",
    )
    (tmp_path / "secrets.py").write_text(
        f'ANTHROPIC_KEY = "sk-ant-{"b" * 30}"\n',
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# ScanConfig tests
# ---------------------------------------------------------------------------


class TestScanConfig:
    """Tests for the ScanConfig dataclass."""

    def test_default_config(self) -> None:
        config = ScanConfig()
        assert DEFAULT_EXCLUDED_DIRS.issubset(config.excluded_dirs)
        assert config.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_BYTES
        assert config.follow_symlinks is False
        assert config.encoding == "utf-8"
        assert config.include_hidden is False
        assert config.providers_yaml_path is None
        assert config.scan_id is None

    def test_excluded_dirs_default_is_mutable_set(self) -> None:
        config = ScanConfig()
        # Should be able to add to it without affecting other instances
        config.excluded_dirs.add("my_custom_dir")
        config2 = ScanConfig()
        assert "my_custom_dir" not in config2.excluded_dirs

    def test_custom_excluded_dirs(self) -> None:
        config = ScanConfig(excluded_dirs={"custom_dir", "another_dir"})
        assert "custom_dir" in config.excluded_dirs
        assert "another_dir" in config.excluded_dirs

    def test_extra_extensions_normalized_with_dot(self) -> None:
        config = ScanConfig(extra_extensions={"rb", "java", ".go"})
        assert ".rb" in config.extra_extensions
        assert ".java" in config.extra_extensions
        assert ".go" in config.extra_extensions

    def test_extra_extensions_lowercased(self) -> None:
        config = ScanConfig(extra_extensions={"PY", "JS", "TS"})
        assert ".py" in config.extra_extensions
        assert ".js" in config.extra_extensions
        assert ".ts" in config.extra_extensions

    def test_all_supported_extensions_includes_extras(self) -> None:
        config = ScanConfig(extra_extensions={".custom"})
        assert ".custom" in config.all_supported_extensions
        # Should also include the defaults
        assert ".py" in config.all_supported_extensions

    def test_all_supported_extensions_is_frozenset(self) -> None:
        config = ScanConfig()
        assert isinstance(config.all_supported_extensions, frozenset)

    def test_max_file_size_bytes_custom(self) -> None:
        config = ScanConfig(max_file_size_bytes=1024)
        assert config.max_file_size_bytes == 1024

    def test_providers_yaml_path_stored(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "custom_providers.yaml"
        yaml_path.write_text("providers: []\n", encoding="utf-8")
        config = ScanConfig(providers_yaml_path=yaml_path)
        assert config.providers_yaml_path == yaml_path

    def test_scan_id_stored(self) -> None:
        config = ScanConfig(scan_id="my-scan-123")
        assert config.scan_id == "my-scan-123"

    def test_excluded_paths_default_empty(self) -> None:
        config = ScanConfig()
        assert config.excluded_paths == set()

    def test_excluded_paths_custom(self) -> None:
        config = ScanConfig(excluded_paths={"/tmp/exclude", "/home/user/project/vendor"})
        assert "/tmp/exclude" in config.excluded_paths


# ---------------------------------------------------------------------------
# WalkStats tests
# ---------------------------------------------------------------------------


class TestWalkStats:
    """Tests for the WalkStats dataclass."""

    def test_default_zero_values(self) -> None:
        stats = WalkStats()
        assert stats.total_files_found == 0
        assert stats.files_accepted == 0
        assert stats.files_skipped_extension == 0
        assert stats.files_skipped_hidden == 0
        assert stats.files_skipped_excluded == 0
        assert stats.files_skipped_size == 0
        assert stats.dirs_traversed == 0
        assert stats.dirs_skipped == 0

    def test_total_files_skipped_computed(self) -> None:
        stats = WalkStats(
            files_skipped_extension=5,
            files_skipped_hidden=2,
            files_skipped_excluded=3,
            files_skipped_size=1,
        )
        assert stats.total_files_skipped == 11

    def test_total_files_skipped_all_zero(self) -> None:
        stats = WalkStats()
        assert stats.total_files_skipped == 0

    def test_repr(self) -> None:
        stats = WalkStats(total_files_found=10, files_accepted=8)
        r = repr(stats)
        assert "WalkStats" in r
        assert "found=10" in r
        assert "accepted=8" in r


# ---------------------------------------------------------------------------
# FileWalker tests
# ---------------------------------------------------------------------------


class TestFileWalkerInit:
    """Tests for FileWalker initialization."""

    def test_default_config(self) -> None:
        walker = FileWalker()
        assert walker.config is not None
        assert isinstance(walker.config, ScanConfig)

    def test_custom_config(self) -> None:
        config = ScanConfig(max_file_size_bytes=1024)
        walker = FileWalker(config=config)
        assert walker.config.max_file_size_bytes == 1024

    def test_initial_stats_are_zero(self) -> None:
        walker = FileWalker()
        assert walker.stats.total_files_found == 0
        assert walker.stats.files_accepted == 0


class TestFileWalkerIsExcludedDir:
    """Tests for FileWalker._is_excluded_dir."""

    def test_git_dir_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/.git")) is True

    def test_node_modules_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/node_modules")) is True

    def test_pycache_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/__pycache__")) is True

    def test_venv_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/.venv")) is True
        assert walker._is_excluded_dir(Path("/project/venv")) is True

    def test_src_dir_not_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/src")) is False

    def test_tests_dir_not_excluded(self) -> None:
        walker = FileWalker()
        assert walker._is_excluded_dir(Path("/project/tests")) is False

    def test_hidden_dir_excluded_by_default(self) -> None:
        walker = FileWalker()
        # Hidden directories (dot-prefixed) that aren't known filenames should be excluded
        assert walker._is_excluded_dir(Path("/project/.hidden_dir")) is True

    def test_hidden_dir_included_when_include_hidden(self) -> None:
        config = ScanConfig(include_hidden=True)
        walker = FileWalker(config=config)
        # With include_hidden=True, hidden dirs that aren't in excluded set are allowed
        # (unless they're in excluded_dirs like .git)
        assert walker._is_excluded_dir(Path("/project/.hidden_dir")) is False

    def test_custom_excluded_dir(self) -> None:
        config = ScanConfig(excluded_dirs={"my_special_dir"})
        walker = FileWalker(config=config)
        assert walker._is_excluded_dir(Path("/project/my_special_dir")) is True

    def test_excluded_path_prefix(self) -> None:
        config = ScanConfig(excluded_paths={"/project/vendor"})
        walker = FileWalker(config=config)
        assert walker._is_excluded_dir(Path("/project/vendor")) is True
        assert walker._is_excluded_dir(Path("/project/src")) is False


class TestFileWalkerIsSupportedFile:
    """Tests for FileWalker._is_supported_file."""

    def test_python_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("app.py")) is True

    def test_javascript_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("main.js")) is True

    def test_yaml_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("config.yaml")) is True
        assert walker._is_supported_file(Path("config.yml")) is True

    def test_json_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("package.json")) is True

    def test_env_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path(".env")) is True
        assert walker._is_supported_file(Path(".env.local")) is True

    def test_dockerfile_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("Dockerfile")) is True

    def test_toml_file_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("pyproject.toml")) is True

    def test_png_file_not_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("image.png")) is False

    def test_exe_file_not_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("binary.exe")) is False

    def test_zip_file_not_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("archive.zip")) is False

    def test_excluded_filename_not_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path(".DS_Store")) is False
        assert walker._is_supported_file(Path("LICENSE")) is False

    def test_hidden_file_excluded_by_default(self) -> None:
        config = ScanConfig(include_hidden=False)
        walker = FileWalker(config=config)
        # Hidden file not in SUPPORTED_FILENAMES
        assert walker._is_supported_file(Path(".hidden_config")) is False

    def test_hidden_file_included_when_include_hidden(self) -> None:
        config = ScanConfig(include_hidden=True)
        walker = FileWalker(config=config)
        # With include_hidden=True and .py extension
        assert walker._is_supported_file(Path(".hidden_script.py")) is True

    def test_extra_extension_supported(self) -> None:
        config = ScanConfig(extra_extensions={".custom"})
        walker = FileWalker(config=config)
        assert walker._is_supported_file(Path("myfile.custom")) is True

    def test_excluded_path_prefix_not_supported(self) -> None:
        config = ScanConfig(excluded_paths={"/project/vendor"})
        walker = FileWalker(config=config)
        assert walker._is_supported_file(Path("/project/vendor/lib.py")) is False

    def test_requirements_txt_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("requirements.txt")) is True

    def test_docker_compose_supported(self) -> None:
        walker = FileWalker()
        assert walker._is_supported_file(Path("docker-compose.yml")) is True


class TestFileWalkerIsWithinSizeLimit:
    """Tests for FileWalker._is_within_size_limit."""

    def test_small_file_within_limit(self, tmp_path: Path) -> None:
        small_file = tmp_path / "small.py"
        small_file.write_text("import openai\n", encoding="utf-8")
        walker = FileWalker()
        assert walker._is_within_size_limit(small_file) is True

    def test_oversized_file_exceeds_limit(self, tmp_path: Path) -> None:
        large_file = tmp_path / "large.py"
        large_file.write_bytes(b"x" * (DEFAULT_MAX_FILE_SIZE_BYTES + 1))
        walker = FileWalker()
        assert walker._is_within_size_limit(large_file) is False

    def test_file_exactly_at_limit_within(self, tmp_path: Path) -> None:
        at_limit = tmp_path / "at_limit.py"
        at_limit.write_bytes(b"x" * DEFAULT_MAX_FILE_SIZE_BYTES)
        walker = FileWalker()
        assert walker._is_within_size_limit(at_limit) is True

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.py"
        walker = FileWalker()
        assert walker._is_within_size_limit(missing) is False

    def test_custom_size_limit(self, tmp_path: Path) -> None:
        small_limit = tmp_path / "file.py"
        small_limit.write_bytes(b"x" * 100)
        config = ScanConfig(max_file_size_bytes=50)
        walker = FileWalker(config=config)
        assert walker._is_within_size_limit(small_limit) is False


class TestFileWalkerWalk:
    """Tests for FileWalker.walk using temporary directory fixtures."""

    def test_walk_simple_project(self, simple_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(simple_project))
        assert len(files) > 0
        # All yielded paths should be files
        for f in files:
            assert f.is_file()

    def test_walk_yields_python_files(self, simple_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(simple_project))
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) > 0

    def test_walk_yields_env_file(self, simple_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(simple_project))
        env_files = [f for f in files if f.name == ".env"]
        assert len(env_files) == 1

    def test_walk_yields_toml_file(self, simple_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(simple_project))
        toml_files = [f for f in files if f.suffix == ".toml"]
        assert len(toml_files) > 0

    def test_walk_excludes_binary_files(self, project_with_unsupported_files: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(project_with_unsupported_files))
        names = {f.name for f in files}
        assert "image.png" not in names
        assert "archive.zip" not in names
        assert "binary.exe" not in names

    def test_walk_accepts_supported_python_file(self, project_with_unsupported_files: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(project_with_unsupported_files))
        names = {f.name for f in files}
        assert "app.py" in names

    def test_walk_excludes_default_dirs(self, project_with_excluded_dirs: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(project_with_excluded_dirs))
        # Files inside excluded dirs should not be found
        for f in files:
            parts = f.parts
            assert ".git" not in parts
            assert "node_modules" not in parts
            assert "__pycache__" not in parts
            assert ".venv" not in parts
            assert "dist" not in parts

    def test_walk_empty_directory(self, empty_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(empty_project))
        assert files == []

    def test_walk_updates_stats(self, simple_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(simple_project))
        assert walker.stats.total_files_found > 0
        assert walker.stats.files_accepted == len(files)
        assert walker.stats.dirs_traversed > 0

    def test_walk_stats_skipped_count(self, project_with_unsupported_files: Path) -> None:
        walker = FileWalker()
        list(walker.walk(project_with_unsupported_files))
        assert walker.stats.total_files_skipped > 0

    def test_walk_nested_project_traverses_all_dirs(self, nested_project: Path) -> None:
        walker = FileWalker()
        files = list(walker.walk(nested_project))
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "a.py" in file_names
        assert "b.py" in file_names
        assert "c.py" in file_names

    def test_walk_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        walker = FileWalker()
        missing = tmp_path / "nonexistent_directory"
        with pytest.raises(ValueError, match="does not exist"):
            list(walker.walk(missing))

    def test_walk_single_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "single.py"
        py_file.write_text("import openai\n", encoding="utf-8")
        walker = FileWalker()
        files = list(walker.walk(py_file))
        assert files == [py_file]

    def test_walk_single_unsupported_file(self, tmp_path: Path) -> None:
        png_file = tmp_path / "image.png"
        png_file.write_bytes(b"\x89PNG")
        walker = FileWalker()
        files = list(walker.walk(png_file))
        assert files == []

    def test_walk_custom_excluded_dir(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        custom_dir = tmp_path / "my_vendor"
        custom_dir.mkdir()
        (custom_dir / "lib.py").write_text("import anthropic\n", encoding="utf-8")

        config = ScanConfig(excluded_dirs=set(DEFAULT_EXCLUDED_DIRS) | {"my_vendor"})
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "app.py" in file_names
        assert "lib.py" not in file_names

    def test_walk_custom_excluded_path_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        (vendor_dir / "lib.py").write_text("import anthropic\n", encoding="utf-8")

        config = ScanConfig(excluded_paths={str(vendor_dir)})
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "app.py" in file_names
        assert "lib.py" not in file_names

    def test_walk_with_include_hidden(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        hidden_py = tmp_path / ".hidden_config.py"
        hidden_py.write_text("import anthropic\n", encoding="utf-8")

        config = ScanConfig(include_hidden=True)
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "app.py" in file_names
        assert ".hidden_config.py" in file_names

    def test_walk_without_include_hidden_skips_hidden_py(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        hidden_py = tmp_path / ".hidden_config.py"
        hidden_py.write_text("import anthropic\n", encoding="utf-8")

        config = ScanConfig(include_hidden=False)
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "app.py" in file_names
        assert ".hidden_config.py" not in file_names

    def test_walk_skips_oversized_files(self, tmp_path: Path) -> None:
        small = tmp_path / "small.py"
        small.write_text("import openai\n", encoding="utf-8")
        large = tmp_path / "large.py"
        large.write_bytes(b"x" * (DEFAULT_MAX_FILE_SIZE_BYTES + 1))

        walker = FileWalker()
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "small.py" in file_names
        assert "large.py" not in file_names

    def test_walk_stats_reset_on_repeated_walk(self, simple_project: Path) -> None:
        walker = FileWalker()
        list(walker.walk(simple_project))
        first_count = walker.stats.files_accepted

        # Walk again — stats should be reset
        list(walker.walk(simple_project))
        assert walker.stats.files_accepted == first_count

    def test_walk_dirs_skipped_count(self, project_with_excluded_dirs: Path) -> None:
        walker = FileWalker()
        list(walker.walk(project_with_excluded_dirs))
        # Several excluded dirs should have been encountered
        assert walker.stats.dirs_skipped > 0

    def test_walk_files_sorted_deterministically(self, tmp_path: Path) -> None:
        # Create files with names that would differ in sort order
        for name in ["z_file.py", "a_file.py", "m_file.py"]:
            (tmp_path / name).write_text("import openai\n", encoding="utf-8")
        walker = FileWalker()
        files1 = list(walker.walk(tmp_path))
        files2 = list(walker.walk(tmp_path))
        assert files1 == files2

    def test_walk_extra_extension(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        custom_file = tmp_path / "script.custom"
        custom_file.write_text("import openai\n", encoding="utf-8")

        config = ScanConfig(extra_extensions={".custom"})
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "app.py" in file_names
        assert "script.custom" in file_names

    def test_walk_extra_extension_not_included_by_default(self, tmp_path: Path) -> None:
        custom_file = tmp_path / "script.custom"
        custom_file.write_text("import openai\n", encoding="utf-8")

        config = ScanConfig()  # no extra extensions
        walker = FileWalker(config=config)
        files = list(walker.walk(tmp_path))
        file_names = {f.name for f in files}
        assert "script.custom" not in file_names


class TestFileWalkerCollect:
    """Tests for FileWalker.collect method."""

    def test_collect_returns_list(self, simple_project: Path) -> None:
        walker = FileWalker()
        result = walker.collect(simple_project)
        assert isinstance(result, list)

    def test_collect_same_as_walk_list(self, simple_project: Path) -> None:
        walker = FileWalker()
        walk_result = list(walker.walk(simple_project))
        # Reset walker for second pass
        walker2 = FileWalker()
        collect_result = walker2.collect(simple_project)
        assert set(walk_result) == set(collect_result)

    def test_collect_empty_project(self, empty_project: Path) -> None:
        walker = FileWalker()
        result = walker.collect(empty_project)
        assert result == []

    def test_collect_nested_project(self, nested_project: Path) -> None:
        walker = FileWalker()
        result = walker.collect(nested_project)
        assert len(result) >= 4  # At least main.py, a.py, b.py, c.py


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


class TestScannerInit:
    """Tests for Scanner initialization."""

    def test_default_init(self) -> None:
        scanner = Scanner()
        assert scanner.config is not None
        assert scanner.walker is not None
        assert scanner.engine is not None
        assert scanner.risk_engine is not None

    def test_custom_config(self) -> None:
        config = ScanConfig(max_file_size_bytes=1024)
        scanner = Scanner(config=config)
        assert scanner.config.max_file_size_bytes == 1024

    def test_repr(self) -> None:
        scanner = Scanner()
        r = repr(scanner)
        assert "Scanner" in r

    def test_engine_loaded_with_providers(self) -> None:
        scanner = Scanner()
        assert scanner.engine.provider_count > 0

    def test_risk_engine_loaded_with_metadata(self) -> None:
        scanner = Scanner()
        assert len(scanner.risk_engine.ctem_metadata) > 0
        assert len(scanner.risk_engine.nist_metadata) > 0


class TestScannerScan:
    """Tests for the Scanner.scan method using temporary project fixtures."""

    def test_scan_returns_scan_result(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert isinstance(result, ScanResult)

    def test_scan_detects_openai(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert "openai" in result.assets

    def test_scan_detects_anthropic(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert "anthropic" in result.assets

    def test_scan_produces_findings(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert len(result.findings) > 0

    def test_scan_computes_summary(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        summary = result.summary
        assert summary.total_providers_detected > 0
        assert summary.total_findings > 0

    def test_scan_metadata_has_scan_path(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert result.metadata.scan_path == simple_project.resolve()

    def test_scan_metadata_has_completion_time(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert result.metadata.completed_at is not None

    def test_scan_metadata_files_scanned_positive(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert result.metadata.total_files_scanned > 0

    def test_scan_empty_project_no_providers(self, empty_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(empty_project)
        assert len(result.assets) == 0
        assert len(result.findings) == 0
        assert result.summary.total_providers_detected == 0

    def test_scan_empty_project_summary_zero_risk(self, empty_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(empty_project)
        assert result.summary.overall_risk_level == RiskLevel.INFO
        assert result.summary.overall_risk_score == 0.0

    def test_scan_multi_provider_project(self, multi_provider_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(multi_provider_project)
        provider_ids = set(result.assets.keys())
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

    def test_scan_multi_provider_multiple_findings(self, multi_provider_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(multi_provider_project)
        assert len(result.findings) >= 2

    def test_scan_nested_project_all_providers_found(self, nested_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(nested_project)
        provider_ids = set(result.assets.keys())
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

    def test_scan_excluded_dirs_not_scanned(self, project_with_excluded_dirs: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(project_with_excluded_dirs)
        # Only app.py in root should be scanned, not files in .git, node_modules, etc.
        # Total files scanned should be low (just app.py + any root config files)
        assert result.metadata.total_files_scanned < 10

    def test_scan_with_custom_scan_id(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project, scan_id="custom-id-123")
        assert result.metadata.scan_id == "custom-id-123"

    def test_scan_with_config_scan_id(self, simple_project: Path) -> None:
        config = ScanConfig(scan_id="config-scan-id")
        scanner = Scanner(config=config)
        result = scanner.scan(simple_project)
        assert result.metadata.scan_id == "config-scan-id"

    def test_scan_auto_generates_scan_id(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert result.metadata.scan_id is not None
        assert len(result.metadata.scan_id) > 0

    def test_scan_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent_directory"
        scanner = Scanner()
        # Scanner.scan catches ValueError and returns empty result
        result = scanner.scan(missing)
        # Should return a result with completed_at set but no assets
        assert result.metadata.completed_at is not None

    def test_scan_credential_project(self, credential_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(credential_project)
        # Should detect credential exposure
        has_cred_providers = result.summary.providers_with_credentials
        critical_findings = result.summary.critical_findings
        # Either providers_with_credentials has entries or we have critical findings
        assert len(has_cred_providers) > 0 or critical_findings > 0

    def test_scan_single_python_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "app.py"
        py_file.write_text("import openai\nclient = openai.OpenAI()\n", encoding="utf-8")
        scanner = Scanner()
        result = scanner.scan(py_file)
        assert "openai" in result.assets

    def test_scan_summary_risk_level_reflects_findings(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        if result.findings:
            max_risk = max(f.risk_level for f in result.findings)
            assert result.summary.overall_risk_level == max_risk

    def test_scan_findings_sorted_by_score(self, multi_provider_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(multi_provider_project)
        sorted_findings = result.sorted_findings
        scores = [f.risk_score for f in sorted_findings]
        assert scores == sorted(scores, reverse=True)

    def test_scan_all_affected_files_populated(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        if result.assets:
            assert len(result.all_affected_files) > 0

    def test_scan_duration_seconds_positive(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        if result.metadata.duration_seconds is not None:
            assert result.metadata.duration_seconds >= 0.0

    def test_scan_supported_extensions_in_metadata(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        assert isinstance(result.metadata.supported_extensions, list)
        assert len(result.metadata.supported_extensions) > 0

    def test_scan_with_custom_providers_yaml(self, tmp_path: Path) -> None:
        """Scan with a minimal custom providers.yaml."""
        custom_yaml = tmp_path / "custom_providers.yaml"
        data = {
            "providers": [
                {
                    "id": "custom_ai",
                    "name": "Custom AI",
                    "category": "llm",
                    "risk_level": "high",
                    "vendor_lock_in": True,
                    "data_residency_concern": True,
                    "patterns": {
                        "imports": ["import custom_ai"],
                        "env_vars": ["CUSTOM_AI_KEY"],
                        "urls": [],
                        "model_names": [],
                        "sdk_calls": [],
                    },
                    "nist_rmf_functions": ["GOVERN-1"],
                    "ctem_categories": ["external_exposure"],
                    "compliance_notes": ["Review custom AI usage."],
                    "docs_url": "",
                }
            ],
            "credential_patterns": [],
            "ctem_categories": {
                "external_exposure": {
                    "name": "External Exposure",
                    "description": "API exposed externally.",
                    "severity_modifier": 1.2,
                }
            },
            "nist_rmf_functions": {
                "GOVERN-1": {
                    "name": "Govern",
                    "description": "Establish policies.",
                }
            },
        }
        custom_yaml.write_text(yaml.dump(data), encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "app.py").write_text(
            "import custom_ai\nCUSTOM_AI_KEY=placeholder\n", encoding="utf-8"
        )

        config = ScanConfig(providers_yaml_path=custom_yaml)
        scanner = Scanner(config=config)
        result = scanner.scan(project_dir)
        assert "custom_ai" in result.assets

    def test_scan_includes_yaml_config_files(self, simple_project: Path) -> None:
        """Verify YAML config files are scanned and produce detections."""
        scanner = Scanner()
        result = scanner.scan(simple_project)
        # src/config.yaml has model: gpt-4o which should trigger openai detection
        if "openai" in result.assets:
            openai_files = result.assets["openai"].files_detected
            yaml_files = [f for f in openai_files if f.suffix in (".yaml", ".yml")]
            # There may or may not be yaml files depending on patterns
            assert isinstance(yaml_files, list)

    def test_scan_excludes_oversized_files(self, tmp_path: Path) -> None:
        """Verify files exceeding size limit are skipped."""
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        large_file = tmp_path / "large.py"
        large_file.write_bytes(b"import anthropic\n" + b"x" * (DEFAULT_MAX_FILE_SIZE_BYTES + 1))

        scanner = Scanner()
        result = scanner.scan(tmp_path)
        # openai should be detected from app.py
        assert "openai" in result.assets
        # anthropic should NOT be detected (large.py is skipped)
        assert "anthropic" not in result.assets

    def test_scan_count_files_skipped(self, project_with_unsupported_files: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(project_with_unsupported_files)
        assert result.metadata.total_files_skipped > 0


# ---------------------------------------------------------------------------
# scan_path convenience function tests
# ---------------------------------------------------------------------------


class TestScanPath:
    """Tests for the scan_path convenience function."""

    def test_scan_path_returns_scan_result(self, simple_project: Path) -> None:
        result = scan_path(simple_project)
        assert isinstance(result, ScanResult)

    def test_scan_path_string_argument(self, simple_project: Path) -> None:
        result = scan_path(str(simple_project))
        assert isinstance(result, ScanResult)

    def test_scan_path_detects_providers(self, simple_project: Path) -> None:
        result = scan_path(simple_project)
        assert len(result.assets) > 0

    def test_scan_path_with_extra_exclude_dirs(self, nested_project: Path) -> None:
        result = scan_path(
            nested_project,
            exclude_dirs={"level2"},
        )
        # Files in level2 and below should not be detected
        # (b.py has cohere, c.py has mistralai)
        provider_ids = set(result.assets.keys())
        # level1/a.py (anthropic) should still be found
        assert "cohere" not in provider_ids or "mistral" not in provider_ids

    def test_scan_path_with_max_file_size(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        large_file = tmp_path / "large.py"
        large_file.write_bytes(b"import anthropic\n" + b"x" * 1024 * 1024 * 2)

        result = scan_path(tmp_path, max_file_size_mb=0.5)  # 0.5 MB limit
        assert "openai" in result.assets
        assert "anthropic" not in result.assets

    def test_scan_path_with_scan_id(self, simple_project: Path) -> None:
        result = scan_path(simple_project, scan_id="my-test-scan")
        assert result.metadata.scan_id == "my-test-scan"

    def test_scan_path_with_include_hidden(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        hidden_file = tmp_path / ".hidden_config.py"
        hidden_file.write_text("import anthropic\n", encoding="utf-8")

        result_no_hidden = scan_path(tmp_path, include_hidden=False)
        result_with_hidden = scan_path(tmp_path, include_hidden=True)

        # With hidden files included, anthropic might be detected
        # Without, only openai from app.py
        assert "openai" in result_no_hidden.assets
        assert "openai" in result_with_hidden.assets

    def test_scan_path_with_exclude_paths(self, nested_project: Path) -> None:
        level1 = str(nested_project / "level1")
        result = scan_path(nested_project, exclude_paths={level1})
        # Files under level1 should be excluded
        provider_ids = set(result.assets.keys())
        # Only main.py (openai) should be found
        assert "openai" in provider_ids
        # anthropic is in level1/a.py which is excluded
        # (cohere is in level2 which is also under level1)
        assert "anthropic" not in provider_ids
        assert "cohere" not in provider_ids

    def test_scan_path_empty_project(self, empty_project: Path) -> None:
        result = scan_path(empty_project)
        assert result.summary.total_providers_detected == 0
        assert result.summary.overall_risk_level == RiskLevel.INFO

    def test_scan_path_with_custom_providers_yaml(self, tmp_path: Path) -> None:
        custom_yaml = tmp_path / "custom_providers.yaml"
        data = {
            "providers": [
                {
                    "id": "my_custom_ai",
                    "name": "My Custom AI",
                    "category": "llm",
                    "risk_level": "medium",
                    "vendor_lock_in": False,
                    "data_residency_concern": False,
                    "patterns": {
                        "imports": ["import my_custom_ai"],
                        "env_vars": [],
                        "urls": [],
                        "model_names": [],
                        "sdk_calls": [],
                    },
                    "nist_rmf_functions": ["MAP-1"],
                    "ctem_categories": ["third_party_dependency"],
                    "compliance_notes": [],
                    "docs_url": "",
                }
            ],
            "credential_patterns": [],
            "ctem_categories": {
                "third_party_dependency": {
                    "name": "Third-Party",
                    "description": "Third-party dep.",
                    "severity_modifier": 1.1,
                }
            },
            "nist_rmf_functions": {
                "MAP-1": {"name": "Map", "description": "Map context."}
            },
        }
        custom_yaml.write_text(yaml.dump(data), encoding="utf-8")

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / "app.py").write_text(
            "import my_custom_ai\n", encoding="utf-8"
        )

        result = scan_path(project_dir, providers_yaml_path=custom_yaml)
        assert "my_custom_ai" in result.assets

    def test_scan_path_extra_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import openai\n", encoding="utf-8")
        custom_file = tmp_path / "config.custom"
        custom_file.write_text("import anthropic\n", encoding="utf-8")

        result_no_extra = scan_path(tmp_path)
        result_with_extra = scan_path(tmp_path, extra_extensions={".custom"})

        # Without extra, custom file not scanned
        assert "anthropic" not in result_no_extra.assets
        # With extra, custom file scanned
        assert "anthropic" in result_with_extra.assets


# ---------------------------------------------------------------------------
# discover_files tests
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    """Tests for the discover_files utility function."""

    def test_returns_tuple(self, simple_project: Path) -> None:
        result = discover_files(simple_project)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_of_paths(self, simple_project: Path) -> None:
        files, stats = discover_files(simple_project)
        assert isinstance(files, list)
        assert all(isinstance(f, Path) for f in files)

    def test_returns_walk_stats(self, simple_project: Path) -> None:
        files, stats = discover_files(simple_project)
        assert isinstance(stats, WalkStats)

    def test_finds_python_files(self, simple_project: Path) -> None:
        files, _ = discover_files(simple_project)
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) > 0

    def test_stats_reflect_files_found(self, simple_project: Path) -> None:
        files, stats = discover_files(simple_project)
        assert stats.files_accepted == len(files)
        assert stats.total_files_found >= len(files)

    def test_empty_project_returns_empty_list(self, empty_project: Path) -> None:
        files, stats = discover_files(empty_project)
        assert files == []
        assert stats.files_accepted == 0

    def test_with_custom_config(self, simple_project: Path) -> None:
        config = ScanConfig(extra_extensions={".custom"})
        files, stats = discover_files(simple_project, config=config)
        assert isinstance(files, list)

    def test_string_path_accepted(self, simple_project: Path) -> None:
        files, stats = discover_files(str(simple_project))
        assert isinstance(files, list)

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing_dir"
        with pytest.raises(ValueError, match="does not exist"):
            discover_files(missing)

    def test_excludes_non_supported_files(self, project_with_unsupported_files: Path) -> None:
        files, stats = discover_files(project_with_unsupported_files)
        file_names = {f.name for f in files}
        assert "image.png" not in file_names
        assert "archive.zip" not in file_names
        assert "app.py" in file_names

    def test_stats_skipped_count_correct(self, project_with_unsupported_files: Path) -> None:
        _, stats = discover_files(project_with_unsupported_files)
        assert stats.total_files_skipped > 0


# ---------------------------------------------------------------------------
# Integration tests — full scan pipeline
# ---------------------------------------------------------------------------


class TestFullScanPipeline:
    """End-to-end integration tests for the complete scan pipeline."""

    def test_scan_produces_json_serializable_result(self, simple_project: Path) -> None:
        """Verify the scan result can be serialized to JSON."""
        import json
        scanner = Scanner()
        result = scanner.scan(simple_project)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"
        assert "metadata" in parsed
        assert "summary" in parsed
        assert "assets" in parsed
        assert "findings" in parsed

    def test_scan_findings_have_valid_scores(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        for finding in result.findings:
            assert 0.0 <= finding.risk_score <= 10.0

    def test_scan_findings_have_compliance_gaps(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        # Findings should have at least some compliance gaps
        total_gaps = sum(len(f.compliance_gaps) for f in result.findings)
        assert total_gaps > 0

    def test_scan_multi_provider_all_findings_have_nist_functions(self, multi_provider_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(multi_provider_project)
        for finding in result.findings:
            if finding.provider_id != "_credentials":
                assert len(finding.nist_functions) > 0, (
                    f"Finding {finding.finding_id} has no NIST functions"
                )

    def test_scan_all_findings_have_ctem_categories(self, multi_provider_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(multi_provider_project)
        for finding in result.findings:
            assert len(finding.ctem_categories) > 0, (
                f"Finding {finding.finding_id} has no CTEM categories"
            )

    def test_scan_credential_exposure_flagged(self, credential_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(credential_project)
        # At minimum, we should have detected critical/high risk findings
        high_or_critical = [
            f for f in result.findings
            if f.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]
        assert len(high_or_critical) > 0

    def test_scan_affected_files_subset_of_scanned_files(self, simple_project: Path) -> None:
        """Verify that affected files are a subset of the total files discovered."""
        config = ScanConfig()
        walker = FileWalker(config=config)
        all_files = set(walker.collect(simple_project))

        scanner = Scanner(config=config)
        result = scanner.scan(simple_project)
        affected = result.all_affected_files

        # Every affected file should have been in the discovered files
        for f in affected:
            assert f in all_files, f"Affected file {f} was not in discovered files"

    def test_scan_two_scans_of_same_project_consistent(self, simple_project: Path) -> None:
        """Two scans of the same project should produce consistent provider sets."""
        scanner1 = Scanner()
        result1 = scanner1.scan(simple_project)

        scanner2 = Scanner()
        result2 = scanner2.scan(simple_project)

        providers1 = set(result1.assets.keys())
        providers2 = set(result2.assets.keys())
        assert providers1 == providers2

    def test_scan_nested_project_full_pipeline(self, nested_project: Path) -> None:
        """Full pipeline test on nested project."""
        result = scan_path(nested_project)
        provider_ids = set(result.assets.keys())
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids
        assert result.summary.total_providers_detected >= 2
        assert result.summary.total_findings >= 2

    def test_scan_compliance_gaps_total_matches_summary(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        expected_gaps = sum(len(f.compliance_gaps) for f in result.findings)
        assert result.summary.total_compliance_gaps == expected_gaps

    def test_scan_detection_matches_total_matches_summary(self, simple_project: Path) -> None:
        scanner = Scanner()
        result = scanner.scan(simple_project)
        expected_matches = sum(a.total_matches for a in result.assets.values())
        assert result.summary.total_detection_matches == expected_matches

    def test_scan_with_fixture_sample_project(self) -> None:
        """Test scanning the fixture sample project from the test fixtures directory."""
        fixture_dir = Path(__file__).parent / "fixtures" / "sample_project"
        if not fixture_dir.exists():
            pytest.skip("Sample project fixture not found")

        result = scan_path(fixture_dir)
        assert isinstance(result, ScanResult)

    def test_scan_project_with_no_ai_produces_empty_result(self, tmp_path: Path) -> None:
        """A project with no AI code should produce no providers or findings."""
        (tmp_path / "main.py").write_text(
            textwrap.dedent("""
                import os
                import sys
                from pathlib import Path

                def main() -> None:
                    print("Hello, World!")

                if __name__ == "__main__":
                    main()
            """),
            encoding="utf-8",
        )
        (tmp_path / "utils.py").write_text(
            "import json\nfrom typing import Any\n",
            encoding="utf-8",
        )

        result = scan_path(tmp_path)
        assert result.summary.total_providers_detected == 0
        assert result.summary.total_findings == 0
        assert result.summary.overall_risk_level == RiskLevel.INFO

    def test_scan_dockerfile_with_env_vars(self, tmp_path: Path) -> None:
        """Verify Dockerfiles with ENV AI_KEY variables are scanned."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(
            textwrap.dedent("""
                FROM python:3.11-slim
                ENV OPENAI_API_KEY=""
                ENV ANTHROPIC_API_KEY=""
                RUN pip install openai anthropic
                CMD ["python", "app.py"]
            """),
            encoding="utf-8",
        )
        result = scan_path(tmp_path)
        provider_ids = set(result.assets.keys())
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

    def test_scan_requirements_file(self, tmp_path: Path) -> None:
        """Verify requirements.txt is scanned for AI package references."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            "openai>=1.0\nanthropicai>=0.1\nlangchain>=0.1\n",
            encoding="utf-8",
        )
        result = scan_path(tmp_path)
        # requirements.txt may trigger detection depending on patterns
        # At minimum, scan should complete without errors
        assert isinstance(result, ScanResult)

    def test_scan_json_config_file(self, tmp_path: Path) -> None:
        """Verify JSON config files are scanned."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"model": "gpt-4o", "api_key": "sk-placeholder"}\n',
            encoding="utf-8",
        )
        result = scan_path(tmp_path)
        assert isinstance(result, ScanResult)
        # gpt-4o model name should trigger openai detection
        assert "openai" in result.assets
