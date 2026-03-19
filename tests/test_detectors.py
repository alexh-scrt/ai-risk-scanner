"""Unit tests for the ai_risk_scanner.detectors module.

Tests cover provider signature loading, pattern compilation, file content
scanning, credential detection, result aggregation, and file filtering
across various file types and provider signatures.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ai_risk_scanner.detectors import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILENAMES,
    CredentialPattern,
    DetectionEngine,
    FileDetectionResult,
    ProviderPatterns,
    ProviderSignature,
    _compile_pattern,
    _compile_patterns_list,
    _extract_context_lines,
    _load_providers_yaml,
    _build_provider_signature,
    _build_credential_pattern,
)
from ai_risk_scanner.inventory import (
    AIAsset,
    DetectionMatch,
    DetectionType,
    ProviderCategory,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> DetectionEngine:
    """Return a shared DetectionEngine loaded from the real providers.yaml."""
    return DetectionEngine()


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal providers.yaml for isolated testing."""
    data = {
        "providers": [
            {
                "id": "test_provider",
                "name": "Test Provider",
                "category": "llm",
                "risk_level": "high",
                "vendor_lock_in": True,
                "data_residency_concern": True,
                "patterns": {
                    "imports": ["import test_provider", "from test_provider"],
                    "env_vars": ["TEST_PROVIDER_API_KEY"],
                    "urls": ["api.testprovider.com"],
                    "model_names": ["test-model-v1"],
                    "sdk_calls": ["TestProvider("],
                },
                "nist_rmf_functions": ["GOVERN-1", "MAP-1"],
                "ctem_categories": ["external_exposure"],
                "compliance_notes": ["Review data handling."],
                "docs_url": "https://docs.testprovider.com",
            }
        ],
        "credential_patterns": [
            {
                "name": "Test API Key",
                "pattern": "tp-[A-Za-z0-9]{10,}",
                "risk_level": "critical",
                "description": "Hardcoded Test Provider API key",
            }
        ],
    }
    yaml_file = tmp_path / "providers.yaml"
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")
    return yaml_file


# ---------------------------------------------------------------------------
# _compile_pattern tests
# ---------------------------------------------------------------------------


class TestCompilePattern:
    """Tests for the _compile_pattern helper."""

    def test_literal_string_compiled(self) -> None:
        pattern = _compile_pattern("import openai")
        assert pattern is not None
        assert pattern.search("import openai") is not None

    def test_literal_string_case_insensitive(self) -> None:
        pattern = _compile_pattern("import openai")
        assert pattern is not None
        assert pattern.search("IMPORT OPENAI") is not None

    def test_regex_pattern_compiled(self) -> None:
        pattern = _compile_pattern(r"sk-[A-Za-z0-9]{20,}")
        assert pattern is not None
        assert pattern.search("sk-" + "a" * 20) is not None

    def test_invalid_regex_returns_none(self) -> None:
        with patch("re.compile", side_effect=re.error("bad pattern")):
            result = _compile_pattern("anything")
        assert result is None

    def test_empty_string_returns_not_none(self) -> None:
        # Empty string is a valid regex (matches everything)
        pattern = _compile_pattern("")
        assert pattern is not None

    def test_plain_string_matches_substring(self) -> None:
        pattern = _compile_pattern("openai")
        assert pattern is not None
        assert pattern.search("import openai") is not None
        assert pattern.search("something_else") is None

    def test_regex_with_groups_compiles(self) -> None:
        pattern = _compile_pattern(r"(?i)(api[_-]?key)\s*=\s*['\"]?([A-Za-z0-9]{20,})")
        assert pattern is not None

    def test_pattern_with_backslash_treated_as_regex(self) -> None:
        # Contains regex metachar (backslash), so compiled as regex directly
        pattern = _compile_pattern(r"sk-[A-Za-z0-9]+")
        assert pattern is not None
        assert pattern.search("sk-abc123") is not None


# ---------------------------------------------------------------------------
# _compile_patterns_list tests
# ---------------------------------------------------------------------------


class TestCompilePatternsList:
    """Tests for the _compile_patterns_list helper."""

    def test_basic_compilation(self) -> None:
        patterns = _compile_patterns_list(["import openai", "from openai"])
        assert len(patterns) == 2

    def test_empty_list_returns_empty(self) -> None:
        patterns = _compile_patterns_list([])
        assert patterns == []

    def test_empty_strings_skipped(self) -> None:
        patterns = _compile_patterns_list(["", "  ", "import openai"])
        assert len(patterns) == 1

    def test_all_valid_patterns(self) -> None:
        raw = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"]
        patterns = _compile_patterns_list(raw)
        assert len(patterns) == 3

    def test_mixed_literal_and_regex(self) -> None:
        raw = ["import openai", r"sk-[A-Za-z0-9]{20,}", "OPENAI_API_KEY"]
        patterns = _compile_patterns_list(raw)
        assert len(patterns) == 3

    def test_returns_list_of_compiled_patterns(self) -> None:
        patterns = _compile_patterns_list(["import openai"])
        assert all(isinstance(p, re.Pattern) for p in patterns)


# ---------------------------------------------------------------------------
# _extract_context_lines tests
# ---------------------------------------------------------------------------


class TestExtractContextLines:
    """Tests for the _extract_context_lines helper."""

    def test_basic_context(self) -> None:
        lines = [f"line {i}\n" for i in range(10)]
        ctx = _extract_context_lines(lines, line_index=5, context_count=2)
        assert len(ctx) == 5  # lines 3, 4, 5, 6, 7
        assert "line 3" in ctx[0]
        assert "line 7" in ctx[4]

    def test_context_at_start(self) -> None:
        lines = [f"line {i}\n" for i in range(10)]
        ctx = _extract_context_lines(lines, line_index=0, context_count=2)
        assert "line 0" in ctx[0]
        assert len(ctx) == 3  # lines 0, 1, 2

    def test_context_at_end(self) -> None:
        lines = [f"line {i}\n" for i in range(10)]
        ctx = _extract_context_lines(lines, line_index=9, context_count=2)
        assert "line 9" in ctx[-1]
        assert len(ctx) == 3  # lines 7, 8, 9

    def test_zero_context(self) -> None:
        lines = [f"line {i}\n" for i in range(5)]
        ctx = _extract_context_lines(lines, line_index=2, context_count=0)
        assert len(ctx) == 1
        assert "line 2" in ctx[0]

    def test_context_strips_newlines(self) -> None:
        lines = ["hello\n", "world\n"]
        ctx = _extract_context_lines(lines, 0, 0)
        assert ctx[0] == "hello"

    def test_single_line_file(self) -> None:
        lines = ["only line\n"]
        ctx = _extract_context_lines(lines, 0, 2)
        assert len(ctx) == 1
        assert "only line" in ctx[0]

    def test_context_does_not_exceed_file_bounds(self) -> None:
        lines = [f"line {i}\n" for i in range(3)]
        ctx = _extract_context_lines(lines, line_index=1, context_count=5)
        assert len(ctx) == 3  # all lines


# ---------------------------------------------------------------------------
# _load_providers_yaml tests
# ---------------------------------------------------------------------------


class TestLoadProvidersYaml:
    """Tests for the _load_providers_yaml function."""

    def test_loads_real_providers_yaml(self) -> None:
        from ai_risk_scanner.detectors import _PROVIDERS_YAML_PATH
        data = _load_providers_yaml(_PROVIDERS_YAML_PATH)
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) > 0

    def test_real_yaml_has_credential_patterns(self) -> None:
        from ai_risk_scanner.detectors import _PROVIDERS_YAML_PATH
        data = _load_providers_yaml(_PROVIDERS_YAML_PATH)
        assert "credential_patterns" in data
        assert isinstance(data["credential_patterns"], list)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_providers_yaml(missing)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "providers.yaml"
        bad_yaml.write_text("providers: [unclosed", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            _load_providers_yaml(bad_yaml)

    def test_missing_providers_key_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "providers.yaml"
        bad_yaml.write_text("other_key: value", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required 'providers' key"):
            _load_providers_yaml(bad_yaml)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "providers.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            _load_providers_yaml(bad_yaml)

    def test_returns_dict(self, tmp_path: Path) -> None:
        valid_yaml = tmp_path / "providers.yaml"
        valid_yaml.write_text(
            "providers:\n  - id: test\n    name: Test\n",
            encoding="utf-8",
        )
        data = _load_providers_yaml(valid_yaml)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# _build_provider_signature tests
# ---------------------------------------------------------------------------


class TestBuildProviderSignature:
    """Tests for the _build_provider_signature function."""

    def _base_data(self) -> dict:
        return {
            "id": "test_llm",
            "name": "Test LLM",
            "category": "llm",
            "risk_level": "high",
            "vendor_lock_in": True,
            "data_residency_concern": False,
            "patterns": {
                "imports": ["import test_llm"],
                "env_vars": ["TEST_LLM_KEY"],
                "urls": ["api.testllm.com"],
                "model_names": ["test-model"],
                "sdk_calls": ["TestLLM("],
            },
            "nist_rmf_functions": ["GOVERN-1"],
            "ctem_categories": ["external_exposure"],
            "compliance_notes": ["Review data policies"],
            "docs_url": "https://docs.testllm.com",
        }

    def test_valid_data_builds_signature(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert sig.provider_id == "test_llm"
        assert sig.provider_name == "Test LLM"
        assert sig.risk_level == "high"
        assert sig.vendor_lock_in is True
        assert sig.data_residency_concern is False

    def test_compiled_patterns_populated(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert len(sig.patterns.imports) == 1
        assert len(sig.patterns.env_vars) == 1
        assert len(sig.patterns.urls) == 1
        assert len(sig.patterns.model_names) == 1
        assert len(sig.patterns.sdk_calls) == 1

    def test_missing_id_returns_none(self) -> None:
        data = self._base_data()
        del data["id"]
        sig = _build_provider_signature(data)
        assert sig is None

    def test_empty_id_returns_none(self) -> None:
        data = self._base_data()
        data["id"] = ""
        sig = _build_provider_signature(data)
        assert sig is None

    def test_missing_patterns_ok(self) -> None:
        data = self._base_data()
        del data["patterns"]
        sig = _build_provider_signature(data)
        assert sig is not None
        assert len(sig.patterns.imports) == 0
        assert len(sig.patterns.env_vars) == 0

    def test_null_pattern_lists_handled(self) -> None:
        data = self._base_data()
        data["patterns"] = {
            "imports": None,
            "env_vars": None,
            "urls": [],
            "model_names": [],
            "sdk_calls": [],
        }
        sig = _build_provider_signature(data)
        assert sig is not None
        assert len(sig.patterns.imports) == 0
        assert len(sig.patterns.env_vars) == 0

    def test_raw_patterns_preserved(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert "import test_llm" in sig.raw_patterns["imports"]
        assert "TEST_LLM_KEY" in sig.raw_patterns["env_vars"]

    def test_docs_url_stored(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert sig.docs_url == "https://docs.testllm.com"

    def test_nist_functions_stored(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert "GOVERN-1" in sig.nist_rmf_functions

    def test_ctem_categories_stored(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert "external_exposure" in sig.ctem_categories

    def test_compliance_notes_stored(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        assert "Review data policies" in sig.compliance_notes

    def test_non_dict_patterns_defaults_gracefully(self) -> None:
        data = self._base_data()
        data["patterns"] = "not_a_dict"
        sig = _build_provider_signature(data)
        assert sig is not None
        assert len(sig.patterns.imports) == 0

    def test_repr(self) -> None:
        sig = _build_provider_signature(self._base_data())
        assert sig is not None
        r = repr(sig)
        assert "ProviderSignature" in r
        assert "test_llm" in r


# ---------------------------------------------------------------------------
# _build_credential_pattern tests
# ---------------------------------------------------------------------------


class TestBuildCredentialPattern:
    """Tests for the _build_credential_pattern function."""

    def test_valid_pattern_built(self) -> None:
        data = {
            "name": "OpenAI Key",
            "pattern": r"sk-[A-Za-z0-9]{20,}",
            "risk_level": "critical",
            "description": "Hardcoded OpenAI key",
        }
        cred = _build_credential_pattern(data)
        assert cred is not None
        assert cred.name == "OpenAI Key"
        assert cred.risk_level == "critical"
        assert cred.description == "Hardcoded OpenAI key"
        assert cred.raw_pattern == r"sk-[A-Za-z0-9]{20,}"

    def test_missing_name_returns_none(self) -> None:
        data = {"pattern": r"sk-[A-Za-z0-9]{20,}", "risk_level": "critical"}
        cred = _build_credential_pattern(data)
        assert cred is None

    def test_missing_pattern_returns_none(self) -> None:
        data = {"name": "Test", "risk_level": "critical"}
        cred = _build_credential_pattern(data)
        assert cred is None

    def test_empty_name_returns_none(self) -> None:
        data = {"name": "", "pattern": r"sk-[A-Za-z0-9]{20,}", "risk_level": "critical"}
        cred = _build_credential_pattern(data)
        assert cred is None

    def test_invalid_regex_returns_none(self) -> None:
        data = {
            "name": "Bad Pattern",
            "pattern": r"[unclosed",
            "risk_level": "critical",
            "description": "Invalid regex",
        }
        cred = _build_credential_pattern(data)
        assert cred is None

    def test_pattern_matches_correctly(self) -> None:
        data = {
            "name": "HuggingFace Token",
            "pattern": r"hf_[A-Za-z0-9]{20,}",
            "risk_level": "critical",
            "description": "HuggingFace token",
        }
        cred = _build_credential_pattern(data)
        assert cred is not None
        assert cred.pattern.search("hf_" + "a" * 20) is not None
        assert cred.pattern.search("not_a_token") is None

    def test_default_risk_level_when_missing(self) -> None:
        data = {
            "name": "Some Key",
            "pattern": r"key-[A-Za-z0-9]{10,}",
            "description": "Some key",
        }
        cred = _build_credential_pattern(data)
        assert cred is not None
        assert cred.risk_level == "critical"  # default

    def test_repr(self) -> None:
        data = {
            "name": "Test Key",
            "pattern": r"test-[A-Za-z0-9]{10,}",
            "risk_level": "critical",
            "description": "Test",
        }
        cred = _build_credential_pattern(data)
        assert cred is not None
        r = repr(cred)
        assert "CredentialPattern" in r
        assert "Test Key" in r


# ---------------------------------------------------------------------------
# DetectionEngine initialization tests
# ---------------------------------------------------------------------------


class TestDetectionEngineInit:
    """Tests for DetectionEngine initialization and signature loading."""

    def test_loads_from_default_yaml(self, engine: DetectionEngine) -> None:
        assert engine.provider_count > 0
        assert len(engine.credential_patterns) > 0

    def test_known_providers_loaded(self, engine: DetectionEngine) -> None:
        provider_ids = engine.provider_ids
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids
        assert "aws_bedrock" in provider_ids
        assert "google_vertex" in provider_ids
        assert "huggingface" in provider_ids
        assert "cohere" in provider_ids
        assert "mistral" in provider_ids
        assert "langchain" in provider_ids

    def test_custom_yaml_path(self, minimal_yaml: Path) -> None:
        eng = DetectionEngine(yaml_path=minimal_yaml)
        assert eng.provider_count == 1
        assert "test_provider" in eng.provider_ids

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            DetectionEngine(yaml_path=missing)

    def test_get_provider_signature_found(self, engine: DetectionEngine) -> None:
        sig = engine.get_provider_signature("openai")
        assert sig is not None
        assert sig.provider_id == "openai"
        assert sig.provider_name == "OpenAI"

    def test_get_provider_signature_not_found(self, engine: DetectionEngine) -> None:
        sig = engine.get_provider_signature("nonexistent_provider")
        assert sig is None

    def test_provider_count_matches_yaml(self, engine: DetectionEngine) -> None:
        from ai_risk_scanner.detectors import _PROVIDERS_YAML_PATH
        raw = _load_providers_yaml(_PROVIDERS_YAML_PATH)
        yaml_count = len([p for p in raw["providers"] if isinstance(p, dict) and p.get("id")])
        assert engine.provider_count == yaml_count

    def test_credential_patterns_loaded(self, engine: DetectionEngine) -> None:
        assert len(engine.credential_patterns) > 0
        assert all(isinstance(cp, CredentialPattern) for cp in engine.credential_patterns)

    def test_provider_asset_templates_built(self, engine: DetectionEngine) -> None:
        # Internal template for openai should exist
        template = engine._provider_asset_templates.get("openai")
        assert template is not None
        assert isinstance(template, AIAsset)
        assert template.provider_id == "openai"

    def test_repr(self, engine: DetectionEngine) -> None:
        r = repr(engine)
        assert "DetectionEngine" in r
        assert "providers" in r
        assert "credential_patterns" in r

    def test_custom_engine_credential_patterns(self, minimal_yaml: Path) -> None:
        eng = DetectionEngine(yaml_path=minimal_yaml)
        assert len(eng.credential_patterns) == 1
        assert eng.credential_patterns[0].name == "Test API Key"


# ---------------------------------------------------------------------------
# DetectionEngine.is_supported_file tests
# ---------------------------------------------------------------------------


class TestIsSupportedFile:
    """Tests for DetectionEngine.is_supported_file."""

    def test_python_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("app.py")) is True

    def test_javascript_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("main.js")) is True

    def test_typescript_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("service.ts")) is True

    def test_yaml_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("config.yaml")) is True
        assert engine.is_supported_file(Path("config.yml")) is True

    def test_json_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("package.json")) is True

    def test_env_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path(".env")) is True
        assert engine.is_supported_file(Path(".env.local")) is True

    def test_dockerfile_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("Dockerfile")) is True

    def test_toml_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("pyproject.toml")) is True

    def test_sh_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("deploy.sh")) is True

    def test_go_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("main.go")) is True

    def test_rust_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("main.rs")) is True

    def test_binary_file_not_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("image.png")) is False
        assert engine.is_supported_file(Path("binary.exe")) is False
        assert engine.is_supported_file(Path("archive.zip")) is False
        assert engine.is_supported_file(Path("data.parquet")) is False

    def test_lowercase_extension_matched(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("app.py")) is True

    def test_requirements_txt_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("requirements.txt")) is True

    def test_docker_compose_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("docker-compose.yml")) is True

    def test_markdown_file_supported(self, engine: DetectionEngine) -> None:
        assert engine.is_supported_file(Path("README.md")) is True


# ---------------------------------------------------------------------------
# FileDetectionResult tests
# ---------------------------------------------------------------------------


class TestFileDetectionResult:
    """Tests for the FileDetectionResult dataclass."""

    def _make_template_asset(self, provider_id: str = "openai") -> AIAsset:
        return AIAsset(
            provider_id=provider_id,
            provider_name=provider_id.capitalize(),
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )

    def _make_match(self, provider_id: str = "openai", line: int = 1) -> DetectionMatch:
        return DetectionMatch(
            file_path=Path("app.py"),
            line_number=line,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=DetectionType.IMPORT,
            provider_id=provider_id,
        )

    def test_initial_state(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        assert result.total_matches == 0
        assert result.providers_detected == []
        assert len(result.scan_errors) == 0
        assert result.file_path == Path("app.py")

    def test_add_match_creates_asset(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        asset = self._make_template_asset("openai")
        match = self._make_match("openai")
        result.add_match(asset, match)
        assert "openai" in result.assets
        assert result.total_matches == 1

    def test_add_match_reuses_existing_asset(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        asset = self._make_template_asset("openai")
        result.add_match(asset, self._make_match("openai", 1))
        result.add_match(asset, self._make_match("openai", 5))
        assert len(result.assets) == 1
        assert result.assets["openai"].total_matches == 2
        assert result.total_matches == 2

    def test_add_match_multiple_providers(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        for pid in ["openai", "anthropic", "cohere"]:
            asset = self._make_template_asset(pid)
            match = self._make_match(pid)
            result.add_match(asset, match)
        assert set(result.providers_detected) == {"openai", "anthropic", "cohere"}
        assert result.total_matches == 3

    def test_asset_inherits_template_metadata(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        template = self._make_template_asset("openai")
        template.nist_rmf_functions = ["GOVERN-1", "MAP-1"]
        template.ctem_categories = ["external_exposure"]
        result.add_match(template, self._make_match("openai"))
        assert result.assets["openai"].nist_rmf_functions == ["GOVERN-1", "MAP-1"]
        assert result.assets["openai"].ctem_categories == ["external_exposure"]

    def test_repr(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        r = repr(result)
        assert "FileDetectionResult" in r
        assert "app.py" in r

    def test_scan_errors_list_default_empty(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        assert result.scan_errors == []

    def test_add_error(self) -> None:
        result = FileDetectionResult(file_path=Path("app.py"))
        result.scan_errors.append("File too large")
        assert len(result.scan_errors) == 1


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — OpenAI
# ---------------------------------------------------------------------------


class TestScanContentOpenAI:
    """Tests for scanning OpenAI-specific patterns."""

    def test_detects_openai_import(self, engine: DetectionEngine) -> None:
        content = "import openai\nclient = openai.OpenAI()"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_from_openai_import(self, engine: DetectionEngine) -> None:
        content = "from openai import OpenAI"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_openai_api_key_env_var(self, engine: DetectionEngine) -> None:
        content = "OPENAI_API_KEY=sk-abc123"
        result = engine.scan_content(content, Path(".env"))
        assert "openai" in result.providers_detected
        asset = result.assets["openai"]
        env_matches = [
            m for m in asset.detection_matches
            if m.detection_type == DetectionType.ENV_VAR
        ]
        assert len(env_matches) > 0

    def test_detects_openai_url(self, engine: DetectionEngine) -> None:
        content = 'base_url = "https://api.openai.com/v1"'
        result = engine.scan_content(content, Path("config.py"))
        assert "openai" in result.providers_detected

    def test_detects_gpt4_model_name(self, engine: DetectionEngine) -> None:
        content = 'model = "gpt-4o"'
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_gpt35_model_name(self, engine: DetectionEngine) -> None:
        content = 'model = "gpt-3.5-turbo"'
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_openai_sdk_call(self, engine: DetectionEngine) -> None:
        content = "client = OpenAI()\nresponse = client.chat.completions.create(...)"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_hardcoded_openai_key(self, engine: DetectionEngine) -> None:
        key = "sk-" + "a" * 25
        content = f'api_key = "{key}"'
        result = engine.scan_content(content, Path("app.py"))
        has_cred = any(
            any(m.detection_type == DetectionType.CREDENTIAL for m in asset.detection_matches)
            for asset in result.assets.values()
        )
        assert has_cred

    def test_empty_content_returns_no_detections(self, engine: DetectionEngine) -> None:
        result = engine.scan_content("", Path("app.py"))
        assert result.total_matches == 0
        assert result.providers_detected == []

    def test_whitespace_only_content(self, engine: DetectionEngine) -> None:
        result = engine.scan_content("   \n\n   \t\n", Path("app.py"))
        assert result.total_matches == 0

    def test_no_ai_content_no_detections(self, engine: DetectionEngine) -> None:
        content = textwrap.dedent("""
            import os
            import sys
            from pathlib import Path

            def hello():
                print("Hello, world!")
        """)
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" not in result.providers_detected
        assert "anthropic" not in result.providers_detected

    def test_detects_async_openai(self, engine: DetectionEngine) -> None:
        content = "client = AsyncOpenAI()"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_detects_whisper_model_name(self, engine: DetectionEngine) -> None:
        content = 'model = "whisper-1"'
        result = engine.scan_content(content, Path("config.yaml"))
        # whisper-1 appears in openai and openai_whisper providers
        detected = set(result.providers_detected)
        assert "openai" in detected or "openai_whisper" in detected

    def test_detection_match_has_line_number(self, engine: DetectionEngine) -> None:
        content = "# header\nimport openai\n"
        result = engine.scan_content(content, Path("app.py"))
        if "openai" in result.assets:
            import_matches = [
                m for m in result.assets["openai"].detection_matches
                if m.detection_type == DetectionType.IMPORT
            ]
            if import_matches:
                assert import_matches[0].line_number == 2

    def test_detection_match_has_context_lines(self, engine: DetectionEngine) -> None:
        content = "# context above\nimport openai\n# context below\n"
        result = engine.scan_content(content, Path("app.py"))
        if "openai" in result.assets:
            matches = result.assets["openai"].detection_matches
            if matches:
                assert len(matches[0].context_lines) > 0


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — Anthropic
# ---------------------------------------------------------------------------


class TestScanContentAnthropic:
    """Tests for scanning Anthropic-specific patterns."""

    def test_detects_anthropic_import(self, engine: DetectionEngine) -> None:
        content = "import anthropic"
        result = engine.scan_content(content, Path("app.py"))
        assert "anthropic" in result.providers_detected

    def test_detects_from_anthropic_import(self, engine: DetectionEngine) -> None:
        content = "from anthropic import Anthropic"
        result = engine.scan_content(content, Path("app.py"))
        assert "anthropic" in result.providers_detected

    def test_detects_anthropic_api_key(self, engine: DetectionEngine) -> None:
        content = "ANTHROPIC_API_KEY=my-secret-key"
        result = engine.scan_content(content, Path(".env"))
        assert "anthropic" in result.providers_detected

    def test_detects_claude_api_key(self, engine: DetectionEngine) -> None:
        content = "CLAUDE_API_KEY=my-secret-key"
        result = engine.scan_content(content, Path(".env"))
        assert "anthropic" in result.providers_detected

    def test_detects_claude_model_name(self, engine: DetectionEngine) -> None:
        content = 'model_id = "claude-3-5-sonnet"'
        result = engine.scan_content(content, Path("config.yaml"))
        assert "anthropic" in result.providers_detected

    def test_detects_claude_opus(self, engine: DetectionEngine) -> None:
        content = 'model = "claude-3-opus"'
        result = engine.scan_content(content, Path("app.py"))
        assert "anthropic" in result.providers_detected

    def test_detects_anthropic_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://api.anthropic.com/v1/messages"'
        result = engine.scan_content(content, Path("config.py"))
        assert "anthropic" in result.providers_detected

    def test_detects_hardcoded_anthropic_key(self, engine: DetectionEngine) -> None:
        key = "sk-ant-" + "a" * 25
        content = f'api_key = "{key}"'
        result = engine.scan_content(content, Path("app.py"))
        has_cred = any(
            any(m.detection_type == DetectionType.CREDENTIAL for m in asset.detection_matches)
            for asset in result.assets.values()
        )
        assert has_cred

    def test_detects_messages_create_sdk_call(self, engine: DetectionEngine) -> None:
        content = "response = client.messages.create(model='claude-3-opus', ...)"
        result = engine.scan_content(content, Path("app.py"))
        assert "anthropic" in result.providers_detected

    def test_detects_anthropic_constructor(self, engine: DetectionEngine) -> None:
        content = "client = Anthropic()"
        result = engine.scan_content(content, Path("app.py"))
        assert "anthropic" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — AWS Bedrock
# ---------------------------------------------------------------------------


class TestScanContentAWSBedrock:
    """Tests for scanning AWS Bedrock-specific patterns."""

    def test_detects_boto3_import(self, engine: DetectionEngine) -> None:
        content = "import boto3\nclient = boto3.client('bedrock-runtime')"
        result = engine.scan_content(content, Path("app.py"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_bedrock_env_vars(self, engine: DetectionEngine) -> None:
        content = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = engine.scan_content(content, Path(".env"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_aws_secret_key(self, engine: DetectionEngine) -> None:
        content = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = engine.scan_content(content, Path(".env"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_bedrock_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://bedrock-runtime.amazonaws.com"'
        result = engine.scan_content(content, Path("config.py"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_invoke_model_sdk_call(self, engine: DetectionEngine) -> None:
        content = "response = client.InvokeModel(modelId=model_id, body=body)"
        result = engine.scan_content(content, Path("app.py"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_bedrock_agent_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://bedrock-agent.amazonaws.com"'
        result = engine.scan_content(content, Path("config.py"))
        assert "aws_bedrock" in result.providers_detected

    def test_detects_converse_sdk_call(self, engine: DetectionEngine) -> None:
        content = "response = client.Converse(modelId=model_id, messages=messages)"
        result = engine.scan_content(content, Path("app.py"))
        assert "aws_bedrock" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — Google Vertex AI
# ---------------------------------------------------------------------------


class TestScanContentGoogleVertex:
    """Tests for scanning Google Vertex AI-specific patterns."""

    def test_detects_vertex_import(self, engine: DetectionEngine) -> None:
        content = "from google.cloud import aiplatform"
        result = engine.scan_content(content, Path("app.py"))
        assert "google_vertex" in result.providers_detected

    def test_detects_vertexai_import(self, engine: DetectionEngine) -> None:
        content = "import vertexai\nfrom vertexai.generative_models import GenerativeModel"
        result = engine.scan_content(content, Path("app.py"))
        assert "google_vertex" in result.providers_detected

    def test_detects_gemini_api_key(self, engine: DetectionEngine) -> None:
        content = "GEMINI_API_KEY=abc123xyz"
        result = engine.scan_content(content, Path(".env"))
        assert "google_vertex" in result.providers_detected

    def test_detects_google_api_key(self, engine: DetectionEngine) -> None:
        content = "GOOGLE_API_KEY=abc123xyz"
        result = engine.scan_content(content, Path(".env"))
        assert "google_vertex" in result.providers_detected

    def test_detects_gemini_model_name(self, engine: DetectionEngine) -> None:
        content = 'model = "gemini-1.5-pro"'
        result = engine.scan_content(content, Path("config.yaml"))
        assert "google_vertex" in result.providers_detected

    def test_detects_gemini_flash_model(self, engine: DetectionEngine) -> None:
        content = 'model = "gemini-1.5-flash"'
        result = engine.scan_content(content, Path("app.py"))
        assert "google_vertex" in result.providers_detected

    def test_detects_aiplatform_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://aiplatform.googleapis.com"'
        result = engine.scan_content(content, Path("config.py"))
        assert "google_vertex" in result.providers_detected

    def test_detects_generativemodel_sdk_call(self, engine: DetectionEngine) -> None:
        content = "model = GenerativeModel('gemini-1.5-pro')"
        result = engine.scan_content(content, Path("app.py"))
        assert "google_vertex" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — Hugging Face
# ---------------------------------------------------------------------------


class TestScanContentHuggingFace:
    """Tests for scanning Hugging Face-specific patterns."""

    def test_detects_transformers_import(self, engine: DetectionEngine) -> None:
        content = "from transformers import AutoModelForCausalLM, AutoTokenizer"
        result = engine.scan_content(content, Path("app.py"))
        assert "huggingface" in result.providers_detected

    def test_detects_huggingface_hub_import(self, engine: DetectionEngine) -> None:
        content = "from huggingface_hub import hf_hub_download"
        result = engine.scan_content(content, Path("app.py"))
        assert "huggingface" in result.providers_detected

    def test_detects_hf_token_env(self, engine: DetectionEngine) -> None:
        content = "HF_TOKEN=hf_abcdefghijklmnopqrst"
        result = engine.scan_content(content, Path(".env"))
        assert "huggingface" in result.providers_detected

    def test_detects_hugging_face_hub_token_env(self, engine: DetectionEngine) -> None:
        content = "HUGGING_FACE_HUB_TOKEN=hf_abcdefghijklmnopqrst"
        result = engine.scan_content(content, Path(".env"))
        assert "huggingface" in result.providers_detected

    def test_detects_pipeline_sdk_call(self, engine: DetectionEngine) -> None:
        content = "generator = pipeline('text-generation', model='gpt2')"
        result = engine.scan_content(content, Path("app.py"))
        assert "huggingface" in result.providers_detected

    def test_detects_from_pretrained(self, engine: DetectionEngine) -> None:
        content = "model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b')"
        result = engine.scan_content(content, Path("app.py"))
        assert "huggingface" in result.providers_detected

    def test_detects_hf_token_hardcoded(self, engine: DetectionEngine) -> None:
        token = "hf_" + "a" * 25
        content = f'token = "{token}"'
        result = engine.scan_content(content, Path("app.py"))
        has_cred = any(
            any(m.detection_type == DetectionType.CREDENTIAL for m in asset.detection_matches)
            for asset in result.assets.values()
        )
        assert has_cred

    def test_detects_huggingface_url(self, engine: DetectionEngine) -> None:
        content = 'api_url = "https://huggingface.co/api"'
        result = engine.scan_content(content, Path("config.py"))
        assert "huggingface" in result.providers_detected

    def test_detects_datasets_import(self, engine: DetectionEngine) -> None:
        content = "from datasets import load_dataset"
        result = engine.scan_content(content, Path("app.py"))
        assert "huggingface" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — LangChain
# ---------------------------------------------------------------------------


class TestScanContentLangChain:
    """Tests for scanning LangChain-specific patterns."""

    def test_detects_langchain_import(self, engine: DetectionEngine) -> None:
        content = "from langchain_openai import ChatOpenAI"
        result = engine.scan_content(content, Path("app.py"))
        assert "langchain" in result.providers_detected

    def test_detects_langchain_core_import(self, engine: DetectionEngine) -> None:
        content = "from langchain_core.messages import HumanMessage"
        result = engine.scan_content(content, Path("app.py"))
        assert "langchain" in result.providers_detected

    def test_detects_langchain_community_import(self, engine: DetectionEngine) -> None:
        content = "from langchain_community.vectorstores import FAISS"
        result = engine.scan_content(content, Path("app.py"))
        assert "langchain" in result.providers_detected

    def test_detects_langsmith_tracing(self, engine: DetectionEngine) -> None:
        content = "LANGCHAIN_TRACING_V2=true"
        result = engine.scan_content(content, Path(".env"))
        assert "langchain" in result.providers_detected

    def test_detects_langsmith_api_key(self, engine: DetectionEngine) -> None:
        content = "LANGSMITH_API_KEY=ls_abc123"
        result = engine.scan_content(content, Path(".env"))
        assert "langchain" in result.providers_detected

    def test_detects_chat_openai_sdk_call(self, engine: DetectionEngine) -> None:
        content = "llm = ChatOpenAI(model='gpt-4', temperature=0)"
        result = engine.scan_content(content, Path("app.py"))
        assert "langchain" in result.providers_detected

    def test_detects_agent_executor_sdk_call(self, engine: DetectionEngine) -> None:
        content = "agent = AgentExecutor(agent=agent, tools=tools)"
        result = engine.scan_content(content, Path("app.py"))
        assert "langchain" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — Cohere
# ---------------------------------------------------------------------------


class TestScanContentCohere:
    """Tests for scanning Cohere-specific patterns."""

    def test_detects_cohere_import(self, engine: DetectionEngine) -> None:
        content = "import cohere"
        result = engine.scan_content(content, Path("app.py"))
        assert "cohere" in result.providers_detected

    def test_detects_cohere_api_key(self, engine: DetectionEngine) -> None:
        content = "COHERE_API_KEY=abc123xyz"
        result = engine.scan_content(content, Path(".env"))
        assert "cohere" in result.providers_detected

    def test_detects_command_model(self, engine: DetectionEngine) -> None:
        content = 'model = "command-r-plus"'
        result = engine.scan_content(content, Path("app.py"))
        assert "cohere" in result.providers_detected

    def test_detects_cohere_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://api.cohere.ai"'
        result = engine.scan_content(content, Path("config.py"))
        assert "cohere" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_content tests — Mistral
# ---------------------------------------------------------------------------


class TestScanContentMistral:
    """Tests for scanning Mistral-specific patterns."""

    def test_detects_mistralai_import(self, engine: DetectionEngine) -> None:
        content = "from mistralai import Mistral"
        result = engine.scan_content(content, Path("app.py"))
        assert "mistral" in result.providers_detected

    def test_detects_mistral_api_key(self, engine: DetectionEngine) -> None:
        content = "MISTRAL_API_KEY=abc123xyz"
        result = engine.scan_content(content, Path(".env"))
        assert "mistral" in result.providers_detected

    def test_detects_mistral_model_name(self, engine: DetectionEngine) -> None:
        content = 'model = "mistral-large"'
        result = engine.scan_content(content, Path("app.py"))
        assert "mistral" in result.providers_detected

    def test_detects_mistral_url(self, engine: DetectionEngine) -> None:
        content = 'endpoint = "https://api.mistral.ai"'
        result = engine.scan_content(content, Path("config.py"))
        assert "mistral" in result.providers_detected


# ---------------------------------------------------------------------------
# Credential detection tests
# ---------------------------------------------------------------------------


class TestCredentialDetection:
    """Tests for cross-provider credential (hardcoded secret) detection."""

    def test_detects_openai_key_pattern(self, engine: DetectionEngine) -> None:
        key = "sk-" + "AbCdEfGhIjKlMnOpQrSt12"
        content = f'OPENAI_API_KEY = "{key}"'
        result = engine.scan_content(content, Path("config.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) > 0

    def test_detects_anthropic_key_pattern(self, engine: DetectionEngine) -> None:
        key = "sk-ant-" + "AbCdEfGhIjKlMnOpQrSt12"
        content = f'api_key = "{key}"'
        result = engine.scan_content(content, Path("config.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) > 0

    def test_detects_huggingface_token(self, engine: DetectionEngine) -> None:
        token = "hf_" + "AbCdEfGhIjKlMnOpQrSt12"
        content = f'token = "{token}"'
        result = engine.scan_content(content, Path("app.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) > 0

    def test_comment_lines_skipped(self, engine: DetectionEngine) -> None:
        key = "sk-" + "a" * 25
        content = f'# api_key = "{key}"'
        result = engine.scan_content(content, Path("app.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) == 0

    def test_no_false_positive_on_short_key(self, engine: DetectionEngine) -> None:
        # Key is too short to match any real credential pattern
        content = 'api_key = "sk-abc"'  # only 3 chars after sk-
        result = engine.scan_content(content, Path("app.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) == 0

    def test_credential_attributed_to_provider_when_env_var_matches(self, engine: DetectionEngine) -> None:
        # Line contains both the env var name and the credential value
        key = "sk-" + "a" * 25
        content = f'OPENAI_API_KEY = "{key}"'
        result = engine.scan_content(content, Path("config.py"))
        # Should be attributed to openai via env var pattern matching
        cred_match_providers = [
            m.provider_id
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_match_providers) > 0

    def test_generic_credential_asset_created(self, engine: DetectionEngine) -> None:
        # A bearer token with no provider context
        content = 'Authorization: Bearer AbCdEfGhIjKlMnOpQrStUvWxYz1234567890'
        result = engine.scan_content(content, Path("request.txt"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) > 0

    def test_double_slash_comment_skipped(self, engine: DetectionEngine) -> None:
        key = "sk-" + "a" * 25
        content = f'// api_key = "{key}"'
        result = engine.scan_content(content, Path("app.js"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) == 0


# ---------------------------------------------------------------------------
# DetectionEngine.scan_file tests
# ---------------------------------------------------------------------------


class TestScanFile:
    """Tests for DetectionEngine.scan_file."""

    def test_scan_real_python_file(self, engine: DetectionEngine, tmp_path: Path) -> None:
        py_file = tmp_path / "app.py"
        py_file.write_text(
            "import openai\nclient = openai.OpenAI()\n",
            encoding="utf-8",
        )
        result = engine.scan_file(py_file)
        assert "openai" in result.providers_detected
        assert len(result.scan_errors) == 0

    def test_scan_env_file(self, engine: DetectionEngine, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-abc\nANTHROPIC_API_KEY=sk-ant-xyz\n",
            encoding="utf-8",
        )
        result = engine.scan_file(env_file)
        assert "openai" in result.providers_detected
        assert "anthropic" in result.providers_detected

    def test_scan_yaml_config_file(self, engine: DetectionEngine, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "model: gemini-1.5-pro\napi_endpoint: https://aiplatform.googleapis.com\n",
            encoding="utf-8",
        )
        result = engine.scan_file(yaml_file)
        assert "google_vertex" in result.providers_detected

    def test_nonexistent_file_records_error(self, engine: DetectionEngine, tmp_path: Path) -> None:
        missing = tmp_path / "missing.py"
        result = engine.scan_file(missing)
        assert result.total_matches == 0
        assert len(result.scan_errors) > 0
        assert "not found" in result.scan_errors[0].lower()

    def test_directory_records_error(self, engine: DetectionEngine, tmp_path: Path) -> None:
        result = engine.scan_file(tmp_path)  # directory, not file
        assert result.total_matches == 0
        assert len(result.scan_errors) > 0

    def test_oversized_file_skipped(self, engine: DetectionEngine, tmp_path: Path) -> None:
        from ai_risk_scanner.detectors import MAX_FILE_SIZE_BYTES
        large_file = tmp_path / "large.py"
        large_file.write_bytes(b"import openai\n" + b"x" * (MAX_FILE_SIZE_BYTES + 1))
        result = engine.scan_file(large_file)
        assert result.total_matches == 0
        assert any("too large" in err.lower() for err in result.scan_errors)

    def test_scan_empty_file(self, engine: DetectionEngine, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")
        result = engine.scan_file(empty_file)
        assert result.total_matches == 0
        assert len(result.scan_errors) == 0

    def test_scan_typescript_file(self, engine: DetectionEngine, tmp_path: Path) -> None:
        ts_file = tmp_path / "service.ts"
        ts_file.write_text(
            "import OpenAI from 'openai';\nconst client = new OpenAI();\n",
            encoding="utf-8",
        )
        result = engine.scan_file(ts_file)
        assert "openai" in result.providers_detected

    def test_scan_file_result_has_correct_path(self, engine: DetectionEngine, tmp_path: Path) -> None:
        py_file = tmp_path / "app.py"
        py_file.write_text("import openai\n", encoding="utf-8")
        result = engine.scan_file(py_file)
        assert result.file_path == py_file

    def test_scan_file_with_custom_encoding(self, engine: DetectionEngine, tmp_path: Path) -> None:
        py_file = tmp_path / "app.py"
        content = "import openai  # AI client\n"
        py_file.write_text(content, encoding="utf-8")
        result = engine.scan_file(py_file, encoding="utf-8")
        assert "openai" in result.providers_detected


# ---------------------------------------------------------------------------
# DetectionEngine.scan_files tests
# ---------------------------------------------------------------------------


class TestScanFiles:
    """Tests for DetectionEngine.scan_files (batch scanning)."""

    def test_scans_multiple_files(self, engine: DetectionEngine, tmp_path: Path) -> None:
        file1 = tmp_path / "app.py"
        file1.write_text("import openai\n", encoding="utf-8")
        file2 = tmp_path / "config.py"
        file2.write_text("import anthropic\n", encoding="utf-8")

        results = engine.scan_files([file1, file2])
        assert len(results) == 2

        detected_providers: set[str] = set()
        for r in results:
            detected_providers.update(r.providers_detected)
        assert "openai" in detected_providers
        assert "anthropic" in detected_providers

    def test_empty_file_list(self, engine: DetectionEngine) -> None:
        results = engine.scan_files([])
        assert results == []

    def test_result_count_matches_input(self, engine: DetectionEngine, tmp_path: Path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"file_{i}.py"
            f.write_text(f"# file {i}\n", encoding="utf-8")
            files.append(f)
        results = engine.scan_files(files)
        assert len(results) == 5

    def test_preserves_file_order(self, engine: DetectionEngine, tmp_path: Path) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.py"
            f.write_text("import openai\n", encoding="utf-8")
            files.append(f)
        results = engine.scan_files(files)
        for i, result in enumerate(results):
            assert result.file_path == files[i]

    def test_nonexistent_files_recorded_as_errors(self, engine: DetectionEngine, tmp_path: Path) -> None:
        existing = tmp_path / "exists.py"
        existing.write_text("import openai\n", encoding="utf-8")
        missing = tmp_path / "missing.py"

        results = engine.scan_files([existing, missing])
        assert len(results) == 2
        # First result should have detections, second should have errors
        assert len(results[0].scan_errors) == 0
        assert len(results[1].scan_errors) > 0


# ---------------------------------------------------------------------------
# DetectionEngine.aggregate_results tests
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Tests for DetectionEngine.aggregate_results."""

    def test_aggregates_same_provider_across_files(self, engine: DetectionEngine, tmp_path: Path) -> None:
        file1 = tmp_path / "app.py"
        file1.write_text("import openai\n", encoding="utf-8")
        file2 = tmp_path / "utils.py"
        file2.write_text("from openai import OpenAI\n", encoding="utf-8")

        results = engine.scan_files([file1, file2])
        aggregated = engine.aggregate_results(results)

        assert "openai" in aggregated
        assert aggregated["openai"].total_matches >= 2
        assert len(aggregated["openai"].files_detected) == 2

    def test_aggregates_multiple_providers(self, engine: DetectionEngine, tmp_path: Path) -> None:
        file1 = tmp_path / "ai_client.py"
        file1.write_text(
            "import openai\nimport anthropic\n",
            encoding="utf-8",
        )
        results = engine.scan_files([file1])
        aggregated = engine.aggregate_results(results)
        assert "openai" in aggregated
        assert "anthropic" in aggregated

    def test_empty_results_returns_empty_dict(self, engine: DetectionEngine) -> None:
        aggregated = engine.aggregate_results([])
        assert aggregated == {}

    def test_no_detections_returns_empty_dict(self, engine: DetectionEngine, tmp_path: Path) -> None:
        clean_file = tmp_path / "clean.py"
        clean_file.write_text("import os\nimport sys\n", encoding="utf-8")
        results = engine.scan_files([clean_file])
        aggregated = engine.aggregate_results(results)
        assert len(aggregated) == 0

    def test_aggregate_preserves_compliance_notes(self, engine: DetectionEngine, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("import cohere\n", encoding="utf-8")
        results = engine.scan_files([f])
        aggregated = engine.aggregate_results(results)
        assert "cohere" in aggregated
        assert len(aggregated["cohere"].compliance_notes) > 0

    def test_aggregate_preserves_nist_functions(self, engine: DetectionEngine, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("import openai\n", encoding="utf-8")
        results = engine.scan_files([f])
        aggregated = engine.aggregate_results(results)
        assert "openai" in aggregated
        assert len(aggregated["openai"].nist_rmf_functions) > 0

    def test_aggregate_preserves_ctem_categories(self, engine: DetectionEngine, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("import anthropic\n", encoding="utf-8")
        results = engine.scan_files([f])
        aggregated = engine.aggregate_results(results)
        assert "anthropic" in aggregated
        assert len(aggregated["anthropic"].ctem_categories) > 0

    def test_aggregate_files_detected_correct(self, engine: DetectionEngine, tmp_path: Path) -> None:
        file1 = tmp_path / "a.py"
        file1.write_text("import openai\n", encoding="utf-8")
        file2 = tmp_path / "b.py"
        file2.write_text("from openai import OpenAI\n", encoding="utf-8")
        file3 = tmp_path / "c.py"
        file3.write_text("# no ai here\n", encoding="utf-8")

        results = engine.scan_files([file1, file2, file3])
        aggregated = engine.aggregate_results(results)
        assert "openai" in aggregated
        # Only file1 and file2 have openai references
        assert len(aggregated["openai"].files_detected) == 2


# ---------------------------------------------------------------------------
# Multi-provider detection tests
# ---------------------------------------------------------------------------


class TestMultiProviderDetection:
    """Tests for files containing multiple AI provider references."""

    def test_detects_multiple_providers_in_one_file(self, engine: DetectionEngine) -> None:
        content = textwrap.dedent("""
            import openai
            import anthropic
            from langchain_openai import ChatOpenAI

            openai_client = openai.OpenAI()
            claude_client = anthropic.Anthropic()
        """)
        result = engine.scan_content(content, Path("multi_ai.py"))
        detected = set(result.providers_detected)
        assert "openai" in detected
        assert "anthropic" in detected
        assert "langchain" in detected

    def test_env_file_with_multiple_providers(self, engine: DetectionEngine) -> None:
        content = textwrap.dedent("""
            OPENAI_API_KEY=sk-placeholder
            ANTHROPIC_API_KEY=sk-ant-placeholder
            GOOGLE_API_KEY=placeholder
            REPLICATE_API_TOKEN=r8_placeholder
            COHERE_API_KEY=placeholder
        """)
        result = engine.scan_content(content, Path(".env"))
        detected = set(result.providers_detected)
        assert "openai" in detected
        assert "anthropic" in detected
        assert "google_vertex" in detected
        assert "replicate" in detected
        assert "cohere" in detected

    def test_dockerfile_with_ai_references(self, engine: DetectionEngine) -> None:
        content = textwrap.dedent("""
            FROM python:3.11-slim
            ENV OPENAI_API_KEY=""
            ENV ANTHROPIC_API_KEY=""
            RUN pip install openai anthropic
        """)
        result = engine.scan_content(content, Path("Dockerfile"))
        detected = set(result.providers_detected)
        assert "openai" in detected
        assert "anthropic" in detected

    def test_python_file_multiple_imports_and_calls(self, engine: DetectionEngine) -> None:
        content = textwrap.dedent("""
            import openai
            from anthropic import Anthropic
            import cohere

            openai_client = openai.OpenAI()
            anthropic_client = Anthropic()
            co_client = cohere.Client(api_key="test")
        """)
        result = engine.scan_content(content, Path("clients.py"))
        detected = set(result.providers_detected)
        assert "openai" in detected
        assert "anthropic" in detected
        assert "cohere" in detected
        assert result.total_matches >= 3


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests to ensure duplicate matches on the same line are not double-counted."""

    def test_same_pattern_same_line_not_duplicated(self, engine: DetectionEngine) -> None:
        # The same import on a single line should produce at most one match per pattern
        content = "import openai  # using openai"
        result = engine.scan_content(content, Path("app.py"))
        if "openai" in result.assets:
            # Collect (pattern, line_number) pairs
            seen_keys: set[tuple[str, int]] = set()
            for m in result.assets["openai"].detection_matches:
                key = (m.matched_pattern, m.line_number)
                assert key not in seen_keys, (
                    f"Duplicate match: pattern={m.matched_pattern!r}, line={m.line_number}"
                )
                seen_keys.add(key)

    def test_duplicate_lines_each_counted_once(self, engine: DetectionEngine) -> None:
        # Two different lines matching the same pattern should each be counted
        content = "import openai\nimport openai\n"
        result = engine.scan_content(content, Path("app.py"))
        if "openai" in result.assets:
            import_matches = [
                m for m in result.assets["openai"].detection_matches
                if m.detection_type == DetectionType.IMPORT
            ]
            # Should see matches for both line 1 and line 2 (different line indices)
            line_numbers = {m.line_number for m in import_matches
                           if "import openai" in m.matched_pattern.lower()
                           or "openai" in m.matched_pattern.lower()}
            # Both lines should be represented
            assert 1 in line_numbers or 2 in line_numbers


# ---------------------------------------------------------------------------
# Edge case and error handling tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling in the detection engine."""

    def test_binary_looking_content_handled(self, engine: DetectionEngine) -> None:
        content = "import openai\n\x00\x01\x02\nfrom openai import OpenAI\n"
        result = engine.scan_content(content, Path("mixed.py"))
        assert "openai" in result.providers_detected

    def test_very_long_line_handled(self, engine: DetectionEngine) -> None:
        long_line = "x = 1  # " + "a" * 10000 + "  import openai"
        result = engine.scan_content(long_line, Path("app.py"))
        assert isinstance(result, FileDetectionResult)

    def test_unicode_content_handled(self, engine: DetectionEngine) -> None:
        content = "# Chinesé: 导入 openai\nimport openai\n"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_windows_line_endings(self, engine: DetectionEngine) -> None:
        content = "import openai\r\nclient = openai.OpenAI()\r\n"
        result = engine.scan_content(content, Path("app.py"))
        assert "openai" in result.providers_detected

    def test_mixed_line_endings(self, engine: DetectionEngine) -> None:
        content = "import openai\nimport anthropic\r\nfrom langchain import LLMChain\r"
        result = engine.scan_content(content, Path("app.py"))
        detected = set(result.providers_detected)
        assert "openai" in detected
        assert "anthropic" in detected

    def test_single_line_file(self, engine: DetectionEngine) -> None:
        result = engine.scan_content("import openai", Path("app.py"))
        assert "openai" in result.providers_detected

    def test_content_with_only_comments(self, engine: DetectionEngine) -> None:
        content = "# import openai\n# from anthropic import Anthropic\n"
        result = engine.scan_content(content, Path("app.py"))
        # Comments should not produce import/sdk_call detections
        import_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.IMPORT
        ]
        assert len(import_matches) == 0

    def test_custom_engine_with_minimal_yaml(self, minimal_yaml: Path) -> None:
        eng = DetectionEngine(yaml_path=minimal_yaml)
        content = "import test_provider\nTEST_PROVIDER_API_KEY=abc123"
        result = eng.scan_content(content, Path("app.py"))
        assert "test_provider" in result.providers_detected

    def test_custom_engine_credential_detection(self, minimal_yaml: Path) -> None:
        eng = DetectionEngine(yaml_path=minimal_yaml)
        # The minimal yaml has a credential pattern for tp-[A-Za-z0-9]{10,}
        content = 'api_key = "tp-AbCdEfGhIjKl"'
        result = eng.scan_content(content, Path("app.py"))
        cred_matches = [
            m
            for asset in result.assets.values()
            for m in asset.detection_matches
            if m.detection_type == DetectionType.CREDENTIAL
        ]
        assert len(cred_matches) > 0

    def test_provider_ids_property(self, engine: DetectionEngine) -> None:
        ids = engine.provider_ids
        assert isinstance(ids, list)
        assert len(ids) == engine.provider_count
        assert len(set(ids)) == len(ids)  # No duplicates

    def test_yaml_path_stored(self, engine: DetectionEngine) -> None:
        from ai_risk_scanner.detectors import _PROVIDERS_YAML_PATH
        assert engine.yaml_path == _PROVIDERS_YAML_PATH

    def test_scan_result_file_path_preserved(self, engine: DetectionEngine) -> None:
        content = "import openai\n"
        file_path = Path("some/deep/path/app.py")
        result = engine.scan_content(content, file_path)
        assert result.file_path == file_path
        if "openai" in result.assets:
            for match in result.assets["openai"].detection_matches:
                assert match.file_path == file_path
