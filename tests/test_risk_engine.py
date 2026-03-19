"""Unit tests for the ai_risk_scanner.risk_engine module.

Tests cover risk score computation, CTEM modifier application, compliance gap
generation, NIST AI RMF and CTEM framework mapping, provider asset analysis,
credential asset analysis, and overall risk aggregation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_risk_scanner.inventory import (
    AIAsset,
    ComplianceGap,
    DetectionMatch,
    DetectionType,
    ProviderCategory,
    RiskFinding,
    RiskLevel,
)
from ai_risk_scanner.risk_engine import (
    RiskEngine,
    RiskScoringConfig,
    _BASE_SCORES,
    _CREDENTIAL_SCORE_INCREMENT,
    _DATA_RESIDENCY_INCREMENT,
    _VENDOR_LOCK_IN_INCREMENT,
    apply_ctem_modifiers,
    compute_base_score,
    score_to_risk_level,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_asset(
    provider_id: str = "openai",
    provider_name: str = "OpenAI",
    category: ProviderCategory = ProviderCategory.LLM,
    risk_level: RiskLevel = RiskLevel.HIGH,
    vendor_lock_in: bool = True,
    data_residency_concern: bool = True,
    nist_rmf_functions: list[str] | None = None,
    ctem_categories: list[str] | None = None,
    compliance_notes: list[str] | None = None,
    docs_url: str = "https://platform.openai.com/docs",
) -> AIAsset:
    """Create an AIAsset with sensible defaults for testing."""
    return AIAsset(
        provider_id=provider_id,
        provider_name=provider_name,
        category=category,
        baseline_risk_level=risk_level,
        vendor_lock_in=vendor_lock_in,
        data_residency_concern=data_residency_concern,
        nist_rmf_functions=nist_rmf_functions or ["GOVERN-1", "MAP-1", "MAP-5", "MEASURE-2"],
        ctem_categories=ctem_categories or ["external_exposure", "data_exfiltration", "third_party_dependency"],
        compliance_notes=compliance_notes or ["Review data retention.", "Ensure API key rotation."],
        docs_url=docs_url,
    )


def _add_import_match(asset: AIAsset, file_path: str = "app.py", line: int = 1) -> DetectionMatch:
    """Add an IMPORT detection match to an asset and return the match."""
    match = DetectionMatch(
        file_path=Path(file_path),
        line_number=line,
        line_content=f"import {asset.provider_id}",
        matched_pattern=f"import {asset.provider_id}",
        detection_type=DetectionType.IMPORT,
        provider_id=asset.provider_id,
    )
    asset.add_match(match)
    return match


def _add_env_var_match(asset: AIAsset, file_path: str = ".env", line: int = 1) -> DetectionMatch:
    """Add an ENV_VAR detection match to an asset and return the match."""
    match = DetectionMatch(
        file_path=Path(file_path),
        line_number=line,
        line_content=f"{asset.provider_id.upper()}_API_KEY=placeholder",
        matched_pattern=f"{asset.provider_id.upper()}_API_KEY",
        detection_type=DetectionType.ENV_VAR,
        provider_id=asset.provider_id,
    )
    asset.add_match(match)
    return match


def _add_credential_match(asset: AIAsset, file_path: str = "config.py", line: int = 5) -> DetectionMatch:
    """Add a CREDENTIAL detection match to an asset and return the match."""
    match = DetectionMatch(
        file_path=Path(file_path),
        line_number=line,
        line_content=f'{asset.provider_id.upper()}_API_KEY = "sk-" + "a" * 30',
        matched_pattern=r"sk-[A-Za-z0-9]{20,}",
        detection_type=DetectionType.CREDENTIAL,
        provider_id=asset.provider_id,
    )
    asset.add_match(match)
    return match


def _add_sdk_call_match(asset: AIAsset, file_path: str = "app.py", line: int = 10) -> DetectionMatch:
    """Add a SDK_CALL detection match to an asset and return the match."""
    match = DetectionMatch(
        file_path=Path(file_path),
        line_number=line,
        line_content=f"client = {asset.provider_name}()",
        matched_pattern=f"{asset.provider_name}(",
        detection_type=DetectionType.SDK_CALL,
        provider_id=asset.provider_id,
    )
    asset.add_match(match)
    return match


# ---------------------------------------------------------------------------
# compute_base_score tests
# ---------------------------------------------------------------------------


class TestComputeBaseScore:
    """Tests for the compute_base_score function."""

    def test_critical_returns_highest(self) -> None:
        score = compute_base_score(RiskLevel.CRITICAL)
        assert score == _BASE_SCORES[RiskLevel.CRITICAL]
        assert score >= 9.0

    def test_high_returns_correct_score(self) -> None:
        score = compute_base_score(RiskLevel.HIGH)
        assert score == _BASE_SCORES[RiskLevel.HIGH]
        assert score >= 7.0

    def test_medium_returns_correct_score(self) -> None:
        score = compute_base_score(RiskLevel.MEDIUM)
        assert score == _BASE_SCORES[RiskLevel.MEDIUM]
        assert score >= 5.0

    def test_low_returns_correct_score(self) -> None:
        score = compute_base_score(RiskLevel.LOW)
        assert score == _BASE_SCORES[RiskLevel.LOW]
        assert score >= 2.0

    def test_info_returns_lowest(self) -> None:
        score = compute_base_score(RiskLevel.INFO)
        assert score == _BASE_SCORES[RiskLevel.INFO]
        assert score >= 1.0

    def test_scores_are_ordered(self) -> None:
        assert compute_base_score(RiskLevel.INFO) < compute_base_score(RiskLevel.LOW)
        assert compute_base_score(RiskLevel.LOW) < compute_base_score(RiskLevel.MEDIUM)
        assert compute_base_score(RiskLevel.MEDIUM) < compute_base_score(RiskLevel.HIGH)
        assert compute_base_score(RiskLevel.HIGH) < compute_base_score(RiskLevel.CRITICAL)

    def test_all_scores_in_valid_range(self) -> None:
        for level in RiskLevel:
            score = compute_base_score(level)
            assert 0.0 <= score <= 10.0, f"Score {score} out of range for {level}"

    def test_returns_float(self) -> None:
        for level in RiskLevel:
            assert isinstance(compute_base_score(level), float)


# ---------------------------------------------------------------------------
# score_to_risk_level tests
# ---------------------------------------------------------------------------


class TestScoreToRiskLevel:
    """Tests for the score_to_risk_level function."""

    def test_high_score_maps_to_critical(self) -> None:
        assert score_to_risk_level(9.0) == RiskLevel.CRITICAL
        assert score_to_risk_level(10.0) == RiskLevel.CRITICAL
        assert score_to_risk_level(8.5) == RiskLevel.CRITICAL

    def test_medium_high_score_maps_to_high(self) -> None:
        assert score_to_risk_level(7.0) == RiskLevel.HIGH
        assert score_to_risk_level(8.4) == RiskLevel.HIGH
        assert score_to_risk_level(6.5) == RiskLevel.HIGH

    def test_mid_score_maps_to_medium(self) -> None:
        assert score_to_risk_level(5.0) == RiskLevel.MEDIUM
        assert score_to_risk_level(4.0) == RiskLevel.MEDIUM
        assert score_to_risk_level(6.4) == RiskLevel.MEDIUM

    def test_low_score_maps_to_low(self) -> None:
        assert score_to_risk_level(2.5) == RiskLevel.LOW
        assert score_to_risk_level(1.0) == RiskLevel.LOW
        assert score_to_risk_level(3.9) == RiskLevel.LOW

    def test_zero_maps_to_info(self) -> None:
        assert score_to_risk_level(0.0) == RiskLevel.INFO

    def test_boundary_values(self) -> None:
        # These should be just at or above thresholds
        from ai_risk_scanner.risk_engine import (
            _CRITICAL_THRESHOLD,
            _HIGH_THRESHOLD,
            _MEDIUM_THRESHOLD,
        )
        assert score_to_risk_level(_CRITICAL_THRESHOLD) == RiskLevel.CRITICAL
        assert score_to_risk_level(_HIGH_THRESHOLD) == RiskLevel.HIGH
        assert score_to_risk_level(_MEDIUM_THRESHOLD) == RiskLevel.MEDIUM

    def test_all_levels_reachable(self) -> None:
        """Verify that all RiskLevel values can be returned by score_to_risk_level."""
        reachable = {
            score_to_risk_level(0.0),
            score_to_risk_level(1.5),
            score_to_risk_level(5.0),
            score_to_risk_level(7.0),
            score_to_risk_level(9.0),
        }
        assert RiskLevel.INFO in reachable
        assert RiskLevel.LOW in reachable
        assert RiskLevel.MEDIUM in reachable
        assert RiskLevel.HIGH in reachable
        assert RiskLevel.CRITICAL in reachable


# ---------------------------------------------------------------------------
# apply_ctem_modifiers tests
# ---------------------------------------------------------------------------


class TestApplyCtemModifiers:
    """Tests for the apply_ctem_modifiers function."""

    def _sample_ctem_metadata(self) -> dict[str, dict[str, Any]]:
        """Return a sample CTEM metadata dict for isolated testing."""
        return {
            "external_exposure": {"severity_modifier": 1.2},
            "data_exfiltration": {"severity_modifier": 1.5},
            "third_party_dependency": {"severity_modifier": 1.1},
            "biometric_data": {"severity_modifier": 1.8},
            "cloud_dependency": {"severity_modifier": 1.0},
            "supply_chain": {"severity_modifier": 1.2},
            "identity_access": {"severity_modifier": 1.3},
            "open_source_risk": {"severity_modifier": 0.9},
        }

    def test_no_categories_returns_base_score(self) -> None:
        result = apply_ctem_modifiers(5.0, [], {})
        assert result == 5.0

    def test_empty_metadata_returns_base_score(self) -> None:
        result = apply_ctem_modifiers(5.0, ["external_exposure"], {})
        assert result == 5.0

    def test_single_modifier_applied(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        # external_exposure has modifier 1.2
        result = apply_ctem_modifiers(5.0, ["external_exposure"], ctem_meta)
        assert abs(result - 6.0) < 0.01

    def test_highest_modifier_used(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        # data_exfiltration (1.5) is higher than external_exposure (1.2)
        result = apply_ctem_modifiers(5.0, ["external_exposure", "data_exfiltration"], ctem_meta)
        assert abs(result - 7.5) < 0.01  # 5.0 * 1.5

    def test_biometric_modifier_highest(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        result = apply_ctem_modifiers(
            5.0,
            ["external_exposure", "data_exfiltration", "biometric_data"],
            ctem_meta,
        )
        assert abs(result - 9.0) < 0.01  # 5.0 * 1.8

    def test_result_clamped_to_ten(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        # 9.0 * 1.8 = 16.2 → clamped to 10.0
        result = apply_ctem_modifiers(9.0, ["biometric_data"], ctem_meta)
        assert result == 10.0

    def test_result_not_below_zero(self) -> None:
        ctem_meta = {"open_source_risk": {"severity_modifier": 0.1}}
        result = apply_ctem_modifiers(0.5, ["open_source_risk"], ctem_meta)
        assert result >= 0.0

    def test_modifier_of_one_unchanged(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        result = apply_ctem_modifiers(5.0, ["cloud_dependency"], ctem_meta)
        assert abs(result - 5.0) < 0.01

    def test_unknown_category_uses_modifier_of_one(self) -> None:
        ctem_meta = self._sample_ctem_metadata()
        # unknown_category not in metadata → modifier defaults to 1.0
        result = apply_ctem_modifiers(5.0, ["unknown_category"], ctem_meta)
        assert abs(result - 5.0) < 0.01

    def test_returns_float(self) -> None:
        result = apply_ctem_modifiers(5.0, [], {})
        assert isinstance(result, float)

    def test_negative_base_score_clamped_to_zero(self) -> None:
        result = apply_ctem_modifiers(-1.0, [], {})
        assert result == 0.0


# ---------------------------------------------------------------------------
# RiskScoringConfig tests
# ---------------------------------------------------------------------------


class TestRiskScoringConfig:
    """Tests for the RiskScoringConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = RiskScoringConfig()
        assert cfg.credential_score_increment == _CREDENTIAL_SCORE_INCREMENT
        assert cfg.vendor_lock_in_increment == _VENDOR_LOCK_IN_INCREMENT
        assert cfg.data_residency_increment == _DATA_RESIDENCY_INCREMENT
        assert cfg.apply_ctem_modifiers is True

    def test_custom_values(self) -> None:
        cfg = RiskScoringConfig(
            credential_score_increment=3.0,
            vendor_lock_in_increment=0.5,
            data_residency_increment=0.6,
            apply_ctem_modifiers=False,
        )
        assert cfg.credential_score_increment == 3.0
        assert cfg.vendor_lock_in_increment == 0.5
        assert cfg.data_residency_increment == 0.6
        assert cfg.apply_ctem_modifiers is False

    def test_multi_file_increment_defaults(self) -> None:
        cfg = RiskScoringConfig()
        assert cfg.multi_file_increment_base > 0.0
        assert cfg.multi_file_increment_max > 0.0
        assert cfg.multi_file_increment_max >= cfg.multi_file_increment_base


# ---------------------------------------------------------------------------
# RiskEngine initialization tests
# ---------------------------------------------------------------------------


class TestRiskEngineInit:
    """Tests for RiskEngine initialization and metadata loading."""

    def test_loads_from_default_yaml(self) -> None:
        engine = RiskEngine()
        assert len(engine.ctem_metadata) > 0
        assert len(engine.nist_metadata) > 0

    def test_ctem_metadata_contains_known_categories(self) -> None:
        engine = RiskEngine()
        assert "external_exposure" in engine.ctem_metadata
        assert "data_exfiltration" in engine.ctem_metadata
        assert "third_party_dependency" in engine.ctem_metadata
        assert "biometric_data" in engine.ctem_metadata

    def test_nist_metadata_contains_known_functions(self) -> None:
        engine = RiskEngine()
        assert "GOVERN-1" in engine.nist_metadata
        assert "MAP-1" in engine.nist_metadata
        assert "MEASURE-2" in engine.nist_metadata
        assert "MANAGE-2" in engine.nist_metadata

    def test_ctem_metadata_has_severity_modifiers(self) -> None:
        engine = RiskEngine()
        for cat_id, cat_data in engine.ctem_metadata.items():
            assert "severity_modifier" in cat_data, (
                f"CTEM category '{cat_id}' missing severity_modifier"
            )

    def test_nist_metadata_has_name_and_description(self) -> None:
        engine = RiskEngine()
        for func_id, func_data in engine.nist_metadata.items():
            assert "name" in func_data, f"NIST function '{func_id}' missing name"
            assert "description" in func_data, f"NIST function '{func_id}' missing description"

    def test_custom_yaml_path(self, tmp_path: Path) -> None:
        """Engine initializes gracefully with a minimal custom YAML."""
        data = {
            "providers": [],
            "credential_patterns": [],
            "ctem_categories": {
                "test_cat": {"name": "Test", "description": "Test cat", "severity_modifier": 1.0}
            },
            "nist_rmf_functions": {
                "TEST-1": {"name": "Test", "description": "Test function"}
            },
        }
        yaml_file = tmp_path / "providers.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        engine = RiskEngine(yaml_path=yaml_file)
        assert "test_cat" in engine.ctem_metadata
        assert "TEST-1" in engine.nist_metadata

    def test_missing_yaml_logs_warning_but_does_not_raise(self, tmp_path: Path) -> None:
        """Engine should not raise even if yaml file is missing (degrades gracefully)."""
        missing = tmp_path / "nonexistent.yaml"
        # Should not raise
        engine = RiskEngine(yaml_path=missing)
        assert engine.ctem_metadata == {}
        assert engine.nist_metadata == {}

    def test_custom_scoring_config(self) -> None:
        cfg = RiskScoringConfig(credential_score_increment=5.0)
        engine = RiskEngine(scoring_config=cfg)
        assert engine.scoring_config.credential_score_increment == 5.0

    def test_default_yaml_path_set(self) -> None:
        engine = RiskEngine()
        from ai_risk_scanner.risk_engine import _PROVIDERS_YAML_PATH
        assert engine.yaml_path == _PROVIDERS_YAML_PATH

    def test_repr(self) -> None:
        engine = RiskEngine()
        r = repr(engine)
        assert "RiskEngine" in r
        assert "ctem_categories" in r
        assert "nist_functions" in r


# ---------------------------------------------------------------------------
# RiskEngine.analyze tests — basic behavior
# ---------------------------------------------------------------------------


class TestRiskEngineAnalyzeBasic:
    """Tests for the core RiskEngine.analyze method behavior."""

    def test_analyze_empty_assets_returns_empty_list(self) -> None:
        engine = RiskEngine()
        findings = engine.analyze({})
        assert findings == []

    def test_analyze_single_provider_returns_one_finding(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings) == 1

    def test_analyze_asset_with_no_matches_returns_no_finding(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()  # no matches added
        findings = engine.analyze({"openai": asset})
        assert findings == []

    def test_analyze_multiple_providers(self) -> None:
        engine = RiskEngine()
        assets = {}
        for pid in ["openai", "anthropic", "cohere"]:
            a = _make_asset(provider_id=pid, provider_name=pid.capitalize())
            _add_import_match(a)
            assets[pid] = a
        findings = engine.analyze(assets)
        assert len(findings) == 3

    def test_findings_sorted_by_score_descending(self) -> None:
        engine = RiskEngine()
        # openai: high risk, anthropic: medium risk
        high_asset = _make_asset("openai", risk_level=RiskLevel.HIGH)
        _add_import_match(high_asset)
        med_asset = _make_asset("anthropic", "Anthropic", risk_level=RiskLevel.MEDIUM)
        _add_import_match(med_asset)
        findings = engine.analyze({"openai": high_asset, "anthropic": med_asset})
        assert len(findings) == 2
        assert findings[0].risk_score >= findings[1].risk_score

    def test_finding_has_correct_provider_id(self) -> None:
        engine = RiskEngine()
        asset = _make_asset("openai")
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].provider_id == "openai"

    def test_finding_has_correct_provider_name(self) -> None:
        engine = RiskEngine()
        asset = _make_asset("openai", "OpenAI")
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].provider_name == "OpenAI"

    def test_finding_risk_score_is_positive(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].risk_score > 0.0

    def test_finding_risk_score_in_valid_range(self) -> None:
        engine = RiskEngine()
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            asset = _make_asset(risk_level=level)
            _add_import_match(asset)
            findings = engine.analyze({"openai": asset})
            assert 0.0 <= findings[0].risk_score <= 10.0

    def test_finding_has_detection_matches(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_env_var_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings[0].detection_matches) == 2

    def test_finding_has_compliance_gaps(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings[0].compliance_gaps) > 0

    def test_finding_inherits_nist_functions(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(nist_rmf_functions=["GOVERN-1", "MAP-1"])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert "GOVERN-1" in findings[0].nist_functions
        assert "MAP-1" in findings[0].nist_functions

    def test_finding_inherits_ctem_categories(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(ctem_categories=["external_exposure", "data_exfiltration"])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert "external_exposure" in findings[0].ctem_categories
        assert "data_exfiltration" in findings[0].ctem_categories

    def test_finding_vendor_lock_in_flag(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(vendor_lock_in=True)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].vendor_lock_in is True

    def test_finding_data_residency_flag(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=True)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].data_residency_concern is True

    def test_finding_without_vendor_lock_in(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(vendor_lock_in=False)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].vendor_lock_in is False

    def test_finding_without_data_residency(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=False)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].data_residency_concern is False


# ---------------------------------------------------------------------------
# Risk score computation tests
# ---------------------------------------------------------------------------


class TestRiskScoreComputation:
    """Tests for the risk score calculation logic within RiskEngine."""

    def test_critical_baseline_produces_high_score(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(risk_level=RiskLevel.CRITICAL, ctem_categories=[])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].risk_score >= compute_base_score(RiskLevel.CRITICAL)

    def test_low_baseline_produces_lower_score_than_high(self) -> None:
        engine = RiskEngine()
        high_asset = _make_asset("openai", risk_level=RiskLevel.HIGH, ctem_categories=[], vendor_lock_in=False, data_residency_concern=False)
        _add_import_match(high_asset)
        low_asset = _make_asset("anthropic", "Anthropic", risk_level=RiskLevel.LOW, ctem_categories=[], vendor_lock_in=False, data_residency_concern=False)
        _add_import_match(low_asset)
        high_findings = engine.analyze({"openai": high_asset})
        low_findings = engine.analyze({"anthropic": low_asset})
        assert high_findings[0].risk_score > low_findings[0].risk_score

    def test_credential_exposure_increases_score(self) -> None:
        engine = RiskEngine()
        # Asset without credentials
        asset_clean = _make_asset("openai", vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_clean)
        findings_clean = engine.analyze({"openai": asset_clean})
        score_clean = findings_clean[0].risk_score

        # Asset with credentials
        asset_cred = _make_asset("openai", vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_cred)
        _add_credential_match(asset_cred)
        findings_cred = engine.analyze({"openai": asset_cred})
        score_cred = findings_cred[0].risk_score

        assert score_cred > score_clean

    def test_credential_increment_applied_correctly(self) -> None:
        cfg = RiskScoringConfig(apply_ctem_modifiers=False, vendor_lock_in_increment=0.0, data_residency_increment=0.0)
        engine = RiskEngine(scoring_config=cfg)

        asset_clean = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_clean)
        findings_clean = engine.analyze({"openai": asset_clean})
        score_clean = findings_clean[0].risk_score

        asset_cred = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_cred)
        _add_credential_match(asset_cred)
        findings_cred = engine.analyze({"openai": asset_cred})
        score_cred = findings_cred[0].risk_score

        expected_increment = cfg.credential_score_increment
        actual_increment = score_cred - score_clean
        assert abs(actual_increment - expected_increment) < 0.01

    def test_vendor_lock_in_increases_score(self) -> None:
        cfg = RiskScoringConfig(apply_ctem_modifiers=False, data_residency_increment=0.0)
        engine = RiskEngine(scoring_config=cfg)

        asset_no_lock = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_no_lock)
        findings_no = engine.analyze({"openai": asset_no_lock})

        asset_lock = _make_asset(vendor_lock_in=True, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_lock)
        findings_lock = engine.analyze({"openai": asset_lock})

        assert findings_lock[0].risk_score > findings_no[0].risk_score

    def test_data_residency_increases_score(self) -> None:
        cfg = RiskScoringConfig(apply_ctem_modifiers=False, vendor_lock_in_increment=0.0)
        engine = RiskEngine(scoring_config=cfg)

        asset_no_dr = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset_no_dr)
        findings_no = engine.analyze({"openai": asset_no_dr})

        asset_dr = _make_asset(vendor_lock_in=False, data_residency_concern=True, ctem_categories=[])
        _add_import_match(asset_dr)
        findings_dr = engine.analyze({"openai": asset_dr})

        assert findings_dr[0].risk_score > findings_no[0].risk_score

    def test_multi_file_spread_increases_score(self) -> None:
        cfg = RiskScoringConfig(apply_ctem_modifiers=False, vendor_lock_in_increment=0.0, data_residency_increment=0.0)
        engine = RiskEngine(scoring_config=cfg)

        # Single file
        single = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(single, "file_a.py")
        findings_single = engine.analyze({"openai": single})

        # Multiple files
        multi = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        for i in range(5):
            _add_import_match(multi, f"file_{i}.py", i + 1)
        findings_multi = engine.analyze({"openai": multi})

        assert findings_multi[0].risk_score >= findings_single[0].risk_score

    def test_multi_file_increment_capped(self) -> None:
        cfg = RiskScoringConfig(
            apply_ctem_modifiers=False,
            vendor_lock_in_increment=0.0,
            data_residency_increment=0.0,
            multi_file_increment_base=0.1,
            multi_file_increment_max=0.5,
        )
        engine = RiskEngine(scoring_config=cfg)

        # 100 files — increment should be capped at 0.5
        asset = _make_asset(vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        for i in range(100):
            _add_import_match(asset, f"file_{i}.py", i + 1)

        findings = engine.analyze({"openai": asset})
        # The increment should not exceed max
        base = compute_base_score(RiskLevel.HIGH)
        assert findings[0].risk_score <= min(10.0, base + cfg.multi_file_increment_max) + 0.01

    def test_ctem_modifiers_disabled(self) -> None:
        cfg = RiskScoringConfig(apply_ctem_modifiers=False)
        engine = RiskEngine(scoring_config=cfg)

        asset = _make_asset(
            ctem_categories=["biometric_data"],
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})

        # Score should be the base score only (no CTEM modifier)
        base = compute_base_score(RiskLevel.HIGH)
        assert abs(findings[0].risk_score - base) < 0.01

    def test_ctem_modifiers_applied_when_enabled(self) -> None:
        engine = RiskEngine()  # apply_ctem_modifiers=True by default

        asset_no_ctem = _make_asset(
            ctem_categories=[],
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_import_match(asset_no_ctem)
        findings_no_ctem = engine.analyze({"openai": asset_no_ctem})

        asset_with_ctem = _make_asset(
            ctem_categories=["data_exfiltration"],  # modifier > 1.0
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_import_match(asset_with_ctem)
        findings_with_ctem = engine.analyze({"openai": asset_with_ctem})

        # With data_exfiltration (modifier 1.5), score should be higher
        assert findings_with_ctem[0].risk_score >= findings_no_ctem[0].risk_score

    def test_score_clamped_to_ten(self) -> None:
        """Even with many aggravating factors, score should not exceed 10.0."""
        engine = RiskEngine()
        asset = _make_asset(
            risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=True,
            data_residency_concern=True,
            ctem_categories=["biometric_data", "data_exfiltration", "identity_access"],
        )
        for i in range(20):
            _add_import_match(asset, f"file_{i}.py", i + 1)
        _add_credential_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].risk_score <= 10.0

    def test_risk_level_derived_from_score(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(risk_level=RiskLevel.LOW, vendor_lock_in=False, data_residency_concern=False, ctem_categories=[])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        # Risk level should be consistent with the computed score
        expected_level = score_to_risk_level(findings[0].risk_score)
        assert findings[0].risk_level == expected_level

    def test_finding_id_format(self) -> None:
        engine = RiskEngine()
        asset = _make_asset("openai")
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].finding_id.startswith("finding_openai_")

    def test_finding_title_contains_provider_name(self) -> None:
        engine = RiskEngine()
        asset = _make_asset("openai", "OpenAI")
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert "OpenAI" in findings[0].title

    def test_finding_description_not_empty(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings[0].description) > 0


# ---------------------------------------------------------------------------
# Credential asset analysis tests
# ---------------------------------------------------------------------------


class TestCredentialAssetAnalysis:
    """Tests for RiskEngine handling of the generic '_credentials' asset."""

    def test_credential_asset_produces_critical_finding(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert len(findings) == 1
        assert findings[0].risk_level == RiskLevel.CRITICAL

    def test_credential_asset_produces_near_max_score(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert findings[0].risk_score >= 9.0

    def test_credential_asset_has_hardcoded_credentials_flag(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert findings[0].has_hardcoded_credentials is True

    def test_credential_asset_finding_id(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert findings[0].finding_id == "finding_hardcoded_credentials"

    def test_credential_asset_no_matches_returns_no_finding(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        # No matches added
        findings = engine.analyze({"_credentials": cred_asset})
        assert findings == []

    def test_credential_finding_nist_functions(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert "GOVERN-1" in findings[0].nist_functions
        assert "MANAGE-2" in findings[0].nist_functions

    def test_credential_finding_ctem_categories(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert "identity_access" in findings[0].ctem_categories

    def test_credential_finding_has_compliance_gap(self) -> None:
        engine = RiskEngine()
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)
        findings = engine.analyze({"_credentials": cred_asset})
        assert len(findings[0].compliance_gaps) > 0
        gap = findings[0].compliance_gaps[0]
        assert gap.risk_level == RiskLevel.CRITICAL
        assert len(gap.remediation) > 0

    def test_mixed_provider_and_credential_assets(self) -> None:
        engine = RiskEngine()
        provider_asset = _make_asset()
        _add_import_match(provider_asset)

        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        _add_credential_match(cred_asset)

        findings = engine.analyze({"openai": provider_asset, "_credentials": cred_asset})
        assert len(findings) == 2
        finding_ids = {f.finding_id for f in findings}
        assert "finding_hardcoded_credentials" in finding_ids


# ---------------------------------------------------------------------------
# Compliance gap generation tests
# ---------------------------------------------------------------------------


class TestComplianceGapGeneration:
    """Tests for the compliance gap generation logic."""

    def test_data_residency_gap_generated(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=True)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert any("data_residency" in gid for gid in gap_ids)

    def test_vendor_lock_in_gap_generated(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(vendor_lock_in=True)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert any("vendor_lock_in" in gid for gid in gap_ids)

    def test_credential_gap_generated_when_credentials_detected(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_credential_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert any("hardcoded_credentials" in gid for gid in gap_ids)

    def test_no_credential_gap_without_credential_detection(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_env_var_match(asset)  # env var but no actual hardcoded credential
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert not any("hardcoded_credentials" in gid for gid in gap_ids)

    def test_no_env_var_gap_when_only_import_detected(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)  # only import, no env var
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert any("no_env_var" in gid for gid in gap_ids)

    def test_no_env_var_gap_when_env_var_detected(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_env_var_match(asset)  # env var present → no gap
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert not any("no_env_var" in gid for gid in gap_ids)

    def test_compliance_notes_generate_gaps(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(compliance_notes=["Note one.", "Note two."])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        note_gaps = [
            g for g in findings[0].compliance_gaps
            if "compliance_note" in g.gap_id
        ]
        assert len(note_gaps) == 2

    def test_compliance_gap_has_risk_level(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        for gap in findings[0].compliance_gaps:
            assert isinstance(gap.risk_level, RiskLevel)

    def test_compliance_gap_has_remediation(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=True)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        for gap in findings[0].compliance_gaps:
            assert isinstance(gap.remediation, str)

    def test_compliance_gap_has_nist_functions(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        for gap in findings[0].compliance_gaps:
            assert isinstance(gap.nist_functions, list)

    def test_credential_gap_is_critical_severity(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_credential_match(asset)
        findings = engine.analyze({"openai": asset})
        cred_gaps = [
            g for g in findings[0].compliance_gaps
            if "hardcoded_credentials" in g.gap_id
        ]
        assert len(cred_gaps) > 0
        assert cred_gaps[0].risk_level == RiskLevel.CRITICAL

    def test_data_residency_gap_is_high_severity(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=True, vendor_lock_in=False)
        _add_import_match(asset)
        _add_env_var_match(asset)
        findings = engine.analyze({"openai": asset})
        dr_gaps = [
            g for g in findings[0].compliance_gaps
            if "data_residency" in g.gap_id
        ]
        assert len(dr_gaps) > 0
        assert dr_gaps[0].risk_level == RiskLevel.HIGH

    def test_vendor_lock_in_gap_is_medium_severity(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(vendor_lock_in=True, data_residency_concern=False)
        _add_import_match(asset)
        _add_env_var_match(asset)
        findings = engine.analyze({"openai": asset})
        lock_gaps = [
            g for g in findings[0].compliance_gaps
            if "vendor_lock_in" in g.gap_id
        ]
        assert len(lock_gaps) > 0
        assert lock_gaps[0].risk_level == RiskLevel.MEDIUM

    def test_no_data_residency_gap_when_no_concern(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(data_residency_concern=False)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert not any("data_residency" in gid for gid in gap_ids)

    def test_no_vendor_lock_in_gap_when_no_lock_in(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(vendor_lock_in=False)
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert not any("vendor_lock_in" in gid for gid in gap_ids)

    def test_finding_has_hardcoded_credentials_flag(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        _add_credential_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].has_hardcoded_credentials is True

    def test_finding_no_hardcoded_credentials_flag_when_clean(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert findings[0].has_hardcoded_credentials is False


# ---------------------------------------------------------------------------
# NIST and CTEM mapping tests
# ---------------------------------------------------------------------------


class TestFrameworkMappings:
    """Tests for NIST AI RMF and CTEM mapping helper methods."""

    def test_build_nist_mapping_empty_findings(self) -> None:
        engine = RiskEngine()
        mapping = engine.build_nist_mapping([])
        assert mapping == {}

    def test_build_nist_mapping_single_finding(self) -> None:
        engine = RiskEngine()
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test finding",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            nist_functions=["GOVERN-1", "MAP-1"],
        )
        mapping = engine.build_nist_mapping([finding])
        assert "GOVERN-1" in mapping
        assert "MAP-1" in mapping
        assert "f1" in mapping["GOVERN-1"]
        assert "f1" in mapping["MAP-1"]

    def test_build_nist_mapping_multiple_findings(self) -> None:
        engine = RiskEngine()
        finding1 = RiskFinding(
            finding_id="f1",
            title="F1",
            description="Finding 1",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            nist_functions=["GOVERN-1", "MAP-1"],
        )
        finding2 = RiskFinding(
            finding_id="f2",
            title="F2",
            description="Finding 2",
            risk_level=RiskLevel.MEDIUM,
            risk_score=5.0,
            provider_id="anthropic",
            provider_name="Anthropic",
            nist_functions=["GOVERN-1", "MEASURE-2"],
        )
        mapping = engine.build_nist_mapping([finding1, finding2])
        # Both findings share GOVERN-1
        assert len(mapping["GOVERN-1"]) == 2
        assert "f1" in mapping["GOVERN-1"]
        assert "f2" in mapping["GOVERN-1"]
        # MAP-1 only in f1
        assert "MAP-1" in mapping
        assert len(mapping["MAP-1"]) == 1
        # MEASURE-2 only in f2
        assert "MEASURE-2" in mapping

    def test_build_ctem_mapping_empty_findings(self) -> None:
        engine = RiskEngine()
        mapping = engine.build_ctem_mapping([])
        assert mapping == {}

    def test_build_ctem_mapping_single_finding(self) -> None:
        engine = RiskEngine()
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test finding",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            ctem_categories=["external_exposure", "data_exfiltration"],
        )
        mapping = engine.build_ctem_mapping([finding])
        assert "external_exposure" in mapping
        assert "data_exfiltration" in mapping
        assert "f1" in mapping["external_exposure"]

    def test_build_ctem_mapping_multiple_findings(self) -> None:
        engine = RiskEngine()
        finding1 = RiskFinding(
            finding_id="f1",
            title="F1",
            description="F1",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            ctem_categories=["external_exposure", "data_exfiltration"],
        )
        finding2 = RiskFinding(
            finding_id="f2",
            title="F2",
            description="F2",
            risk_level=RiskLevel.MEDIUM,
            risk_score=5.0,
            provider_id="anthropic",
            provider_name="Anthropic",
            ctem_categories=["external_exposure", "third_party_dependency"],
        )
        mapping = engine.build_ctem_mapping([finding1, finding2])
        assert len(mapping["external_exposure"]) == 2
        assert len(mapping["data_exfiltration"]) == 1
        assert len(mapping["third_party_dependency"]) == 1

    def test_get_nist_function_description_known(self) -> None:
        engine = RiskEngine()
        desc = engine.get_nist_function_description("GOVERN-1")
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "GOVERN-1" not in desc.lower() or "govern" in desc.lower()

    def test_get_nist_function_description_unknown(self) -> None:
        engine = RiskEngine()
        desc = engine.get_nist_function_description("UNKNOWN-99")
        assert "UNKNOWN-99" in desc

    def test_get_nist_function_name_known(self) -> None:
        engine = RiskEngine()
        name = engine.get_nist_function_name("MAP-1")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_nist_function_name_unknown(self) -> None:
        engine = RiskEngine()
        name = engine.get_nist_function_name("NONEXISTENT")
        assert name == "NONEXISTENT"

    def test_get_ctem_category_description_known(self) -> None:
        engine = RiskEngine()
        desc = engine.get_ctem_category_description("external_exposure")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_ctem_category_description_unknown(self) -> None:
        engine = RiskEngine()
        desc = engine.get_ctem_category_description("nonexistent_cat")
        assert "nonexistent_cat" in desc

    def test_get_ctem_category_name_known(self) -> None:
        engine = RiskEngine()
        name = engine.get_ctem_category_name("data_exfiltration")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_ctem_category_name_unknown(self) -> None:
        engine = RiskEngine()
        name = engine.get_ctem_category_name("unknown_cat")
        assert name == "unknown_cat"

    def test_all_nist_functions_have_descriptions(self) -> None:
        engine = RiskEngine()
        for func_id in engine.nist_metadata:
            desc = engine.get_nist_function_description(func_id)
            assert len(desc) > 0, f"Empty description for NIST function {func_id}"

    def test_all_ctem_categories_have_descriptions(self) -> None:
        engine = RiskEngine()
        for cat_id in engine.ctem_metadata:
            desc = engine.get_ctem_category_description(cat_id)
            assert len(desc) > 0, f"Empty description for CTEM category {cat_id}"


# ---------------------------------------------------------------------------
# compute_overall_risk_score tests
# ---------------------------------------------------------------------------


class TestComputeOverallRiskScore:
    """Tests for RiskEngine.compute_overall_risk_score."""

    def test_empty_findings_returns_zero_info(self) -> None:
        engine = RiskEngine()
        score, level = engine.compute_overall_risk_score([])
        assert score == 0.0
        assert level == RiskLevel.INFO

    def test_single_finding_score(self) -> None:
        engine = RiskEngine()
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
        )
        score, level = engine.compute_overall_risk_score([finding])
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0
        assert level == score_to_risk_level(score)

    def test_multiple_findings_averaged(self) -> None:
        engine = RiskEngine()
        findings = [
            RiskFinding(
                finding_id=f"f{i}",
                title=f"Finding {i}",
                description="Test",
                risk_level=RiskLevel.HIGH,
                risk_score=float(score),
                provider_id=f"provider_{i}",
                provider_name=f"Provider {i}",
            )
            for i, score in enumerate([6.0, 8.0, 4.0])
        ]
        score, level = engine.compute_overall_risk_score(findings)
        # Average of 6.0, 8.0, 4.0 = 6.0 (no critical boost)
        assert 0.0 <= score <= 10.0

    def test_critical_finding_boosts_score(self) -> None:
        engine = RiskEngine()
        findings_no_critical = [
            RiskFinding(
                finding_id="f1",
                title="F1",
                description="Test",
                risk_level=RiskLevel.HIGH,
                risk_score=7.0,
                provider_id="openai",
                provider_name="OpenAI",
            ),
        ]
        score_no_crit, _ = engine.compute_overall_risk_score(findings_no_critical)

        findings_with_critical = [
            RiskFinding(
                finding_id="f2",
                title="F2",
                description="Test",
                risk_level=RiskLevel.CRITICAL,
                risk_score=7.0,
                provider_id="openai",
                provider_name="OpenAI",
            ),
        ]
        score_with_crit, _ = engine.compute_overall_risk_score(findings_with_critical)

        assert score_with_crit >= score_no_crit

    def test_overall_score_clamped_to_ten(self) -> None:
        engine = RiskEngine()
        findings = [
            RiskFinding(
                finding_id=f"f{i}",
                title=f"F{i}",
                description="Test",
                risk_level=RiskLevel.CRITICAL,
                risk_score=10.0,
                provider_id=f"p{i}",
                provider_name=f"P{i}",
            )
            for i in range(10)
        ]
        score, level = engine.compute_overall_risk_score(findings)
        assert score <= 10.0

    def test_risk_level_consistent_with_score(self) -> None:
        engine = RiskEngine()
        findings = [
            RiskFinding(
                finding_id="f1",
                title="Test",
                description="Test",
                risk_level=RiskLevel.MEDIUM,
                risk_score=5.0,
                provider_id="openai",
                provider_name="OpenAI",
            ),
        ]
        score, level = engine.compute_overall_risk_score(findings)
        assert level == score_to_risk_level(score)

    def test_returns_tuple(self) -> None:
        engine = RiskEngine()
        result = engine.compute_overall_risk_score([])
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests with the real providers.yaml
# ---------------------------------------------------------------------------


class TestRiskEngineIntegration:
    """Integration tests using a real DetectionEngine and RiskEngine pipeline."""

    def test_openai_asset_from_real_yaml(self) -> None:
        """Verify risk engine produces a finding for a realistic OpenAI asset."""
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\nclient = openai.OpenAI()\nOPENAI_API_KEY=sk-placeholder\n"
        from io import BytesIO
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        assert len(findings) > 0
        openai_findings = [f for f in findings if f.provider_id == "openai"]
        assert len(openai_findings) > 0
        assert openai_findings[0].risk_score > 0.0

    def test_multiple_providers_produce_multiple_findings(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\nimport anthropic\nimport cohere\n"
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        provider_ids = {f.provider_id for f in findings}
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids
        assert "cohere" in provider_ids

    def test_hardcoded_credential_produces_critical_finding(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = f'OPENAI_API_KEY = "sk-{"a" * 30}"\n'
        file_result = det_engine.scan_content(content, Path("config.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        critical_findings = [f for f in findings if f.risk_level == RiskLevel.CRITICAL]
        assert len(critical_findings) > 0

    def test_clean_code_produces_no_findings(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    print('Hello')\n"
        file_result = det_engine.scan_content(content, Path("clean.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)
        assert findings == []

    def test_findings_sorted_by_score_descending_integration(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = (
            "import openai\n"
            "import anthropic\n"
            "import weaviate\n"  # lower risk
        )
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        scores = [f.risk_score for f in findings]
        assert scores == sorted(scores, reverse=True)

    def test_all_findings_have_valid_scores(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\nimport anthropic\nfrom langchain_openai import ChatOpenAI\n"
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        for finding in findings:
            assert 0.0 <= finding.risk_score <= 10.0, (
                f"Score {finding.risk_score} out of range for {finding.provider_id}"
            )

    def test_scan_files_to_risk_findings_pipeline(self, tmp_path: Path) -> None:
        """End-to-end test from files on disk to risk findings."""
        from ai_risk_scanner.detectors import DetectionEngine

        # Write test files
        app_py = tmp_path / "app.py"
        app_py.write_text(
            "import openai\nclient = openai.OpenAI()\n",
            encoding="utf-8",
        )
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-placeholder\nANTHROPIC_API_KEY=placeholder\n",
            encoding="utf-8",
        )

        det_engine = DetectionEngine()
        file_results = det_engine.scan_files([app_py, env_file])
        assets = det_engine.aggregate_results(file_results)

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        assert len(findings) > 0
        provider_ids = {f.provider_id for f in findings}
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

    def test_finding_description_contains_provider_info(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\n"
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)

        openai_findings = [f for f in findings if f.provider_id == "openai"]
        assert len(openai_findings) > 0
        desc = openai_findings[0].description
        assert "OpenAI" in desc

    def test_nist_mapping_from_analyzed_assets(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\nimport anthropic\n"
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)
        nist_mapping = risk_engine.build_nist_mapping(findings)

        # openai and anthropic both map to GOVERN-1
        assert "GOVERN-1" in nist_mapping
        assert len(nist_mapping["GOVERN-1"]) >= 2

    def test_ctem_mapping_from_analyzed_assets(self) -> None:
        from ai_risk_scanner.detectors import DetectionEngine

        det_engine = DetectionEngine()
        content = "import openai\n"
        file_result = det_engine.scan_content(content, Path("app.py"))
        assets = det_engine.aggregate_results([file_result])

        risk_engine = RiskEngine()
        findings = risk_engine.analyze(assets)
        ctem_mapping = risk_engine.build_ctem_mapping(findings)

        # openai has external_exposure and data_exfiltration
        assert "external_exposure" in ctem_mapping or "third_party_dependency" in ctem_mapping


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestRiskEngineEdgeCases:
    """Tests for edge cases and error handling in the risk engine."""

    def test_asset_with_sdk_call_only_gets_no_env_var_gap(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_sdk_call_match(asset)
        findings = engine.analyze({"openai": asset})
        gap_ids = {g.gap_id for g in findings[0].compliance_gaps}
        assert any("no_env_var" in gid for gid in gap_ids)

    def test_asset_with_url_match_only_still_analyzed(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        url_match = DetectionMatch(
            file_path=Path("config.py"),
            line_number=1,
            line_content='endpoint = "https://api.openai.com/v1"',
            matched_pattern="api.openai.com",
            detection_type=DetectionType.URL,
            provider_id="openai",
        )
        asset.add_match(url_match)
        findings = engine.analyze({"openai": asset})
        assert len(findings) == 1

    def test_low_risk_provider_can_still_elevate_to_critical_with_credentials(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(risk_level=RiskLevel.LOW, ctem_categories=["identity_access"])
        _add_import_match(asset)
        _add_credential_match(asset)
        findings = engine.analyze({"openai": asset})
        # Score should be elevated significantly by credential + CTEM modifier
        assert findings[0].risk_score > compute_base_score(RiskLevel.LOW)

    def test_finding_score_rounded_to_two_decimals(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        score_str = str(findings[0].risk_score)
        decimal_places = len(score_str.split(".")[1]) if "." in score_str else 0
        assert decimal_places <= 2

    def test_assets_dict_not_mutated_by_analyze(self) -> None:
        engine = RiskEngine()
        asset = _make_asset()
        _add_import_match(asset)
        original_match_count = asset.total_matches
        assets = {"openai": asset}
        engine.analyze(assets)
        # The asset should not have been mutated
        assert asset.total_matches == original_match_count

    def test_empty_compliance_notes_no_note_gaps(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(compliance_notes=[])
        _add_import_match(asset)
        _add_env_var_match(asset)
        findings = engine.analyze({"openai": asset})
        note_gaps = [
            g for g in findings[0].compliance_gaps
            if "compliance_note" in g.gap_id
        ]
        assert len(note_gaps) == 0

    def test_many_compliance_notes_generate_many_gaps(self) -> None:
        notes = [f"Note {i}." for i in range(10)]
        engine = RiskEngine()
        asset = _make_asset(compliance_notes=notes)
        _add_import_match(asset)
        _add_env_var_match(asset)
        findings = engine.analyze({"openai": asset})
        note_gaps = [
            g for g in findings[0].compliance_gaps
            if "compliance_note" in g.gap_id
        ]
        assert len(note_gaps) == 10

    def test_provider_with_zero_ctem_categories_analyzed(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(ctem_categories=[])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings) == 1
        # Score should be base score ± other increments
        assert findings[0].risk_score > 0.0

    def test_provider_with_zero_nist_functions_analyzed(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(nist_rmf_functions=[])
        _add_import_match(asset)
        findings = engine.analyze({"openai": asset})
        assert len(findings) == 1
        assert findings[0].nist_functions == []

    def test_compliance_gap_provider_id_set_correctly(self) -> None:
        engine = RiskEngine()
        asset = _make_asset(provider_id="mistral", provider_name="Mistral", data_residency_concern=True)
        _add_import_match(asset)
        findings = engine.analyze({"mistral": asset})
        dr_gaps = [
            g for g in findings[0].compliance_gaps
            if "data_residency" in g.gap_id
        ]
        assert len(dr_gaps) > 0
        assert dr_gaps[0].provider_id == "mistral"
