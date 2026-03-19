"""Unit tests for the ai_risk_scanner.inventory module.

Tests cover data model creation, validation, serialization,
and aggregation logic for all inventory data classes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_risk_scanner.inventory import (
    AIAsset,
    CTEMCategory,
    ComplianceGap,
    DetectionMatch,
    DetectionType,
    NISTFunction,
    ProviderCategory,
    RiskFinding,
    RiskLevel,
    ScanMetadata,
    ScanResult,
    ScanSummary,
    make_scan_result,
)


# ---------------------------------------------------------------------------
# RiskLevel tests
# ---------------------------------------------------------------------------


class TestRiskLevel:
    """Tests for the RiskLevel enumeration."""

    def test_values_exist(self) -> None:
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.INFO.value == "info"

    def test_ordering_less_than(self) -> None:
        assert RiskLevel.INFO < RiskLevel.LOW
        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_ordering_greater_than(self) -> None:
        assert RiskLevel.CRITICAL > RiskLevel.HIGH
        assert RiskLevel.HIGH > RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM > RiskLevel.LOW
        assert RiskLevel.LOW > RiskLevel.INFO

    def test_ordering_equal(self) -> None:
        assert RiskLevel.HIGH >= RiskLevel.HIGH
        assert RiskLevel.HIGH <= RiskLevel.HIGH

    def test_numeric_score(self) -> None:
        assert RiskLevel.INFO.numeric_score == 1
        assert RiskLevel.LOW.numeric_score == 2
        assert RiskLevel.MEDIUM.numeric_score == 3
        assert RiskLevel.HIGH.numeric_score == 4
        assert RiskLevel.CRITICAL.numeric_score == 5

    def test_max_of_risk_levels(self) -> None:
        levels = [RiskLevel.LOW, RiskLevel.CRITICAL, RiskLevel.MEDIUM]
        assert max(levels) == RiskLevel.CRITICAL

    def test_from_string_value(self) -> None:
        assert RiskLevel("critical") == RiskLevel.CRITICAL
        assert RiskLevel("high") == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# DetectionMatch tests
# ---------------------------------------------------------------------------


class TestDetectionMatch:
    """Tests for the DetectionMatch dataclass."""

    def test_basic_creation(self) -> None:
        match = DetectionMatch(
            file_path=Path("app.py"),
            line_number=10,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=DetectionType.IMPORT,
            provider_id="openai",
        )
        assert match.file_path == Path("app.py")
        assert match.line_number == 10
        assert match.provider_id == "openai"
        assert match.detection_type == DetectionType.IMPORT

    def test_string_path_converted(self) -> None:
        match = DetectionMatch(
            file_path="src/main.py",
            line_number=1,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=DetectionType.IMPORT,
            provider_id="openai",
        )
        assert isinstance(match.file_path, Path)

    def test_string_detection_type_converted(self) -> None:
        match = DetectionMatch(
            file_path=Path("app.py"),
            line_number=1,
            line_content="OPENAI_API_KEY=sk-abc",
            matched_pattern="OPENAI_API_KEY",
            detection_type="env_var",  # type: ignore[arg-type]
            provider_id="openai",
        )
        assert match.detection_type == DetectionType.ENV_VAR

    def test_invalid_string_detection_type_defaults(self) -> None:
        match = DetectionMatch(
            file_path=Path("app.py"),
            line_number=1,
            line_content="something",
            matched_pattern="something",
            detection_type="not_a_type",  # type: ignore[arg-type]
            provider_id="openai",
        )
        assert match.detection_type == DetectionType.UNKNOWN

    def test_invalid_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="line_number must be >= 1"):
            DetectionMatch(
                file_path=Path("app.py"),
                line_number=0,
                line_content="import openai",
                matched_pattern="import openai",
                detection_type=DetectionType.IMPORT,
                provider_id="openai",
            )

    def test_to_dict(self) -> None:
        match = DetectionMatch(
            file_path=Path("src/app.py"),
            line_number=5,
            line_content="  import openai  ",
            matched_pattern="import openai",
            detection_type=DetectionType.IMPORT,
            provider_id="openai",
            context_lines=["# comment", "import openai", "client = openai.OpenAI()"],
        )
        d = match.to_dict()
        assert d["file_path"] == "src/app.py"
        assert d["line_number"] == 5
        assert d["line_content"] == "import openai"  # stripped
        assert d["detection_type"] == "import"
        assert d["provider_id"] == "openai"
        assert len(d["context_lines"]) == 3

    def test_repr(self) -> None:
        match = DetectionMatch(
            file_path=Path("app.py"),
            line_number=1,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=DetectionType.IMPORT,
            provider_id="openai",
        )
        assert "DetectionMatch" in repr(match)
        assert "openai" in repr(match)


# ---------------------------------------------------------------------------
# ComplianceGap tests
# ---------------------------------------------------------------------------


class TestComplianceGap:
    """Tests for the ComplianceGap dataclass."""

    def test_basic_creation(self) -> None:
        gap = ComplianceGap(
            gap_id="gap_001",
            title="Missing API Key Rotation",
            description="No evidence of API key rotation policy.",
            risk_level=RiskLevel.HIGH,
            provider_id="openai",
        )
        assert gap.gap_id == "gap_001"
        assert gap.risk_level == RiskLevel.HIGH

    def test_string_risk_level_converted(self) -> None:
        gap = ComplianceGap(
            gap_id="gap_002",
            title="Data Residency",
            description="Data may leave jurisdiction.",
            risk_level="critical",  # type: ignore[arg-type]
        )
        assert gap.risk_level == RiskLevel.CRITICAL

    def test_invalid_risk_level_defaults(self) -> None:
        gap = ComplianceGap(
            gap_id="gap_003",
            title="Unknown Risk",
            description="Unknown severity.",
            risk_level="unknown_level",  # type: ignore[arg-type]
        )
        assert gap.risk_level == RiskLevel.INFO

    def test_to_dict(self) -> None:
        gap = ComplianceGap(
            gap_id="gap_001",
            title="Missing Rate Limiting",
            description="No rate limit controls found.",
            risk_level=RiskLevel.MEDIUM,
            nist_functions=["MANAGE-1"],
            ctem_categories=["external_exposure"],
            remediation="Implement token bucket rate limiting.",
            provider_id="openai",
        )
        d = gap.to_dict()
        assert d["gap_id"] == "gap_001"
        assert d["risk_level"] == "medium"
        assert d["nist_functions"] == ["MANAGE-1"]
        assert d["remediation"] == "Implement token bucket rate limiting."

    def test_repr(self) -> None:
        gap = ComplianceGap(
            gap_id="g1",
            title="Test",
            description="Test gap",
            risk_level=RiskLevel.LOW,
        )
        assert "ComplianceGap" in repr(gap)


# ---------------------------------------------------------------------------
# RiskFinding tests
# ---------------------------------------------------------------------------


class TestRiskFinding:
    """Tests for the RiskFinding dataclass."""

    def _make_match(self, line: int = 1, dtype: DetectionType = DetectionType.IMPORT) -> DetectionMatch:
        return DetectionMatch(
            file_path=Path("app.py"),
            line_number=line,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=dtype,
            provider_id="openai",
        )

    def test_basic_creation(self) -> None:
        finding = RiskFinding(
            finding_id="finding_openai_001",
            title="OpenAI Integration Detected",
            description="OpenAI SDK is used in this codebase.",
            risk_level=RiskLevel.HIGH,
            risk_score=7.5,
            provider_id="openai",
            provider_name="OpenAI",
        )
        assert finding.finding_id == "finding_openai_001"
        assert finding.risk_level == RiskLevel.HIGH
        assert finding.risk_score == 7.5
        assert finding.evidence_count == 0

    def test_invalid_risk_score_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_score must be between 0.0 and 10.0"):
            RiskFinding(
                finding_id="f1",
                title="Test",
                description="Test",
                risk_level=RiskLevel.LOW,
                risk_score=11.0,
                provider_id="openai",
                provider_name="OpenAI",
            )

    def test_string_risk_level_converted(self) -> None:
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test",
            risk_level="critical",  # type: ignore[arg-type]
            risk_score=9.0,
            provider_id="openai",
            provider_name="OpenAI",
        )
        assert finding.risk_level == RiskLevel.CRITICAL

    def test_affected_files_deduped(self) -> None:
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            detection_matches=[
                self._make_match(1),
                self._make_match(2),
                self._make_match(3),
            ],
        )
        # All matches point to the same file (app.py)
        assert len(finding.affected_files) == 1
        assert finding.affected_files[0] == Path("app.py")

    def test_evidence_count(self) -> None:
        matches = [self._make_match(i) for i in range(1, 6)]
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test",
            risk_level=RiskLevel.HIGH,
            risk_score=7.0,
            provider_id="openai",
            provider_name="OpenAI",
            detection_matches=matches,
        )
        assert finding.evidence_count == 5

    def test_to_dict(self) -> None:
        finding = RiskFinding(
            finding_id="finding_openai_001",
            title="OpenAI Detected",
            description="OpenAI SDK found.",
            risk_level=RiskLevel.HIGH,
            risk_score=7.5,
            provider_id="openai",
            provider_name="OpenAI",
            nist_functions=["GOVERN-1", "MAP-1"],
            ctem_categories=["external_exposure"],
            vendor_lock_in=True,
            data_residency_concern=True,
            detection_matches=[self._make_match()],
        )
        d = finding.to_dict()
        assert d["finding_id"] == "finding_openai_001"
        assert d["risk_level"] == "high"
        assert d["risk_score"] == 7.5
        assert d["vendor_lock_in"] is True
        assert d["evidence_count"] == 1
        assert len(d["detection_matches"]) == 1

    def test_repr(self) -> None:
        finding = RiskFinding(
            finding_id="f1",
            title="Test",
            description="Test",
            risk_level=RiskLevel.MEDIUM,
            risk_score=5.0,
            provider_id="anthropic",
            provider_name="Anthropic",
        )
        r = repr(finding)
        assert "RiskFinding" in r
        assert "anthropic" in r


# ---------------------------------------------------------------------------
# AIAsset tests
# ---------------------------------------------------------------------------


class TestAIAsset:
    """Tests for the AIAsset dataclass."""

    def _make_match(
        self,
        path: str = "app.py",
        line: int = 1,
        dtype: DetectionType = DetectionType.IMPORT,
    ) -> DetectionMatch:
        return DetectionMatch(
            file_path=Path(path),
            line_number=line,
            line_content="import openai",
            matched_pattern="import openai",
            detection_type=dtype,
            provider_id="openai",
        )

    def test_basic_creation(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        assert asset.provider_id == "openai"
        assert asset.category == ProviderCategory.LLM
        assert asset.total_matches == 0
        assert len(asset.files_detected) == 0

    def test_string_category_converted(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category="llm",  # type: ignore[arg-type]
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        assert asset.category == ProviderCategory.LLM

    def test_invalid_category_defaults(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category="not_a_category",  # type: ignore[arg-type]
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        assert asset.category == ProviderCategory.UNKNOWN

    def test_files_detected_list_converted_to_set(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
            files_detected=["app.py", "config.py"],  # type: ignore[arg-type]
        )
        assert isinstance(asset.files_detected, set)
        assert Path("app.py") in asset.files_detected

    def test_add_match(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        match = self._make_match("app.py", 5)
        asset.add_match(match)
        assert asset.total_matches == 1
        assert Path("app.py") in asset.files_detected

    def test_add_match_multiple_files(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        asset.add_match(self._make_match("app.py", 1))
        asset.add_match(self._make_match("config.py", 2))
        asset.add_match(self._make_match("app.py", 10))
        assert asset.total_matches == 3
        assert len(asset.files_detected) == 2

    def test_has_credential_exposure_true(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        cred_match = DetectionMatch(
            file_path=Path("app.py"),
            line_number=1,
            line_content='OPENAI_API_KEY = "sk-abc123"',
            matched_pattern="sk-[A-Za-z0-9]{20,}",
            detection_type=DetectionType.CREDENTIAL,
            provider_id="openai",
        )
        asset.add_match(cred_match)
        assert asset.has_credential_exposure is True

    def test_has_credential_exposure_false(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        asset.add_match(self._make_match(dtype=DetectionType.IMPORT))
        assert asset.has_credential_exposure is False

    def test_has_env_var_usage(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        env_match = DetectionMatch(
            file_path=Path(".env"),
            line_number=1,
            line_content="OPENAI_API_KEY=\"\"",
            matched_pattern="OPENAI_API_KEY",
            detection_type=DetectionType.ENV_VAR,
            provider_id="openai",
        )
        asset.add_match(env_match)
        assert asset.has_env_var_usage is True

    def test_detection_types_found(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        asset.add_match(self._make_match(dtype=DetectionType.IMPORT))
        asset.add_match(self._make_match(dtype=DetectionType.URL))
        types = asset.detection_types_found
        assert DetectionType.IMPORT in types
        assert DetectionType.URL in types
        assert DetectionType.ENV_VAR not in types

    def test_matches_by_type(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        asset.add_match(self._make_match("a.py", 1, DetectionType.IMPORT))
        asset.add_match(self._make_match("b.py", 2, DetectionType.IMPORT))
        asset.add_match(self._make_match("c.py", 3, DetectionType.URL))
        by_type = asset.matches_by_type
        assert len(by_type[DetectionType.IMPORT]) == 2
        assert len(by_type[DetectionType.URL]) == 1

    def test_to_dict(self) -> None:
        asset = AIAsset(
            provider_id="openai",
            provider_name="OpenAI",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
            nist_rmf_functions=["GOVERN-1"],
            ctem_categories=["external_exposure"],
            compliance_notes=["Review data retention."],
            docs_url="https://platform.openai.com/docs",
        )
        asset.add_match(self._make_match())
        d = asset.to_dict()
        assert d["provider_id"] == "openai"
        assert d["category"] == "llm"
        assert d["baseline_risk_level"] == "high"
        assert d["total_matches"] == 1
        assert isinstance(d["files_detected"], list)
        assert d["vendor_lock_in"] is True

    def test_repr(self) -> None:
        asset = AIAsset(
            provider_id="anthropic",
            provider_name="Anthropic",
            category=ProviderCategory.LLM,
            baseline_risk_level=RiskLevel.HIGH,
            vendor_lock_in=True,
            data_residency_concern=True,
        )
        r = repr(asset)
        assert "AIAsset" in r
        assert "anthropic" in r


# ---------------------------------------------------------------------------
# ScanMetadata tests
# ---------------------------------------------------------------------------


class TestScanMetadata:
    """Tests for the ScanMetadata dataclass."""

    def test_basic_creation(self) -> None:
        now = datetime.now(tz=timezone.utc)
        meta = ScanMetadata(
            scan_id="test-scan-001",
            scan_path=Path("/tmp/project"),
            started_at=now,
        )
        assert meta.scan_id == "test-scan-001"
        assert meta.completed_at is None
        assert meta.duration_seconds is None

    def test_duration_seconds(self) -> None:
        from datetime import timedelta

        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        meta = ScanMetadata(
            scan_id="test",
            scan_path=Path("/tmp/project"),
            started_at=start,
            completed_at=end,
        )
        assert meta.duration_seconds == 30.0

    def test_string_path_converted(self) -> None:
        meta = ScanMetadata(
            scan_id="test",
            scan_path="/tmp/project",  # type: ignore[arg-type]
            started_at=datetime.now(tz=timezone.utc),
        )
        assert isinstance(meta.scan_path, Path)

    def test_to_dict(self) -> None:
        now = datetime.now(tz=timezone.utc)
        meta = ScanMetadata(
            scan_id="abc-123",
            scan_path=Path("/tmp/project"),
            started_at=now,
            scanner_version="0.1.0",
            total_files_scanned=42,
            total_files_skipped=3,
        )
        d = meta.to_dict()
        assert d["scan_id"] == "abc-123"
        assert d["total_files_scanned"] == 42
        assert d["total_files_skipped"] == 3
        assert d["completed_at"] is None
        assert d["duration_seconds"] is None

    def test_repr(self) -> None:
        meta = ScanMetadata(
            scan_id="abc",
            scan_path=Path("/tmp"),
            started_at=datetime.now(tz=timezone.utc),
        )
        assert "ScanMetadata" in repr(meta)


# ---------------------------------------------------------------------------
# ScanSummary tests
# ---------------------------------------------------------------------------


class TestScanSummary:
    """Tests for the ScanSummary dataclass."""

    def test_default_creation(self) -> None:
        summary = ScanSummary()
        assert summary.total_providers_detected == 0
        assert summary.overall_risk_level == RiskLevel.INFO
        assert summary.overall_risk_score == 0.0

    def test_risk_score_clamped(self) -> None:
        summary = ScanSummary(overall_risk_score=15.0)
        assert summary.overall_risk_score == 10.0

    def test_risk_score_negative_clamped(self) -> None:
        summary = ScanSummary(overall_risk_score=-5.0)
        assert summary.overall_risk_score == 0.0

    def test_to_dict(self) -> None:
        summary = ScanSummary(
            total_providers_detected=3,
            total_findings=5,
            critical_findings=1,
            high_findings=2,
            overall_risk_level=RiskLevel.CRITICAL,
            overall_risk_score=8.5,
        )
        d = summary.to_dict()
        assert d["total_providers_detected"] == 3
        assert d["findings_by_severity"]["critical"] == 1
        assert d["overall_risk_level"] == "critical"
        assert d["overall_risk_score"] == 8.5

    def test_repr(self) -> None:
        summary = ScanSummary(total_providers_detected=2, overall_risk_score=5.0)
        assert "ScanSummary" in repr(summary)


# ---------------------------------------------------------------------------
# ScanResult tests
# ---------------------------------------------------------------------------


class TestScanResult:
    """Tests for the ScanResult dataclass."""

    def _make_asset(self, provider_id: str, risk: RiskLevel = RiskLevel.HIGH) -> AIAsset:
        return AIAsset(
            provider_id=provider_id,
            provider_name=provider_id.capitalize(),
            category=ProviderCategory.LLM,
            baseline_risk_level=risk,
            vendor_lock_in=True,
            data_residency_concern=True,
        )

    def _make_finding(
        self,
        finding_id: str,
        provider_id: str,
        risk: RiskLevel = RiskLevel.HIGH,
        score: float = 7.0,
    ) -> RiskFinding:
        return RiskFinding(
            finding_id=finding_id,
            title=f"Finding {finding_id}",
            description="Test finding.",
            risk_level=risk,
            risk_score=score,
            provider_id=provider_id,
            provider_name=provider_id.capitalize(),
        )

    def _make_metadata(self) -> ScanMetadata:
        return ScanMetadata(
            scan_id="test-001",
            scan_path=Path("/tmp/test"),
            started_at=datetime.now(tz=timezone.utc),
        )

    def test_add_asset(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        asset = self._make_asset("openai")
        result.add_asset(asset)
        assert "openai" in result.assets

    def test_add_asset_merges_existing(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        asset1 = self._make_asset("openai")
        asset1.add_match(
            DetectionMatch(
                file_path=Path("a.py"),
                line_number=1,
                line_content="import openai",
                matched_pattern="import openai",
                detection_type=DetectionType.IMPORT,
                provider_id="openai",
            )
        )
        result.add_asset(asset1)

        asset2 = self._make_asset("openai")
        asset2.add_match(
            DetectionMatch(
                file_path=Path("b.py"),
                line_number=2,
                line_content="import openai",
                matched_pattern="import openai",
                detection_type=DetectionType.IMPORT,
                provider_id="openai",
            )
        )
        result.add_asset(asset2)

        assert len(result.assets) == 1
        assert result.assets["openai"].total_matches == 2

    def test_add_finding(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        finding = self._make_finding("f1", "openai")
        result.add_finding(finding)
        assert len(result.findings) == 1

    def test_get_asset_found(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        asset = self._make_asset("openai")
        result.add_asset(asset)
        assert result.get_asset("openai") is asset

    def test_get_asset_not_found(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        assert result.get_asset("nonexistent") is None

    def test_get_findings_by_risk_level(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        result.add_finding(self._make_finding("f1", "openai", RiskLevel.HIGH, 7.0))
        result.add_finding(self._make_finding("f2", "anthropic", RiskLevel.CRITICAL, 9.0))
        result.add_finding(self._make_finding("f3", "cohere", RiskLevel.HIGH, 7.5))

        high_findings = result.get_findings_by_risk_level(RiskLevel.HIGH)
        assert len(high_findings) == 2

        critical_findings = result.get_findings_by_risk_level(RiskLevel.CRITICAL)
        assert len(critical_findings) == 1

    def test_sorted_findings_order(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        result.add_finding(self._make_finding("f1", "openai", RiskLevel.HIGH, 5.0))
        result.add_finding(self._make_finding("f2", "anthropic", RiskLevel.CRITICAL, 9.5))
        result.add_finding(self._make_finding("f3", "cohere", RiskLevel.MEDIUM, 3.0))

        sorted_f = result.sorted_findings
        assert sorted_f[0].risk_score == 9.5
        assert sorted_f[1].risk_score == 5.0
        assert sorted_f[2].risk_score == 3.0

    def test_all_affected_files(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        asset1 = self._make_asset("openai")
        asset1.files_detected = {Path("a.py"), Path("b.py")}
        asset2 = self._make_asset("anthropic")
        asset2.files_detected = {Path("b.py"), Path("c.py")}
        result.add_asset(asset1)
        result.add_asset(asset2)
        all_files = result.all_affected_files
        assert len(all_files) == 3
        assert Path("a.py") in all_files
        assert Path("b.py") in all_files
        assert Path("c.py") in all_files

    def test_compute_summary(self) -> None:
        result = ScanResult(metadata=self._make_metadata())

        asset = self._make_asset("openai")
        asset.add_match(
            DetectionMatch(
                file_path=Path("app.py"),
                line_number=1,
                line_content="import openai",
                matched_pattern="import openai",
                detection_type=DetectionType.IMPORT,
                provider_id="openai",
            )
        )
        result.add_asset(asset)

        result.add_finding(self._make_finding("f1", "openai", RiskLevel.HIGH, 7.0))
        result.add_finding(self._make_finding("f2", "openai", RiskLevel.CRITICAL, 9.0))

        summary = result.compute_summary()
        assert summary.total_providers_detected == 1
        assert summary.total_findings == 2
        assert summary.total_detection_matches == 1
        assert summary.critical_findings == 1
        assert summary.high_findings == 1
        assert summary.overall_risk_level == RiskLevel.CRITICAL

    def test_compute_summary_no_findings(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        summary = result.compute_summary()
        assert summary.overall_risk_level == RiskLevel.INFO
        assert summary.overall_risk_score == 0.0

    def test_to_dict_structure(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        result.add_asset(self._make_asset("openai"))
        result.add_finding(self._make_finding("f1", "openai"))
        result.compute_summary()

        d = result.to_dict()
        assert "schema_version" in d
        assert "metadata" in d
        assert "summary" in d
        assert "assets" in d
        assert "findings" in d
        assert d["schema_version"] == "1.0"

    def test_to_json_valid(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        result.add_asset(self._make_asset("openai"))
        result.compute_summary()

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"
        assert "openai" in parsed["assets"]

    def test_repr(self) -> None:
        result = ScanResult(metadata=self._make_metadata())
        assert "ScanResult" in repr(result)


# ---------------------------------------------------------------------------
# make_scan_result factory tests
# ---------------------------------------------------------------------------


class TestMakeScanResult:
    """Tests for the make_scan_result factory function."""

    def test_creates_scan_result(self) -> None:
        result = make_scan_result("/tmp/project")
        assert isinstance(result, ScanResult)
        assert isinstance(result.metadata, ScanMetadata)

    def test_auto_generates_scan_id(self) -> None:
        result1 = make_scan_result("/tmp/project")
        result2 = make_scan_result("/tmp/project")
        assert result1.metadata.scan_id != result2.metadata.scan_id

    def test_custom_scan_id(self) -> None:
        result = make_scan_result("/tmp/project", scan_id="my-custom-id")
        assert result.metadata.scan_id == "my-custom-id"

    def test_scan_path_set(self) -> None:
        result = make_scan_result("/tmp/myproject")
        assert result.metadata.scan_path == Path("/tmp/myproject")

    def test_scanner_version_set(self) -> None:
        result = make_scan_result("/tmp/project")
        assert result.metadata.scanner_version == "0.1.0"

    def test_started_at_is_utc(self) -> None:
        result = make_scan_result("/tmp/project")
        assert result.metadata.started_at.tzinfo is not None

    def test_path_object_accepted(self) -> None:
        result = make_scan_result(Path("/tmp/project"))
        assert isinstance(result.metadata.scan_path, Path)


# ---------------------------------------------------------------------------
# Enum completeness tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Sanity tests to ensure all enums have expected members."""

    def test_detection_type_members(self) -> None:
        expected = {"import", "env_var", "url", "model_name", "sdk_call", "credential", "unknown"}
        actual = {dt.value for dt in DetectionType}
        assert expected == actual

    def test_provider_category_members(self) -> None:
        expected = {"llm", "image_gen", "speech", "embedding", "mlops", "cloud_ai", "unknown"}
        actual = {pc.value for pc in ProviderCategory}
        assert expected == actual

    def test_nist_function_members(self) -> None:
        expected = {
            "GOVERN-1", "GOVERN-2",
            "MAP-1", "MAP-3", "MAP-4", "MAP-5",
            "MEASURE-1", "MEASURE-2",
            "MANAGE-1", "MANAGE-2", "MANAGE-3",
        }
        actual = {f.value for f in NISTFunction}
        assert expected == actual

    def test_ctem_category_members(self) -> None:
        expected = {
            "external_exposure",
            "data_exfiltration",
            "third_party_dependency",
            "cloud_dependency",
            "supply_chain",
            "identity_access",
            "open_source_risk",
            "biometric_data",
        }
        actual = {c.value for c in CTEMCategory}
        assert expected == actual
