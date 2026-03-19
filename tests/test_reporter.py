"""Unit tests for the ai_risk_scanner.reporter module.

Tests cover JSON generation, Markdown generation, console rendering,
ReportConfig defaults, and the Reporter facade class across various
ScanResult states (empty, single provider, multi-provider, with credentials).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from ai_risk_scanner.inventory import (
    AIAsset,
    ComplianceGap,
    DetectionMatch,
    DetectionType,
    ProviderCategory,
    RiskFinding,
    RiskLevel,
    ScanMetadata,
    ScanResult,
    ScanSummary,
    make_scan_result,
)
from ai_risk_scanner.reporter import (
    ConsoleReporter,
    JSONReporter,
    MarkdownReporter,
    ReportConfig,
    Reporter,
    _bool_icon,
    _format_score,
    _nist_functions_summary,
    _risk_level_badge,
    _severity_bar,
    _truncate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata(scan_id: str = "test-001") -> ScanMetadata:
    """Create a minimal ScanMetadata for test use."""
    started = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2024, 6, 1, 10, 0, 5, tzinfo=timezone.utc)
    return ScanMetadata(
        scan_id=scan_id,
        scan_path=Path("/tmp/test_project"),
        started_at=started,
        completed_at=completed,
        scanner_version="0.1.0",
        total_files_scanned=10,
        total_files_skipped=2,
    )


def _make_asset(
    provider_id: str = "openai",
    provider_name: str = "OpenAI",
    risk_level: RiskLevel = RiskLevel.HIGH,
    vendor_lock_in: bool = True,
    data_residency: bool = True,
    category: ProviderCategory = ProviderCategory.LLM,
    add_match: bool = True,
    add_credential: bool = False,
) -> AIAsset:
    """Create a populated AIAsset for test use."""
    asset = AIAsset(
        provider_id=provider_id,
        provider_name=provider_name,
        category=category,
        baseline_risk_level=risk_level,
        vendor_lock_in=vendor_lock_in,
        data_residency_concern=data_residency,
        nist_rmf_functions=["GOVERN-1", "MAP-1"],
        ctem_categories=["external_exposure", "data_exfiltration"],
        compliance_notes=["Review data retention policy."],
        docs_url=f"https://docs.{provider_id}.com",
    )
    if add_match:
        asset.add_match(
            DetectionMatch(
                file_path=Path(f"src/{provider_id}_client.py"),
                line_number=1,
                line_content=f"import {provider_id}",
                matched_pattern=f"import {provider_id}",
                detection_type=DetectionType.IMPORT,
                provider_id=provider_id,
            )
        )
    if add_credential:
        asset.add_match(
            DetectionMatch(
                file_path=Path(".env"),
                line_number=5,
                line_content=f"{provider_id.upper()}_API_KEY=sk-abc123xyz456def789",
                matched_pattern=r"sk-[A-Za-z0-9]{20,}",
                detection_type=DetectionType.CREDENTIAL,
                provider_id=provider_id,
            )
        )
    return asset


def _make_finding(
    finding_id: str = "finding_openai_llm",
    provider_id: str = "openai",
    provider_name: str = "OpenAI",
    risk_level: RiskLevel = RiskLevel.HIGH,
    risk_score: float = 7.5,
    has_credentials: bool = False,
    add_gap: bool = True,
) -> RiskFinding:
    """Create a populated RiskFinding for test use."""
    gaps: list[ComplianceGap] = []
    if add_gap:
        gaps.append(
            ComplianceGap(
                gap_id=f"gap_{provider_id}_data_residency",
                title=f"{provider_name} Data Residency Risk",
                description="Data may be transferred outside jurisdiction.",
                risk_level=RiskLevel.HIGH,
                nist_functions=["MAP-5", "MEASURE-2"],
                ctem_categories=["data_exfiltration"],
                remediation="Review data processing agreement.",
                provider_id=provider_id,
            )
        )
    matches = [
        DetectionMatch(
            file_path=Path(f"src/{provider_id}.py"),
            line_number=1,
            line_content=f"import {provider_id}",
            matched_pattern=f"import {provider_id}",
            detection_type=DetectionType.IMPORT,
            provider_id=provider_id,
        )
    ]
    return RiskFinding(
        finding_id=finding_id,
        title=f"{provider_name} Integration Detected",
        description=f"{provider_name} SDK detected in codebase.",
        risk_level=risk_level,
        risk_score=risk_score,
        provider_id=provider_id,
        provider_name=provider_name,
        nist_functions=["GOVERN-1", "MAP-1", "MAP-5"],
        ctem_categories=["external_exposure", "data_exfiltration"],
        detection_matches=matches,
        compliance_gaps=gaps,
        vendor_lock_in=True,
        data_residency_concern=True,
        has_hardcoded_credentials=has_credentials,
    )


@pytest.fixture
def empty_result() -> ScanResult:
    """Return a ScanResult with no detections."""
    result = ScanResult(metadata=_make_metadata("empty-001"))
    result.compute_summary()
    return result


@pytest.fixture
def single_provider_result() -> ScanResult:
    """Return a ScanResult with one detected provider and finding."""
    result = ScanResult(metadata=_make_metadata("single-001"))
    asset = _make_asset()
    result.add_asset(asset)
    finding = _make_finding()
    result.add_finding(finding)
    result.compute_summary()
    return result


@pytest.fixture
def multi_provider_result() -> ScanResult:
    """Return a ScanResult with multiple providers and findings at various severity levels."""
    result = ScanResult(metadata=_make_metadata("multi-001"))

    providers = [
        ("openai", "OpenAI", RiskLevel.HIGH, 7.5, False),
        ("anthropic", "Anthropic", RiskLevel.HIGH, 8.0, False),
        ("aws_bedrock", "AWS Bedrock", RiskLevel.MEDIUM, 5.5, False),
        ("huggingface", "Hugging Face", RiskLevel.MEDIUM, 4.5, False),
    ]

    for pid, pname, risk, score, cred in providers:
        asset = _make_asset(
            provider_id=pid,
            provider_name=pname,
            risk_level=risk,
            add_credential=cred,
        )
        result.add_asset(asset)
        finding = _make_finding(
            finding_id=f"finding_{pid}_llm",
            provider_id=pid,
            provider_name=pname,
            risk_level=risk,
            risk_score=score,
            has_credentials=cred,
        )
        result.add_finding(finding)

    result.compute_summary()
    return result


@pytest.fixture
def critical_credential_result() -> ScanResult:
    """Return a ScanResult with hardcoded credential exposure."""
    result = ScanResult(metadata=_make_metadata("cred-001"))

    asset = _make_asset(
        provider_id="openai",
        provider_name="OpenAI",
        risk_level=RiskLevel.CRITICAL,
        add_credential=True,
    )
    result.add_asset(asset)

    finding = _make_finding(
        finding_id="finding_openai_llm",
        risk_level=RiskLevel.CRITICAL,
        risk_score=9.5,
        has_credentials=True,
    )
    result.add_finding(finding)
    result.compute_summary()
    return result


@pytest.fixture
def string_console() -> Console:
    """Return a rich Console that writes to a StringIO buffer."""
    return Console(file=StringIO(), width=120, highlight=False, no_color=True)


# ---------------------------------------------------------------------------
# ReportConfig tests
# ---------------------------------------------------------------------------


class TestReportConfig:
    """Tests for ReportConfig defaults and attributes."""

    def test_defaults(self) -> None:
        cfg = ReportConfig()
        assert cfg.show_evidence is False
        assert cfg.max_evidence_lines == 3
        assert cfg.show_compliance_gaps is True
        assert cfg.show_nist_mapping is True
        assert cfg.show_ctem_mapping is True
        assert cfg.show_affected_files is True
        assert cfg.truncate_line_length == 120
        assert cfg.indent == 2
        assert cfg.console_width is None
        assert cfg.verbose is False

    def test_custom_values(self) -> None:
        cfg = ReportConfig(
            show_evidence=True,
            max_evidence_lines=10,
            verbose=True,
            indent=4,
        )
        assert cfg.show_evidence is True
        assert cfg.max_evidence_lines == 10
        assert cfg.verbose is True
        assert cfg.indent == 4


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_risk_level_badge_returns_text(self) -> None:
        from rich.text import Text
        badge = _risk_level_badge(RiskLevel.CRITICAL)
        assert isinstance(badge, Text)
        assert "CRITICAL" in badge.plain

    def test_risk_level_badge_all_levels(self) -> None:
        for level in RiskLevel:
            badge = _risk_level_badge(level)
            assert level.value.upper() in badge.plain

    def test_truncate_short_string(self) -> None:
        assert _truncate("hello", 20) == "hello"

    def test_truncate_long_string(self) -> None:
        result = _truncate("a" * 50, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_exact_length(self) -> None:
        s = "a" * 20
        assert _truncate(s, 20) == s

    def test_truncate_strips_whitespace(self) -> None:
        assert _truncate("  hello  ", 20) == "hello"

    def test_bool_icon_true(self) -> None:
        assert _bool_icon(True) == "\u2713"

    def test_bool_icon_false(self) -> None:
        assert _bool_icon(False) == "\u2717"

    def test_format_score(self) -> None:
        assert _format_score(7.5) == "7.5"
        assert _format_score(10.0) == "10.0"
        assert _format_score(0.0) == "0.0"

    def test_nist_functions_summary_empty(self) -> None:
        assert _nist_functions_summary([]) == "-"

    def test_nist_functions_summary_deduplicates(self) -> None:
        result = _nist_functions_summary(["MAP-1", "MAP-1", "GOVERN-1"])
        assert "GOVERN-1" in result
        assert "MAP-1" in result
        # Check no duplicates
        parts = result.split(", ")
        assert len(parts) == len(set(parts))

    def test_severity_bar_max(self) -> None:
        bar = _severity_bar(10.0)
        assert "10.0/10" in bar
        assert "=========" in bar

    def test_severity_bar_zero(self) -> None:
        bar = _severity_bar(0.0)
        assert "0.0/10" in bar

    def test_severity_bar_mid(self) -> None:
        bar = _severity_bar(5.0)
        assert "5.0/10" in bar


# ---------------------------------------------------------------------------
# JSONReporter tests
# ---------------------------------------------------------------------------


class TestJSONReporter:
    """Tests for JSONReporter output correctness."""

    def test_generate_returns_valid_json(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        output = reporter.generate(single_provider_result)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_contains_schema_version(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        assert parsed["schema_version"] == "1.0"

    def test_json_contains_metadata(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        assert "metadata" in parsed
        assert parsed["metadata"]["scan_id"] == "single-001"

    def test_json_contains_summary(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        assert "summary" in parsed
        assert "total_providers_detected" in parsed["summary"]
        assert parsed["summary"]["total_providers_detected"] == 1

    def test_json_contains_assets(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        assert "assets" in parsed
        assert "openai" in parsed["assets"]

    def test_json_contains_findings(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        assert "findings" in parsed
        assert len(parsed["findings"]) == 1

    def test_json_empty_result(self, empty_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(empty_result))
        assert parsed["summary"]["total_providers_detected"] == 0
        assert parsed["assets"] == {}
        assert parsed["findings"] == []

    def test_json_multi_provider(self, multi_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(multi_provider_result))
        assert len(parsed["assets"]) == 4
        assert len(parsed["findings"]) == 4

    def test_json_custom_indent(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter(config=ReportConfig(indent=4))
        output = reporter.generate(single_provider_result)
        # 4-space indented JSON will have "    " at the start of nested keys
        assert "    " in output

    def test_json_findings_sorted_by_score(self, multi_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(multi_provider_result))
        scores = [f["risk_score"] for f in parsed["findings"]]
        assert scores == sorted(scores, reverse=True)

    def test_write_to_file(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = JSONReporter()
        output_file = tmp_path / "report.json"
        reporter.write(single_provider_result, output_file)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert parsed["schema_version"] == "1.0"

    def test_write_creates_parent_dirs(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = JSONReporter()
        output_file = tmp_path / "nested" / "dir" / "report.json"
        reporter.write(single_provider_result, output_file)
        assert output_file.exists()

    def test_write_stream(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        buf = StringIO()
        reporter.write_stream(single_provider_result, buf)
        content = buf.getvalue()
        parsed = json.loads(content)
        assert "schema_version" in parsed

    def test_json_credential_finding(self, critical_credential_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(critical_credential_result))
        findings = parsed["findings"]
        assert len(findings) >= 1
        top_finding = findings[0]
        assert top_finding["risk_level"] in ("critical", "high")

    def test_json_compliance_gaps_present(self, single_provider_result: ScanResult) -> None:
        reporter = JSONReporter()
        parsed = json.loads(reporter.generate(single_provider_result))
        findings = parsed["findings"]
        assert len(findings) > 0
        assert "compliance_gaps" in findings[0]


# ---------------------------------------------------------------------------
# MarkdownReporter tests
# ---------------------------------------------------------------------------


class TestMarkdownReporter:
    """Tests for MarkdownReporter output correctness."""

    def test_generate_returns_string(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_contains_title(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "# AI Risk Inventory Report" in output

    def test_contains_scan_id(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "single-001" in output

    def test_contains_executive_summary(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Executive Summary" in output

    def test_contains_providers_table(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Detected AI Providers" in output
        assert "OpenAI" in output

    def test_contains_findings_section(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Risk Findings" in output
        assert "OpenAI Integration Detected" in output

    def test_contains_compliance_gaps(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_compliance_gaps=True))
        output = reporter.generate(single_provider_result)
        assert "## Compliance Gaps" in output

    def test_compliance_gaps_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_compliance_gaps=False))
        output = reporter.generate(single_provider_result)
        assert "## Compliance Gaps" not in output

    def test_contains_nist_mapping(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_nist_mapping=True))
        output = reporter.generate(single_provider_result)
        assert "## NIST AI RMF Mapping" in output

    def test_nist_mapping_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_nist_mapping=False))
        output = reporter.generate(single_provider_result)
        assert "## NIST AI RMF Mapping" not in output

    def test_contains_ctem_mapping(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_ctem_mapping=True))
        output = reporter.generate(single_provider_result)
        assert "## CTEM Exposure Mapping" in output

    def test_ctem_mapping_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_ctem_mapping=False))
        output = reporter.generate(single_provider_result)
        assert "## CTEM Exposure Mapping" not in output

    def test_contains_affected_files(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_affected_files=True))
        output = reporter.generate(single_provider_result)
        assert "## Affected Files" in output

    def test_affected_files_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_affected_files=False))
        output = reporter.generate(single_provider_result)
        assert "## Affected Files" not in output

    def test_contains_evidence_appendix(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Detection Evidence Appendix" in output

    def test_contains_toc(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Table of Contents" in output

    def test_contains_footer(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "AI Risk Inventory Scanner" in output

    def test_empty_result_no_providers(self, empty_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(empty_result)
        assert "No AI providers detected" in output

    def test_empty_result_no_findings(self, empty_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(empty_result)
        assert "No risk findings generated" in output

    def test_multi_provider_all_present(self, multi_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(multi_provider_result)
        assert "OpenAI" in output
        assert "Anthropic" in output
        assert "AWS Bedrock" in output
        assert "Hugging Face" in output

    def test_critical_credential_warning(self, critical_credential_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(critical_credential_result)
        # Should mention credentials
        assert "CRITICAL" in output

    def test_risk_score_in_findings(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        # Score should appear
        assert "7.5" in output

    def test_write_to_file(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        output_file = tmp_path / "report.md"
        reporter.write(single_provider_result, output_file)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# AI Risk Inventory Report" in content

    def test_write_creates_parent_dirs(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        output_file = tmp_path / "deep" / "nested" / "report.md"
        reporter.write(single_provider_result, output_file)
        assert output_file.exists()

    def test_write_stream(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        buf = StringIO()
        reporter.write_stream(single_provider_result, buf)
        content = buf.getvalue()
        assert "# AI Risk Inventory Report" in content

    def test_nist_function_reference_table(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_nist_mapping=True))
        output = reporter.generate(single_provider_result)
        assert "GOVERN-1" in output
        assert "MAP-1" in output

    def test_ctem_category_reference_table(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_ctem_mapping=True))
        output = reporter.generate(single_provider_result)
        assert "external_exposure" in output
        assert "data_exfiltration" in output

    def test_vendor_lock_in_in_providers_table(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        # The providers table should have lock-in column
        assert "Lock-in" in output

    def test_evidence_limited_to_max_lines(self) -> None:
        result = ScanResult(metadata=_make_metadata("evidence-test"))
        asset = _make_asset()
        # Add extra matches
        for i in range(2, 15):
            asset.add_match(
                DetectionMatch(
                    file_path=Path(f"file_{i}.py"),
                    line_number=i,
                    line_content=f"import openai  # {i}",
                    matched_pattern="import openai",
                    detection_type=DetectionType.IMPORT,
                    provider_id="openai",
                )
            )
        result.add_asset(asset)
        finding = _make_finding()
        # Add extra matches to finding
        for i in range(2, 15):
            finding.detection_matches.append(
                DetectionMatch(
                    file_path=Path(f"file_{i}.py"),
                    line_number=i,
                    line_content=f"import openai  # {i}",
                    matched_pattern="import openai",
                    detection_type=DetectionType.IMPORT,
                    provider_id="openai",
                )
            )
        result.add_finding(finding)
        result.compute_summary()

        cfg = ReportConfig(max_evidence_lines=3)
        reporter = MarkdownReporter(config=cfg)
        output = reporter.generate(result)
        # Should mention 'more matches'
        assert "more match" in output.lower()

    def test_remediation_guidance_in_gaps(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter(config=ReportConfig(show_compliance_gaps=True))
        output = reporter.generate(single_provider_result)
        # Remediation guidance section should appear
        assert "Remediation Guidance" in output

    def test_scan_metadata_table(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        assert "## Scan Metadata" in output
        assert "single-001" in output
        assert "0.1.0" in output

    def test_duration_in_metadata(self, single_provider_result: ScanResult) -> None:
        reporter = MarkdownReporter()
        output = reporter.generate(single_provider_result)
        # Duration of 5 seconds should appear
        assert "5.00s" in output


# ---------------------------------------------------------------------------
# ConsoleReporter tests
# ---------------------------------------------------------------------------


class TestConsoleReporter:
    """Tests for ConsoleReporter terminal output."""

    def _capture(self, reporter: ConsoleReporter) -> str:
        """Return the string captured by the reporter's console StringIO buffer."""
        assert isinstance(reporter.console.file, StringIO)
        return reporter.console.file.getvalue()

    def _make_reporter(self, config: ReportConfig | None = None) -> ConsoleReporter:
        console = Console(file=StringIO(), width=120, highlight=False, no_color=True)
        return ConsoleReporter(config=config, console=console)

    def test_print_does_not_raise(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)  # Should not raise

    def test_print_contains_provider_name(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "OpenAI" in output

    def test_print_contains_scan_id(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "single-001" in output

    def test_print_contains_risk_level(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "HIGH" in output

    def test_print_empty_result(self, empty_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(empty_result)
        output = self._capture(reporter)
        assert "No AI providers detected" in output

    def test_print_multi_provider(self, multi_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(multi_provider_result)
        output = self._capture(reporter)
        assert "OpenAI" in output
        assert "Anthropic" in output

    def test_print_critical_finding(self, critical_credential_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(critical_credential_result)
        output = self._capture(reporter)
        assert "CRITICAL" in output

    def test_compliance_gaps_shown_by_default(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(config=ReportConfig(show_compliance_gaps=True))
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "Compliance Gaps" in output

    def test_compliance_gaps_hidden_when_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(config=ReportConfig(show_compliance_gaps=False))
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "Compliance Gaps" not in output

    def test_nist_mapping_shown_by_default(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(config=ReportConfig(show_nist_mapping=True))
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "NIST" in output

    def test_nist_mapping_hidden_when_disabled(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(config=ReportConfig(show_nist_mapping=False))
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "NIST AI RMF" not in output

    def test_evidence_shown_when_verbose(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(
            config=ReportConfig(show_evidence=True, verbose=True)
        )
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "Detection Evidence Detail" in output

    def test_evidence_hidden_when_not_verbose(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter(
            config=ReportConfig(show_evidence=True, verbose=False)
        )
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "Detection Evidence Detail" not in output

    def test_summary_score_shown(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        # Score should appear somewhere in the summary
        assert "/10" in output

    def test_files_scanned_count_in_header(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print(single_provider_result)
        output = self._capture(reporter)
        assert "10" in output  # total_files_scanned

    def test_custom_console_width(self, single_provider_result: ScanResult) -> None:
        console = Console(file=StringIO(), width=80, highlight=False, no_color=True)
        reporter = ConsoleReporter(
            config=ReportConfig(console_width=80), console=console
        )
        reporter.print(single_provider_result)  # Should not raise


# ---------------------------------------------------------------------------
# Reporter facade tests
# ---------------------------------------------------------------------------


class TestReporter:
    """Tests for the Reporter facade class."""

    def _make_reporter(self, config: ReportConfig | None = None) -> Reporter:
        console = Console(file=StringIO(), width=120, highlight=False, no_color=True)
        return Reporter(config=config, console=console)

    def test_init_creates_sub_reporters(self) -> None:
        reporter = self._make_reporter()
        assert reporter.console_reporter is not None
        assert reporter.json_reporter is not None
        assert reporter.markdown_reporter is not None

    def test_generate_json_returns_valid_json(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        output = reporter.generate_json(single_provider_result)
        parsed = json.loads(output)
        assert "schema_version" in parsed

    def test_generate_markdown_returns_string(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        output = reporter.generate_markdown(single_provider_result)
        assert "# AI Risk Inventory Report" in output

    def test_print_console_does_not_raise(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        reporter.print_console(single_provider_result)  # Should not raise

    def test_write_json(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = self._make_reporter()
        output_file = tmp_path / "report.json"
        reporter.write_json(single_provider_result, output_file)
        assert output_file.exists()
        assert json.loads(output_file.read_text(encoding="utf-8"))["schema_version"] == "1.0"

    def test_write_markdown(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = self._make_reporter()
        output_file = tmp_path / "report.md"
        reporter.write_markdown(single_provider_result, output_file)
        assert output_file.exists()
        assert "# AI Risk Inventory Report" in output_file.read_text(encoding="utf-8")

    def test_report_console_format(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        result = reporter.report(single_provider_result, format="console")
        assert result is None

    def test_report_json_format(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        result = reporter.report(single_provider_result, format="json")
        assert result is not None
        parsed = json.loads(result)
        assert "schema_version" in parsed

    def test_report_markdown_format(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        result = reporter.report(single_provider_result, format="markdown")
        assert result is not None
        assert "# AI Risk Inventory Report" in result

    def test_report_md_alias(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        result = reporter.report(single_provider_result, format="md")
        assert result is not None
        assert "# AI Risk Inventory Report" in result

    def test_report_invalid_format_raises(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        with pytest.raises(ValueError, match="Unsupported report format"):
            reporter.report(single_provider_result, format="xml")

    def test_report_json_writes_file(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = self._make_reporter()
        output_file = tmp_path / "out.json"
        reporter.report(single_provider_result, format="json", output_path=output_file)
        assert output_file.exists()
        parsed = json.loads(output_file.read_text(encoding="utf-8"))
        assert "schema_version" in parsed

    def test_report_markdown_writes_file(self, single_provider_result: ScanResult, tmp_path: Path) -> None:
        reporter = self._make_reporter()
        output_file = tmp_path / "out.md"
        reporter.report(single_provider_result, format="markdown", output_path=output_file)
        assert output_file.exists()
        assert "# AI Risk Inventory Report" in output_file.read_text(encoding="utf-8")

    def test_console_property(self) -> None:
        reporter = self._make_reporter()
        assert reporter.console is reporter.console_reporter.console

    def test_repr(self) -> None:
        reporter = self._make_reporter()
        assert "Reporter" in repr(reporter)

    def test_shared_config_used_across_sub_reporters(self) -> None:
        cfg = ReportConfig(show_evidence=True, verbose=True, indent=4)
        reporter = Reporter(config=cfg)
        assert reporter.console_reporter.config is cfg
        assert reporter.json_reporter.config is cfg
        assert reporter.markdown_reporter.config is cfg

    def test_case_insensitive_format(self, single_provider_result: ScanResult) -> None:
        reporter = self._make_reporter()
        result_upper = reporter.report(single_provider_result, format="JSON")
        assert result_upper is not None
        parsed = json.loads(result_upper)
        assert "schema_version" in parsed


# ---------------------------------------------------------------------------
# Integration-style tests using make_scan_result
# ---------------------------------------------------------------------------


class TestReporterIntegration:
    """Integration tests using make_scan_result with the full pipeline."""

    def test_full_pipeline_json_roundtrip(self, tmp_path: Path) -> None:
        """Verify that a ScanResult can be serialized to JSON and the key
        structure is preserved."""
        result = make_scan_result(tmp_path)
        asset = _make_asset()
        result.add_asset(asset)
        result.add_finding(_make_finding())
        result.compute_summary()
        result.metadata.completed_at = datetime.now(tz=timezone.utc)

        reporter = JSONReporter()
        json_str = reporter.generate(result)
        parsed = json.loads(json_str)

        assert parsed["schema_version"] == "1.0"
        assert "openai" in parsed["assets"]
        assert len(parsed["findings"]) == 1
        assert parsed["findings"][0]["provider_id"] == "openai"

    def test_full_pipeline_markdown_includes_all_sections(self, tmp_path: Path) -> None:
        """Verify a complete Markdown report includes all major sections."""
        result = make_scan_result(tmp_path)
        asset = _make_asset()
        result.add_asset(asset)
        result.add_finding(_make_finding())
        result.compute_summary()

        reporter = MarkdownReporter()
        output = reporter.generate(result)

        required_sections = [
            "# AI Risk Inventory Report",
            "## Table of Contents",
            "## Scan Metadata",
            "## Executive Summary",
            "## Detected AI Providers",
            "## Risk Findings",
            "## Compliance Gaps",
            "## NIST AI RMF Mapping",
            "## CTEM Exposure Mapping",
            "## Detection Evidence Appendix",
        ]
        for section in required_sections:
            assert section in output, f"Missing section: {section!r}"

    def test_reporter_handles_result_with_no_findings(self, tmp_path: Path) -> None:
        """Verify the reporter handles a result with assets but no findings gracefully."""
        result = make_scan_result(tmp_path)
        asset = _make_asset()
        result.add_asset(asset)
        result.compute_summary()

        reporter = Reporter()
        json_str = reporter.generate_json(result)
        parsed = json.loads(json_str)
        assert parsed["findings"] == []

        md_str = reporter.generate_markdown(result)
        assert "No risk findings generated" in md_str

    def test_reporter_with_hardcoded_cred_finding(self) -> None:
        """Verify the credential finding renders correctly in all formats."""
        result = ScanResult(metadata=_make_metadata("cred-integration"))
        cred_asset = AIAsset(
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            category=ProviderCategory.UNKNOWN,
            baseline_risk_level=RiskLevel.CRITICAL,
            vendor_lock_in=False,
            data_residency_concern=False,
        )
        cred_asset.add_match(
            DetectionMatch(
                file_path=Path("config.py"),
                line_number=10,
                line_content='OPENAI_API_KEY = "sk-abc123xyz456def789012"',
                matched_pattern=r"sk-[A-Za-z0-9]{20,}",
                detection_type=DetectionType.CREDENTIAL,
                provider_id="_credentials",
            )
        )
        result.add_asset(cred_asset)

        cred_finding = RiskFinding(
            finding_id="finding_hardcoded_credentials",
            title="Hardcoded AI API Credentials Detected",
            description="Detected 1 hardcoded credential pattern(s).",
            risk_level=RiskLevel.CRITICAL,
            risk_score=9.5,
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            nist_functions=["GOVERN-1", "MANAGE-2"],
            ctem_categories=["identity_access"],
            detection_matches=list(cred_asset.detection_matches),
            compliance_gaps=[
                ComplianceGap(
                    gap_id="gap_hardcoded_credentials",
                    title="Hardcoded Credentials",
                    description="API keys hardcoded in source.",
                    risk_level=RiskLevel.CRITICAL,
                    remediation="Use environment variables.",
                    provider_id="_credentials",
                )
            ],
            has_hardcoded_credentials=True,
        )
        result.add_finding(cred_finding)
        result.compute_summary()

        json_reporter = JSONReporter()
        parsed = json.loads(json_reporter.generate(result))
        assert parsed["summary"]["critical_findings"] == 1

        md_reporter = MarkdownReporter()
        md_output = md_reporter.generate(result)
        assert "Hardcoded" in md_output
        assert "CRITICAL" in md_output
