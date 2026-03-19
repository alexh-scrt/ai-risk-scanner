"""Data models for the AI Risk Inventory Scanner.

This module defines the core data structures representing detected AI assets,
findings, risk assessments, and scan results. These models serve as the shared
schema across all scanner modules including detectors, risk engine, and reporter.

Key classes:
    - DetectionMatch: A single pattern match found in a file.
    - AIAsset: A detected AI provider/service usage within the codebase.
    - RiskFinding: A risk assessment finding mapped to NIST AI RMF and CTEM.
    - ComplianceGap: A specific compliance gap identified for an asset.
    - ScanResult: The aggregated result of a full codebase scan.
    - ScanMetadata: Metadata about the scan execution itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class RiskLevel(str, Enum):
    """Enumeration of risk severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __lt__(self, other: "RiskLevel") -> bool:
        """Enable ordering of risk levels from lowest to highest severity."""
        order = [RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "RiskLevel") -> bool:
        order = [RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) <= order.index(other)

    def __gt__(self, other: "RiskLevel") -> bool:
        order = [RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) > order.index(other)

    def __ge__(self, other: "RiskLevel") -> bool:
        order = [RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) >= order.index(other)

    @property
    def numeric_score(self) -> int:
        """Return a numeric score for the risk level (higher = more severe)."""
        scores = {
            RiskLevel.INFO: 1,
            RiskLevel.LOW: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.HIGH: 4,
            RiskLevel.CRITICAL: 5,
        }
        return scores[self]


class ProviderCategory(str, Enum):
    """Enumeration of AI provider/service categories."""

    LLM = "llm"
    IMAGE_GEN = "image_gen"
    SPEECH = "speech"
    EMBEDDING = "embedding"
    MLOPS = "mlops"
    CLOUD_AI = "cloud_ai"
    UNKNOWN = "unknown"


class DetectionType(str, Enum):
    """Enumeration of detection pattern types."""

    IMPORT = "import"
    ENV_VAR = "env_var"
    URL = "url"
    MODEL_NAME = "model_name"
    SDK_CALL = "sdk_call"
    CREDENTIAL = "credential"
    UNKNOWN = "unknown"


class NISTFunction(str, Enum):
    """NIST AI RMF core function identifiers."""

    GOVERN_1 = "GOVERN-1"
    GOVERN_2 = "GOVERN-2"
    MAP_1 = "MAP-1"
    MAP_3 = "MAP-3"
    MAP_4 = "MAP-4"
    MAP_5 = "MAP-5"
    MEASURE_1 = "MEASURE-1"
    MEASURE_2 = "MEASURE-2"
    MANAGE_1 = "MANAGE-1"
    MANAGE_2 = "MANAGE-2"
    MANAGE_3 = "MANAGE-3"


class CTEMCategory(str, Enum):
    """CTEM (Continuous Threat Exposure Management) exposure categories."""

    EXTERNAL_EXPOSURE = "external_exposure"
    DATA_EXFILTRATION = "data_exfiltration"
    THIRD_PARTY_DEPENDENCY = "third_party_dependency"
    CLOUD_DEPENDENCY = "cloud_dependency"
    SUPPLY_CHAIN = "supply_chain"
    IDENTITY_ACCESS = "identity_access"
    OPEN_SOURCE_RISK = "open_source_risk"
    BIOMETRIC_DATA = "biometric_data"


@dataclass
class DetectionMatch:
    """Represents a single pattern match found within a file.

    Attributes:
        file_path: Path to the file where the match was found.
        line_number: 1-based line number of the match.
        line_content: The actual line content containing the match.
        matched_pattern: The pattern string that produced the match.
        detection_type: The category of detection (import, env_var, url, etc.).
        provider_id: The identifier of the matched AI provider.
        context_lines: Surrounding lines for additional context (optional).
    """

    file_path: Path
    line_number: int
    line_content: str
    matched_pattern: str
    detection_type: DetectionType
    provider_id: str
    context_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if isinstance(self.detection_type, str):
            try:
                self.detection_type = DetectionType(self.detection_type)
            except ValueError:
                self.detection_type = DetectionType.UNKNOWN
        if self.line_number < 1:
            raise ValueError(f"line_number must be >= 1, got {self.line_number}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the detection match to a dictionary."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "line_content": self.line_content.strip(),
            "matched_pattern": self.matched_pattern,
            "detection_type": self.detection_type.value,
            "provider_id": self.provider_id,
            "context_lines": [line.rstrip() for line in self.context_lines],
        }

    def __repr__(self) -> str:
        return (
            f"DetectionMatch(file={self.file_path.name!r}, "
            f"line={self.line_number}, "
            f"type={self.detection_type.value!r}, "
            f"provider={self.provider_id!r})"
        )


@dataclass
class ComplianceGap:
    """Represents a specific compliance gap identified for an AI asset.

    Attributes:
        gap_id: Unique identifier for this gap type.
        title: Short human-readable title of the compliance gap.
        description: Detailed description of the compliance gap.
        risk_level: Severity of this compliance gap.
        nist_functions: NIST AI RMF functions this gap relates to.
        ctem_categories: CTEM categories this gap maps to.
        remediation: Suggested remediation steps.
        provider_id: The AI provider this gap is associated with.
    """

    gap_id: str
    title: str
    description: str
    risk_level: RiskLevel
    nist_functions: list[str] = field(default_factory=list)
    ctem_categories: list[str] = field(default_factory=list)
    remediation: str = ""
    provider_id: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.risk_level, str):
            try:
                self.risk_level = RiskLevel(self.risk_level)
            except ValueError:
                self.risk_level = RiskLevel.INFO

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compliance gap to a dictionary."""
        return {
            "gap_id": self.gap_id,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "nist_functions": self.nist_functions,
            "ctem_categories": self.ctem_categories,
            "remediation": self.remediation,
            "provider_id": self.provider_id,
        }

    def __repr__(self) -> str:
        return (
            f"ComplianceGap(id={self.gap_id!r}, "
            f"risk={self.risk_level.value!r}, "
            f"provider={self.provider_id!r})"
        )


@dataclass
class RiskFinding:
    """A risk assessment finding mapped to NIST AI RMF and CTEM frameworks.

    Attributes:
        finding_id: Unique identifier for this finding.
        title: Short human-readable title.
        description: Detailed description of the risk finding.
        risk_level: Assessed severity of this finding.
        risk_score: Numeric risk score (0.0 - 10.0).
        provider_id: The AI provider this finding relates to.
        provider_name: Human-readable provider name.
        nist_functions: Applicable NIST AI RMF function identifiers.
        ctem_categories: Applicable CTEM exposure category identifiers.
        detection_matches: List of evidence supporting this finding.
        compliance_gaps: Specific compliance gaps identified.
        vendor_lock_in: Whether this finding involves vendor lock-in risk.
        data_residency_concern: Whether this involves data residency risk.
        has_hardcoded_credentials: Whether hardcoded credentials were detected.
    """

    finding_id: str
    title: str
    description: str
    risk_level: RiskLevel
    risk_score: float
    provider_id: str
    provider_name: str
    nist_functions: list[str] = field(default_factory=list)
    ctem_categories: list[str] = field(default_factory=list)
    detection_matches: list[DetectionMatch] = field(default_factory=list)
    compliance_gaps: list[ComplianceGap] = field(default_factory=list)
    vendor_lock_in: bool = False
    data_residency_concern: bool = False
    has_hardcoded_credentials: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.risk_level, str):
            try:
                self.risk_level = RiskLevel(self.risk_level)
            except ValueError:
                self.risk_level = RiskLevel.INFO
        if not (0.0 <= self.risk_score <= 10.0):
            raise ValueError(
                f"risk_score must be between 0.0 and 10.0, got {self.risk_score}"
            )

    @property
    def affected_files(self) -> list[Path]:
        """Return deduplicated list of files where this finding was detected."""
        seen: set[Path] = set()
        result: list[Path] = []
        for match in self.detection_matches:
            if match.file_path not in seen:
                seen.add(match.file_path)
                result.append(match.file_path)
        return result

    @property
    def evidence_count(self) -> int:
        """Return the total number of detection matches for this finding."""
        return len(self.detection_matches)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the risk finding to a dictionary."""
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 2),
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "nist_functions": self.nist_functions,
            "ctem_categories": self.ctem_categories,
            "vendor_lock_in": self.vendor_lock_in,
            "data_residency_concern": self.data_residency_concern,
            "has_hardcoded_credentials": self.has_hardcoded_credentials,
            "evidence_count": self.evidence_count,
            "affected_files": [str(p) for p in self.affected_files],
            "detection_matches": [m.to_dict() for m in self.detection_matches],
            "compliance_gaps": [g.to_dict() for g in self.compliance_gaps],
        }

    def __repr__(self) -> str:
        return (
            f"RiskFinding(id={self.finding_id!r}, "
            f"provider={self.provider_id!r}, "
            f"risk={self.risk_level.value!r}, "
            f"score={self.risk_score:.1f}, "
            f"matches={self.evidence_count})"
        )


@dataclass
class AIAsset:
    """Represents a detected AI provider or service usage within the codebase.

    An AIAsset aggregates all detection matches for a single provider/service
    and stores the provider's metadata for downstream risk analysis.

    Attributes:
        provider_id: Unique identifier matching providers.yaml entry.
        provider_name: Human-readable display name of the provider.
        category: Category of AI service (llm, cloud_ai, mlops, etc.).
        baseline_risk_level: Default risk level from provider database.
        vendor_lock_in: Whether the provider creates vendor lock-in.
        data_residency_concern: Whether data leaves the user's jurisdiction.
        detection_matches: All pattern matches found for this provider.
        files_detected: Set of file paths where this provider was found.
        nist_rmf_functions: Applicable NIST AI RMF function IDs.
        ctem_categories: Applicable CTEM exposure category IDs.
        compliance_notes: Provider-specific compliance concern notes.
        docs_url: URL to provider documentation.
    """

    provider_id: str
    provider_name: str
    category: ProviderCategory
    baseline_risk_level: RiskLevel
    vendor_lock_in: bool
    data_residency_concern: bool
    detection_matches: list[DetectionMatch] = field(default_factory=list)
    files_detected: set[Path] = field(default_factory=set)
    nist_rmf_functions: list[str] = field(default_factory=list)
    ctem_categories: list[str] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)
    docs_url: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.category, str):
            try:
                self.category = ProviderCategory(self.category)
            except ValueError:
                self.category = ProviderCategory.UNKNOWN
        if isinstance(self.baseline_risk_level, str):
            try:
                self.baseline_risk_level = RiskLevel(self.baseline_risk_level)
            except ValueError:
                self.baseline_risk_level = RiskLevel.INFO
        # Ensure files_detected is a set even if provided as list
        if isinstance(self.files_detected, list):
            self.files_detected = set(
                Path(p) if isinstance(p, str) else p for p in self.files_detected
            )

    def add_match(self, match: DetectionMatch) -> None:
        """Add a detection match and register its file path.

        Args:
            match: The DetectionMatch to add to this asset.
        """
        self.detection_matches.append(match)
        self.files_detected.add(match.file_path)

    @property
    def total_matches(self) -> int:
        """Return the total number of detection matches."""
        return len(self.detection_matches)

    @property
    def detection_types_found(self) -> set[DetectionType]:
        """Return the set of detection types found for this asset."""
        return {match.detection_type for match in self.detection_matches}

    @property
    def has_credential_exposure(self) -> bool:
        """Return True if any credential pattern was detected."""
        return any(
            match.detection_type == DetectionType.CREDENTIAL
            for match in self.detection_matches
        )

    @property
    def has_env_var_usage(self) -> bool:
        """Return True if any environment variable reference was detected."""
        return any(
            match.detection_type == DetectionType.ENV_VAR
            for match in self.detection_matches
        )

    @property
    def matches_by_type(self) -> dict[DetectionType, list[DetectionMatch]]:
        """Return detection matches grouped by detection type."""
        result: dict[DetectionType, list[DetectionMatch]] = {}
        for match in self.detection_matches:
            result.setdefault(match.detection_type, []).append(match)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the AI asset to a dictionary."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "category": self.category.value,
            "baseline_risk_level": self.baseline_risk_level.value,
            "vendor_lock_in": self.vendor_lock_in,
            "data_residency_concern": self.data_residency_concern,
            "total_matches": self.total_matches,
            "files_detected": sorted(str(p) for p in self.files_detected),
            "detection_types_found": sorted(dt.value for dt in self.detection_types_found),
            "has_credential_exposure": self.has_credential_exposure,
            "nist_rmf_functions": self.nist_rmf_functions,
            "ctem_categories": self.ctem_categories,
            "compliance_notes": self.compliance_notes,
            "docs_url": self.docs_url,
            "detection_matches": [m.to_dict() for m in self.detection_matches],
        }

    def __repr__(self) -> str:
        return (
            f"AIAsset(id={self.provider_id!r}, "
            f"name={self.provider_name!r}, "
            f"category={self.category.value!r}, "
            f"matches={self.total_matches}, "
            f"files={len(self.files_detected)})"
        )


@dataclass
class ScanMetadata:
    """Metadata about a scan execution.

    Attributes:
        scan_id: Unique identifier for this scan run.
        scan_path: Root path that was scanned.
        started_at: UTC datetime when the scan began.
        completed_at: UTC datetime when the scan completed (None if in progress).
        scanner_version: Version of the scanner that produced this result.
        total_files_scanned: Number of files processed.
        total_files_skipped: Number of files skipped (e.g., binary, too large).
        supported_extensions: File extensions that were eligible for scanning.
        excluded_paths: Paths that were excluded from scanning.
    """

    scan_id: str
    scan_path: Path
    started_at: datetime
    completed_at: datetime | None = None
    scanner_version: str = "0.1.0"
    total_files_scanned: int = 0
    total_files_skipped: int = 0
    supported_extensions: list[str] = field(default_factory=list)
    excluded_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.scan_path, str):
            self.scan_path = Path(self.scan_path)
        if isinstance(self.started_at, str):
            self.started_at = datetime.fromisoformat(self.started_at)
        if isinstance(self.completed_at, str):
            self.completed_at = datetime.fromisoformat(self.completed_at)

    @property
    def duration_seconds(self) -> float | None:
        """Return the scan duration in seconds, or None if not completed."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize scan metadata to a dictionary."""
        return {
            "scan_id": self.scan_id,
            "scan_path": str(self.scan_path.resolve()),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "scanner_version": self.scanner_version,
            "total_files_scanned": self.total_files_scanned,
            "total_files_skipped": self.total_files_skipped,
            "supported_extensions": self.supported_extensions,
            "excluded_paths": self.excluded_paths,
        }

    def __repr__(self) -> str:
        return (
            f"ScanMetadata(id={self.scan_id!r}, "
            f"path={str(self.scan_path)!r}, "
            f"files={self.total_files_scanned})"
        )


@dataclass
class ScanSummary:
    """High-level summary statistics for a scan result.

    Attributes:
        total_providers_detected: Number of unique AI providers found.
        total_findings: Total number of risk findings generated.
        total_compliance_gaps: Total number of compliance gaps identified.
        total_detection_matches: Total raw pattern matches across all files.
        critical_findings: Count of CRITICAL severity findings.
        high_findings: Count of HIGH severity findings.
        medium_findings: Count of MEDIUM severity findings.
        low_findings: Count of LOW severity findings.
        providers_with_credentials: Providers with hardcoded credential exposure.
        providers_with_data_residency_risk: Providers with data residency concerns.
        providers_with_vendor_lock_in: Providers with vendor lock-in risk.
        overall_risk_level: Highest risk level found across all findings.
        overall_risk_score: Aggregate risk score (0.0 - 10.0).
    """

    total_providers_detected: int = 0
    total_findings: int = 0
    total_compliance_gaps: int = 0
    total_detection_matches: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    providers_with_credentials: list[str] = field(default_factory=list)
    providers_with_data_residency_risk: list[str] = field(default_factory=list)
    providers_with_vendor_lock_in: list[str] = field(default_factory=list)
    overall_risk_level: RiskLevel = RiskLevel.INFO
    overall_risk_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        if isinstance(self.overall_risk_level, str):
            try:
                self.overall_risk_level = RiskLevel(self.overall_risk_level)
            except ValueError:
                self.overall_risk_level = RiskLevel.INFO
        if not (0.0 <= self.overall_risk_score <= 10.0):
            self.overall_risk_score = max(0.0, min(10.0, self.overall_risk_score))

    def to_dict(self) -> dict[str, Any]:
        """Serialize scan summary to a dictionary."""
        return {
            "total_providers_detected": self.total_providers_detected,
            "total_findings": self.total_findings,
            "total_compliance_gaps": self.total_compliance_gaps,
            "total_detection_matches": self.total_detection_matches,
            "findings_by_severity": {
                "critical": self.critical_findings,
                "high": self.high_findings,
                "medium": self.medium_findings,
                "low": self.low_findings,
            },
            "providers_with_credentials": self.providers_with_credentials,
            "providers_with_data_residency_risk": self.providers_with_data_residency_risk,
            "providers_with_vendor_lock_in": self.providers_with_vendor_lock_in,
            "overall_risk_level": self.overall_risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 2),
        }

    def __repr__(self) -> str:
        return (
            f"ScanSummary(providers={self.total_providers_detected}, "
            f"findings={self.total_findings}, "
            f"risk={self.overall_risk_level.value!r}, "
            f"score={self.overall_risk_score:.1f})"
        )


@dataclass
class ScanResult:
    """The complete, aggregated result of a full codebase scan.

    This is the top-level output object that contains all detected AI assets,
    risk findings, compliance gaps, and metadata from a single scan run.

    Attributes:
        metadata: Metadata about the scan execution.
        assets: Dictionary mapping provider_id to detected AIAsset.
        findings: List of risk findings sorted by severity.
        summary: High-level summary statistics.
    """

    metadata: ScanMetadata
    assets: dict[str, AIAsset] = field(default_factory=dict)
    findings: list[RiskFinding] = field(default_factory=list)
    summary: ScanSummary = field(default_factory=ScanSummary)

    def add_asset(self, asset: AIAsset) -> None:
        """Add or update an AI asset in the result.

        If an asset with the same provider_id already exists, its detection
        matches are merged into the existing asset.

        Args:
            asset: The AIAsset to add or merge.
        """
        if asset.provider_id in self.assets:
            existing = self.assets[asset.provider_id]
            for match in asset.detection_matches:
                existing.add_match(match)
        else:
            self.assets[asset.provider_id] = asset

    def add_finding(self, finding: RiskFinding) -> None:
        """Add a risk finding to the result.

        Args:
            finding: The RiskFinding to add.
        """
        self.findings.append(finding)

    def get_findings_by_risk_level(self, risk_level: RiskLevel) -> list[RiskFinding]:
        """Return all findings matching the specified risk level.

        Args:
            risk_level: The RiskLevel to filter by.

        Returns:
            List of RiskFinding objects at the specified severity.
        """
        return [f for f in self.findings if f.risk_level == risk_level]

    def get_asset(self, provider_id: str) -> AIAsset | None:
        """Return the AIAsset for a given provider_id, or None if not found.

        Args:
            provider_id: The provider identifier to look up.

        Returns:
            The AIAsset if found, otherwise None.
        """
        return self.assets.get(provider_id)

    @property
    def sorted_findings(self) -> list[RiskFinding]:
        """Return findings sorted by risk score descending."""
        return sorted(self.findings, key=lambda f: f.risk_score, reverse=True)

    @property
    def all_affected_files(self) -> set[Path]:
        """Return the union of all files affected across all assets."""
        result: set[Path] = set()
        for asset in self.assets.values():
            result.update(asset.files_detected)
        return result

    def compute_summary(self) -> ScanSummary:
        """Compute and update the scan summary from current assets and findings.

        Returns:
            The updated ScanSummary reflecting current scan state.
        """
        critical = sum(1 for f in self.findings if f.risk_level == RiskLevel.CRITICAL)
        high = sum(1 for f in self.findings if f.risk_level == RiskLevel.HIGH)
        medium = sum(1 for f in self.findings if f.risk_level == RiskLevel.MEDIUM)
        low = sum(1 for f in self.findings if f.risk_level == RiskLevel.LOW)

        total_matches = sum(a.total_matches for a in self.assets.values())
        total_gaps = sum(len(f.compliance_gaps) for f in self.findings)

        with_creds = [
            a.provider_id for a in self.assets.values() if a.has_credential_exposure
        ]
        with_residency = [
            a.provider_id for a in self.assets.values() if a.data_residency_concern
        ]
        with_lock_in = [
            a.provider_id for a in self.assets.values() if a.vendor_lock_in
        ]

        # Determine overall risk level from highest finding
        overall_risk = RiskLevel.INFO
        if self.findings:
            overall_risk = max(f.risk_level for f in self.findings)

        # Compute aggregate risk score as weighted average
        overall_score = 0.0
        if self.findings:
            total_score = sum(f.risk_score for f in self.findings)
            overall_score = min(10.0, total_score / len(self.findings))
            # Boost score toward maximum if critical findings exist
            if critical > 0:
                overall_score = min(10.0, overall_score + (critical * 0.5))

        self.summary = ScanSummary(
            total_providers_detected=len(self.assets),
            total_findings=len(self.findings),
            total_compliance_gaps=total_gaps,
            total_detection_matches=total_matches,
            critical_findings=critical,
            high_findings=high,
            medium_findings=medium,
            low_findings=low,
            providers_with_credentials=with_creds,
            providers_with_data_residency_risk=with_residency,
            providers_with_vendor_lock_in=with_lock_in,
            overall_risk_level=overall_risk,
            overall_risk_score=round(overall_score, 2),
        )
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete scan result to a dictionary.

        Returns:
            A fully serializable dictionary representation of the scan result.
        """
        return {
            "schema_version": "1.0",
            "metadata": self.metadata.to_dict(),
            "summary": self.summary.to_dict(),
            "assets": {
                provider_id: asset.to_dict()
                for provider_id, asset in self.assets.items()
            },
            "findings": [f.to_dict() for f in self.sorted_findings],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the scan result to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.

        Returns:
            A formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return (
            f"ScanResult("
            f"assets={len(self.assets)}, "
            f"findings={len(self.findings)}, "
            f"files={len(self.all_affected_files)})"
        )


def make_scan_result(scan_path: str | Path, scan_id: str | None = None) -> ScanResult:
    """Factory function to create an initialized ScanResult for a new scan.

    Args:
        scan_path: The root directory path being scanned.
        scan_id: Optional unique scan identifier. Auto-generated if not provided.

    Returns:
        A new ScanResult with metadata initialized and ready for use.
    """
    import uuid

    if scan_id is None:
        scan_id = str(uuid.uuid4())

    from ai_risk_scanner import __version__

    metadata = ScanMetadata(
        scan_id=scan_id,
        scan_path=Path(scan_path),
        started_at=datetime.now(tz=timezone.utc),
        scanner_version=__version__,
    )
    return ScanResult(metadata=metadata)
