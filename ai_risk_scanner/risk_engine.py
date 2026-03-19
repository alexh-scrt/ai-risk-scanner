"""Risk scoring and classification engine for the AI Risk Inventory Scanner.

This module implements the risk scoring logic that maps detected AI provider
assets to NIST AI RMF functions and CTEM exposure categories, generates
compliance gap findings, and produces structured RiskFinding objects with
numeric severity scores.

Key classes:
    - RiskEngine: Main engine that analyzes detected assets and produces findings.
    - RiskScoringConfig: Configuration for risk score weighting and thresholds.

Key functions:
    - compute_base_score: Calculate a numeric score from a risk level.
    - apply_ctem_modifiers: Apply CTEM severity modifiers to a base score.

Risk scoring methodology:
    - Base score is derived from the provider's baseline_risk_level.
    - CTEM category modifiers (from providers.yaml) are applied multiplicatively.
    - Presence of hardcoded credentials elevates score by a fixed increment.
    - Vendor lock-in and data residency flags each add minor score increments.
    - Final score is clamped to [0.0, 10.0].

Usage example::

    from ai_risk_scanner.risk_engine import RiskEngine
    from ai_risk_scanner.detectors import DetectionEngine
    from pathlib import Path

    det_engine = DetectionEngine()
    file_results = det_engine.scan_files([Path('app.py'), Path('.env')])
    assets = det_engine.aggregate_results(file_results)

    risk_engine = RiskEngine()
    findings = risk_engine.analyze(assets)
    for finding in findings:
        print(finding.risk_level.value, finding.title, finding.risk_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai_risk_scanner.inventory import (
    AIAsset,
    ComplianceGap,
    DetectionType,
    RiskFinding,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# Path to the bundled providers.yaml signature database
_PROVIDERS_YAML_PATH = Path(__file__).parent / "providers.yaml"

# Base numeric scores per risk level (before CTEM modifiers)
_BASE_SCORES: dict[RiskLevel, float] = {
    RiskLevel.CRITICAL: 9.0,
    RiskLevel.HIGH: 7.0,
    RiskLevel.MEDIUM: 5.0,
    RiskLevel.LOW: 2.5,
    RiskLevel.INFO: 1.0,
}

# Score increments for specific risk conditions
_CREDENTIAL_SCORE_INCREMENT: float = 2.0
_VENDOR_LOCK_IN_INCREMENT: float = 0.3
_DATA_RESIDENCY_INCREMENT: float = 0.4
_MULTI_FILE_INCREMENT_BASE: float = 0.1  # per additional file beyond the first
_MULTI_FILE_INCREMENT_MAX: float = 0.5

# Minimum score that elevates a finding to CRITICAL
_CRITICAL_THRESHOLD: float = 8.5
_HIGH_THRESHOLD: float = 6.5
_MEDIUM_THRESHOLD: float = 4.0
_LOW_THRESHOLD: float = 0.0


def compute_base_score(risk_level: RiskLevel) -> float:
    """Return the base numeric risk score for a given risk level.

    Args:
        risk_level: The RiskLevel to convert to a numeric score.

    Returns:
        A float in the range [1.0, 9.0] representing the base score.
    """
    return _BASE_SCORES.get(risk_level, 5.0)


def score_to_risk_level(score: float) -> RiskLevel:
    """Convert a numeric risk score back to a RiskLevel enum value.

    Args:
        score: A numeric risk score in [0.0, 10.0].

    Returns:
        The corresponding RiskLevel.
    """
    if score >= _CRITICAL_THRESHOLD:
        return RiskLevel.CRITICAL
    elif score >= _HIGH_THRESHOLD:
        return RiskLevel.HIGH
    elif score >= _MEDIUM_THRESHOLD:
        return RiskLevel.MEDIUM
    elif score > _LOW_THRESHOLD:
        return RiskLevel.LOW
    return RiskLevel.INFO


def apply_ctem_modifiers(
    base_score: float,
    ctem_categories: list[str],
    ctem_metadata: dict[str, dict[str, Any]],
) -> float:
    """Apply CTEM category severity modifiers to a base risk score.

    Each CTEM category has an associated severity_modifier from the
    providers.yaml database. The highest modifier among applicable
    categories is applied to the base score (multiplicative).

    Args:
        base_score: The starting numeric risk score.
        ctem_categories: List of CTEM category IDs applicable to this finding.
        ctem_metadata: Dictionary of CTEM category metadata from providers.yaml,
            keyed by category ID.

    Returns:
        Modified score, clamped to [0.0, 10.0].
    """
    if not ctem_categories or not ctem_metadata:
        return min(10.0, max(0.0, base_score))

    max_modifier = 1.0
    for cat_id in ctem_categories:
        cat_info = ctem_metadata.get(cat_id, {})
        modifier = float(cat_info.get("severity_modifier", 1.0))
        if modifier > max_modifier:
            max_modifier = modifier

    modified = base_score * max_modifier
    return min(10.0, max(0.0, modified))


@dataclass
class RiskScoringConfig:
    """Configuration parameters for risk score calculation.

    Attributes:
        credential_score_increment: Score added when hardcoded credentials are found.
        vendor_lock_in_increment: Score added for vendor lock-in risk.
        data_residency_increment: Score added for data residency concerns.
        multi_file_increment_base: Per-additional-file score increment.
        multi_file_increment_max: Maximum total increment from multi-file spread.
        apply_ctem_modifiers: Whether to apply CTEM severity modifiers.
    """

    credential_score_increment: float = _CREDENTIAL_SCORE_INCREMENT
    vendor_lock_in_increment: float = _VENDOR_LOCK_IN_INCREMENT
    data_residency_increment: float = _DATA_RESIDENCY_INCREMENT
    multi_file_increment_base: float = _MULTI_FILE_INCREMENT_BASE
    multi_file_increment_max: float = _MULTI_FILE_INCREMENT_MAX
    apply_ctem_modifiers: bool = True


class RiskEngine:
    """Risk scoring and classification engine.

    The RiskEngine analyzes a dictionary of detected AIAsset objects and
    produces a list of RiskFinding objects, each containing:
    - A numeric risk score (0.0 - 10.0).
    - A mapped RiskLevel severity.
    - Applicable NIST AI RMF function references.
    - Applicable CTEM exposure category references.
    - A list of ComplianceGap findings.

    The engine loads CTEM and NIST metadata from the providers.yaml database
    to apply category-specific severity modifiers and populate framework
    mapping information.

    Attributes:
        yaml_path: Path to the providers.yaml database.
        scoring_config: Configuration for risk score computation.
        ctem_metadata: Loaded CTEM category definitions.
        nist_metadata: Loaded NIST AI RMF function definitions.
    """

    def __init__(
        self,
        yaml_path: Path | None = None,
        scoring_config: RiskScoringConfig | None = None,
    ) -> None:
        """Initialize the RiskEngine and load framework metadata.

        Args:
            yaml_path: Optional path to a custom providers.yaml.
                       Defaults to the bundled providers.yaml.
            scoring_config: Optional custom scoring configuration.
                            Uses defaults if not provided.

        Raises:
            FileNotFoundError: If the providers.yaml file is not found.
            yaml.YAMLError: If the providers.yaml content is invalid.
        """
        self.yaml_path: Path = yaml_path or _PROVIDERS_YAML_PATH
        self.scoring_config: RiskScoringConfig = scoring_config or RiskScoringConfig()
        self.ctem_metadata: dict[str, dict[str, Any]] = {}
        self.nist_metadata: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load CTEM and NIST AI RMF metadata from the providers.yaml database.

        Populates self.ctem_metadata and self.nist_metadata.
        Logs warnings on failure rather than raising to allow graceful degradation.
        """
        if not self.yaml_path.exists():
            logger.warning(
                "providers.yaml not found at %s; CTEM/NIST metadata unavailable.",
                self.yaml_path,
            )
            return

        try:
            with self.yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load providers.yaml for risk metadata: %s", exc)
            return

        if not isinstance(data, dict):
            logger.warning("providers.yaml has unexpected root structure.")
            return

        raw_ctem = data.get("ctem_categories", {})
        if isinstance(raw_ctem, dict):
            self.ctem_metadata = raw_ctem

        raw_nist = data.get("nist_rmf_functions", {})
        if isinstance(raw_nist, dict):
            self.nist_metadata = raw_nist

        logger.debug(
            "Loaded %d CTEM categories and %d NIST RMF functions from %s.",
            len(self.ctem_metadata),
            len(self.nist_metadata),
            self.yaml_path,
        )

    def analyze(
        self,
        assets: dict[str, AIAsset],
    ) -> list[RiskFinding]:
        """Analyze detected AI assets and produce a list of risk findings.

        For each detected provider asset, this method:
        1. Computes a risk score based on baseline level, CTEM modifiers, and
           aggravating factors (credentials, vendor lock-in, data residency).
        2. Generates compliance gap findings from the provider's compliance_notes
           and detected evidence types.
        3. Constructs a RiskFinding with all relevant metadata.

        Assets with provider_id '_credentials' (generic credential catches)
        are processed separately into a credential-specific finding.

        Args:
            assets: Dictionary mapping provider_id to AIAsset objects.

        Returns:
            List of RiskFinding objects sorted by risk_score descending.
        """
        findings: list[RiskFinding] = []

        for provider_id, asset in assets.items():
            if provider_id == "_credentials":
                finding = self._analyze_credential_asset(asset)
            else:
                finding = self._analyze_provider_asset(asset)

            if finding is not None:
                findings.append(finding)

        # Sort by risk_score descending, then by provider name for stability
        findings.sort(key=lambda f: (-f.risk_score, f.provider_name))
        return findings

    def _analyze_provider_asset(self, asset: AIAsset) -> RiskFinding | None:
        """Produce a RiskFinding for a specific AI provider asset.

        Args:
            asset: The AIAsset to analyze.

        Returns:
            A RiskFinding, or None if the asset has no detection matches.
        """
        if asset.total_matches == 0:
            return None

        # Compute base score from the provider's baseline risk level
        base_score = compute_base_score(asset.baseline_risk_level)

        # Apply CTEM modifiers if configured
        if self.scoring_config.apply_ctem_modifiers:
            score = apply_ctem_modifiers(
                base_score=base_score,
                ctem_categories=asset.ctem_categories,
                ctem_metadata=self.ctem_metadata,
            )
        else:
            score = base_score

        # Aggravating factor: hardcoded credentials
        has_credentials = asset.has_credential_exposure
        if has_credentials:
            score = min(10.0, score + self.scoring_config.credential_score_increment)

        # Aggravating factor: vendor lock-in
        if asset.vendor_lock_in:
            score = min(10.0, score + self.scoring_config.vendor_lock_in_increment)

        # Aggravating factor: data residency concern
        if asset.data_residency_concern:
            score = min(10.0, score + self.scoring_config.data_residency_increment)

        # Aggravating factor: wide spread across multiple files
        num_files = len(asset.files_detected)
        if num_files > 1:
            file_increment = min(
                self.scoring_config.multi_file_increment_max,
                (num_files - 1) * self.scoring_config.multi_file_increment_base,
            )
            score = min(10.0, score + file_increment)

        score = round(score, 2)
        risk_level = score_to_risk_level(score)

        # Generate compliance gap findings
        compliance_gaps = self._generate_compliance_gaps(asset, has_credentials)

        # Build description
        description = self._build_finding_description(asset)

        finding = RiskFinding(
            finding_id=f"finding_{asset.provider_id}_{asset.category.value}",
            title=f"{asset.provider_name} Integration Detected",
            description=description,
            risk_level=risk_level,
            risk_score=score,
            provider_id=asset.provider_id,
            provider_name=asset.provider_name,
            nist_functions=list(asset.nist_rmf_functions),
            ctem_categories=list(asset.ctem_categories),
            detection_matches=list(asset.detection_matches),
            compliance_gaps=compliance_gaps,
            vendor_lock_in=asset.vendor_lock_in,
            data_residency_concern=asset.data_residency_concern,
            has_hardcoded_credentials=has_credentials,
        )
        return finding

    def _analyze_credential_asset(self, asset: AIAsset) -> RiskFinding | None:
        """Produce a RiskFinding for unattributed hardcoded credentials.

        Args:
            asset: The generic '_credentials' AIAsset.

        Returns:
            A RiskFinding for unattributed credential exposure, or None if empty.
        """
        if asset.total_matches == 0:
            return None

        score = 9.5  # Hardcoded credentials are always near-critical
        risk_level = RiskLevel.CRITICAL

        compliance_gaps = [
            ComplianceGap(
                gap_id="gap_hardcoded_credentials",
                title="Hardcoded Credentials Detected",
                description=(
                    "One or more API keys, tokens, or secrets appear to be "
                    "hardcoded directly in source files. This poses a critical "
                    "security risk as credentials may be exposed via version "
                    "control history or unauthorized access."
                ),
                risk_level=RiskLevel.CRITICAL,
                nist_functions=["GOVERN-1", "MANAGE-2"],
                ctem_categories=["identity_access"],
                remediation=(
                    "Immediately rotate all exposed credentials. "
                    "Use environment variables, a secrets manager (e.g., AWS Secrets Manager, "
                    "HashiCorp Vault, or Azure Key Vault), or a .env file excluded from "
                    "version control to store sensitive credentials."
                ),
                provider_id="_credentials",
            )
        ]

        finding = RiskFinding(
            finding_id="finding_hardcoded_credentials",
            title="Hardcoded AI API Credentials Detected",
            description=(
                f"Detected {asset.total_matches} hardcoded credential pattern(s) "
                f"across {len(asset.files_detected)} file(s). "
                "Hardcoded API keys and tokens in source code represent a critical "
                "security vulnerability that can lead to unauthorized access, cost "
                "overruns, and data breaches."
            ),
            risk_level=risk_level,
            risk_score=score,
            provider_id="_credentials",
            provider_name="Hardcoded Credentials",
            nist_functions=["GOVERN-1", "MANAGE-2"],
            ctem_categories=["identity_access"],
            detection_matches=list(asset.detection_matches),
            compliance_gaps=compliance_gaps,
            vendor_lock_in=False,
            data_residency_concern=False,
            has_hardcoded_credentials=True,
        )
        return finding

    def _generate_compliance_gaps(
        self,
        asset: AIAsset,
        has_credentials: bool,
    ) -> list[ComplianceGap]:
        """Generate a list of ComplianceGap objects for a detected provider asset.

        This method inspects the asset's detection evidence and provider metadata
        to produce targeted compliance gap findings covering:
        - Hardcoded credential exposure.
        - Data residency concerns.
        - Vendor lock-in risk.
        - Missing rate limiting / access controls.
        - Provider-specific compliance notes.

        Args:
            asset: The AIAsset being analyzed.
            has_credentials: Whether hardcoded credentials were detected.

        Returns:
            List of ComplianceGap objects for this asset.
        """
        gaps: list[ComplianceGap] = []

        # Gap 1: Hardcoded credentials
        if has_credentials:
            gaps.append(ComplianceGap(
                gap_id=f"gap_{asset.provider_id}_hardcoded_credentials",
                title=f"Hardcoded {asset.provider_name} Credentials",
                description=(
                    f"Hardcoded API keys or tokens for {asset.provider_name} were "
                    f"detected in source files. This is a critical security risk."
                ),
                risk_level=RiskLevel.CRITICAL,
                nist_functions=["GOVERN-1", "MANAGE-2"],
                ctem_categories=["identity_access"],
                remediation=(
                    f"Remove hardcoded {asset.provider_name} credentials immediately. "
                    "Store secrets in environment variables, a dedicated secrets manager, "
                    "or a vault solution. Rotate all exposed credentials."
                ),
                provider_id=asset.provider_id,
            ))

        # Gap 2: Data residency concern
        if asset.data_residency_concern:
            gaps.append(ComplianceGap(
                gap_id=f"gap_{asset.provider_id}_data_residency",
                title=f"{asset.provider_name} Data Residency Risk",
                description=(
                    f"Data processed by {asset.provider_name} may be transferred to "
                    "third-party infrastructure outside your jurisdiction, potentially "
                    "violating GDPR, CCPA, HIPAA, or other data protection regulations."
                ),
                risk_level=RiskLevel.HIGH,
                nist_functions=["MAP-5", "MEASURE-2"],
                ctem_categories=["data_exfiltration", "external_exposure"],
                remediation=(
                    f"Review {asset.provider_name}'s data processing agreement and "
                    "privacy policy. Configure region-specific endpoints where available. "
                    "Avoid sending PII, PHI, or sensitive business data to this API "
                    "without appropriate legal safeguards."
                ),
                provider_id=asset.provider_id,
            ))

        # Gap 3: Vendor lock-in
        if asset.vendor_lock_in:
            gaps.append(ComplianceGap(
                gap_id=f"gap_{asset.provider_id}_vendor_lock_in",
                title=f"{asset.provider_name} Vendor Lock-In Risk",
                description=(
                    f"The codebase uses {asset.provider_name}'s proprietary SDK or API, "
                    "creating a dependency that may impede switching providers, negotiating "
                    "pricing, or maintaining service continuity if the provider changes "
                    "their terms or experiences an outage."
                ),
                risk_level=RiskLevel.MEDIUM,
                nist_functions=["MAP-3", "MANAGE-1"],
                ctem_categories=["third_party_dependency"],
                remediation=(
                    f"Abstract {asset.provider_name} API calls behind an interface or "
                    "adapter layer to allow provider substitution. Consider multi-provider "
                    "strategies or open standards where feasible."
                ),
                provider_id=asset.provider_id,
            ))

        # Gap 4: No env var usage detected (potential for hardcoding)
        if not asset.has_env_var_usage and not has_credentials:
            # Only flag if we detected imports/SDK calls but no env var references
            has_imports = DetectionType.IMPORT in asset.detection_types_found
            has_sdk = DetectionType.SDK_CALL in asset.detection_types_found
            if has_imports or has_sdk:
                gaps.append(ComplianceGap(
                    gap_id=f"gap_{asset.provider_id}_no_env_var",
                    title=f"{asset.provider_name} API Key Management Not Verified",
                    description=(
                        f"{asset.provider_name} SDK usage was detected but no environment "
                        "variable references for API keys were found. Credentials may be "
                        "managed outside this codebase, or may be hardcoded elsewhere."
                    ),
                    risk_level=RiskLevel.MEDIUM,
                    nist_functions=["GOVERN-1", "MANAGE-2"],
                    ctem_categories=["identity_access"],
                    remediation=(
                        f"Ensure {asset.provider_name} API keys are loaded from environment "
                        "variables (e.g., os.getenv) or a secrets manager. Never store "
                        "credentials in source code or configuration files committed to "
                        "version control."
                    ),
                    provider_id=asset.provider_id,
                ))

        # Gap 5: Provider-specific compliance notes (from providers.yaml)
        for idx, note in enumerate(asset.compliance_notes):
            gaps.append(ComplianceGap(
                gap_id=f"gap_{asset.provider_id}_compliance_note_{idx}",
                title=f"{asset.provider_name} Compliance Note",
                description=note,
                risk_level=RiskLevel.LOW,
                nist_functions=list(asset.nist_rmf_functions),
                ctem_categories=list(asset.ctem_categories),
                remediation=(
                    f"Review the compliance note and assess applicability to your "
                    f"deployment context. Consult {asset.provider_name} documentation "
                    "for guidance: " + (asset.docs_url or "see provider docs.")
                ),
                provider_id=asset.provider_id,
            ))

        return gaps

    def _build_finding_description(self, asset: AIAsset) -> str:
        """Build a human-readable description for a provider finding.

        Args:
            asset: The AIAsset being described.

        Returns:
            A multi-sentence description string.
        """
        num_files = len(asset.files_detected)
        num_matches = asset.total_matches
        detection_types = sorted(dt.value for dt in asset.detection_types_found)

        parts = [
            f"{asset.provider_name} ({asset.category.value}) integration detected "
            f"with {num_matches} reference(s) across {num_files} file(s).",
        ]

        if detection_types:
            parts.append(
                f"Evidence types: {', '.join(detection_types)}."
            )

        risk_factors = []
        if asset.has_credential_exposure:
            risk_factors.append("hardcoded credentials")
        if asset.data_residency_concern:
            risk_factors.append("data residency concern")
        if asset.vendor_lock_in:
            risk_factors.append("vendor lock-in risk")

        if risk_factors:
            parts.append(f"Risk factors: {', '.join(risk_factors)}.")

        if asset.docs_url:
            parts.append(f"Provider documentation: {asset.docs_url}")

        return " ".join(parts)

    def get_nist_function_description(self, function_id: str) -> str:
        """Return the human-readable description for a NIST AI RMF function.

        Args:
            function_id: The NIST AI RMF function identifier (e.g., 'GOVERN-1').

        Returns:
            The function description string, or a generic fallback if not found.
        """
        func_data = self.nist_metadata.get(function_id, {})
        return func_data.get("description", f"NIST AI RMF function {function_id}")

    def get_nist_function_name(self, function_id: str) -> str:
        """Return the human-readable name for a NIST AI RMF function.

        Args:
            function_id: The NIST AI RMF function identifier (e.g., 'GOVERN-1').

        Returns:
            The function name string, or the function_id as fallback.
        """
        func_data = self.nist_metadata.get(function_id, {})
        return func_data.get("name", function_id)

    def get_ctem_category_description(self, category_id: str) -> str:
        """Return the human-readable description for a CTEM exposure category.

        Args:
            category_id: The CTEM category identifier (e.g., 'external_exposure').

        Returns:
            The category description string, or a generic fallback if not found.
        """
        cat_data = self.ctem_metadata.get(category_id, {})
        return cat_data.get(
            "description",
            f"CTEM exposure category: {category_id}",
        )

    def get_ctem_category_name(self, category_id: str) -> str:
        """Return the human-readable name for a CTEM exposure category.

        Args:
            category_id: The CTEM category identifier.

        Returns:
            The category name string, or the category_id as fallback.
        """
        cat_data = self.ctem_metadata.get(category_id, {})
        return cat_data.get("name", category_id)

    def build_nist_mapping(
        self,
        findings: list[RiskFinding],
    ) -> dict[str, list[str]]:
        """Build a mapping from NIST AI RMF function IDs to finding IDs.

        This is useful for reporting which findings relate to each
        NIST AI RMF function.

        Args:
            findings: List of RiskFinding objects to map.

        Returns:
            Dictionary mapping NIST function ID to list of finding IDs.
        """
        mapping: dict[str, list[str]] = {}
        for finding in findings:
            for func_id in finding.nist_functions:
                mapping.setdefault(func_id, []).append(finding.finding_id)
        return mapping

    def build_ctem_mapping(
        self,
        findings: list[RiskFinding],
    ) -> dict[str, list[str]]:
        """Build a mapping from CTEM category IDs to finding IDs.

        Args:
            findings: List of RiskFinding objects to map.

        Returns:
            Dictionary mapping CTEM category ID to list of finding IDs.
        """
        mapping: dict[str, list[str]] = {}
        for finding in findings:
            for cat_id in finding.ctem_categories:
                mapping.setdefault(cat_id, []).append(finding.finding_id)
        return mapping

    def compute_overall_risk_score(
        self,
        findings: list[RiskFinding],
    ) -> tuple[float, RiskLevel]:
        """Compute the aggregate risk score and level across all findings.

        The aggregate score is computed as the weighted average of individual
        finding scores, with additional weight given to CRITICAL findings.

        Args:
            findings: List of RiskFinding objects.

        Returns:
            A tuple of (overall_score: float, overall_level: RiskLevel).
        """
        if not findings:
            return 0.0, RiskLevel.INFO

        total_score = sum(f.risk_score for f in findings)
        avg_score = total_score / len(findings)

        # Boost for critical findings
        critical_count = sum(
            1 for f in findings if f.risk_level == RiskLevel.CRITICAL
        )
        if critical_count > 0:
            avg_score = min(10.0, avg_score + critical_count * 0.5)

        overall_score = round(min(10.0, avg_score), 2)
        overall_level = score_to_risk_level(overall_score)

        return overall_score, overall_level

    def __repr__(self) -> str:
        return (
            f"RiskEngine("
            f"yaml_path={str(self.yaml_path)!r}, "
            f"ctem_categories={len(self.ctem_metadata)}, "
            f"nist_functions={len(self.nist_metadata)})"
        )
