"""Report generation module for the AI Risk Inventory Scanner.

This module implements multi-format report generation from structured ScanResult
objects. It produces:
  - A rich terminal summary with colored tables and panels.
  - A machine-readable JSON inventory export.
  - A Markdown audit report suitable for documentation and compliance workflows.

Key classes:
    - ReportConfig: Configuration options for report generation.
    - ConsoleReporter: Renders rich terminal output using the rich library.
    - JSONReporter: Serializes ScanResult to a formatted JSON string or file.
    - MarkdownReporter: Generates a Markdown audit report from a ScanResult.
    - Reporter: Facade class that delegates to all three reporters.

Usage example::

    from pathlib import Path
    from ai_risk_scanner.reporter import Reporter, ReportConfig

    config = ReportConfig(show_evidence=True, max_evidence_lines=5)
    reporter = Reporter(config=config)

    # Print to terminal
    reporter.print_console(scan_result)

    # Export JSON
    reporter.write_json(scan_result, Path('report.json'))

    # Export Markdown
    reporter.write_markdown(scan_result, Path('report.md'))
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ai_risk_scanner.inventory import (
    AIAsset,
    ComplianceGap,
    DetectionMatch,
    RiskFinding,
    RiskLevel,
    ScanResult,
)

# ---------------------------------------------------------------------------
# Colour / style constants for rich terminal output
# ---------------------------------------------------------------------------

_RISK_COLOURS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "bold red",
    RiskLevel.HIGH: "bold yellow",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.LOW: "cyan",
    RiskLevel.INFO: "dim",
}

_RISK_ICONS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "\u2718",  # ✘
    RiskLevel.HIGH: "\u26a0",      # ⚠
    RiskLevel.MEDIUM: "\u25cf",    # ●
    RiskLevel.LOW: "\u25cb",       # ○
    RiskLevel.INFO: "\u2139",      # ℹ
}

_CATEGORY_LABELS: dict[str, str] = {
    "llm": "LLM",
    "image_gen": "Image Gen",
    "speech": "Speech",
    "embedding": "Embedding",
    "mlops": "MLOps",
    "cloud_ai": "Cloud AI",
    "unknown": "Unknown",
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReportConfig:
    """Configuration options for report generation.

    Attributes:
        show_evidence: Whether to include individual detection match evidence
            in detailed sections of the report.
        max_evidence_lines: Maximum number of evidence lines to show per finding.
        show_compliance_gaps: Whether to include compliance gap detail.
        show_nist_mapping: Whether to include the NIST AI RMF function mapping table.
        show_ctem_mapping: Whether to include the CTEM exposure category mapping.
        show_affected_files: Whether to list affected files per finding.
        truncate_line_length: Maximum line content length for display (characters).
        indent: Number of spaces for JSON indentation.
        console_width: Override terminal width for rich output (None = auto).
        verbose: Whether to include additional verbose details.
    """

    show_evidence: bool = False
    max_evidence_lines: int = 3
    show_compliance_gaps: bool = True
    show_nist_mapping: bool = True
    show_ctem_mapping: bool = True
    show_affected_files: bool = True
    truncate_line_length: int = 120
    indent: int = 2
    console_width: int | None = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _risk_level_badge(level: RiskLevel) -> Text:
    """Return a styled rich Text badge for a risk level.

    Args:
        level: The RiskLevel to render as a badge.

    Returns:
        A rich Text object styled with the appropriate colour.
    """
    icon = _RISK_ICONS.get(level, "")
    colour = _RISK_COLOURS.get(level, "")
    label = f"{icon} {level.value.upper()}"
    return Text(label, style=colour)


def _truncate(text: str, max_len: int) -> str:
    """Truncate a string to max_len characters, appending ellipsis if needed.

    Args:
        text: The string to truncate.
        max_len: Maximum allowed length.

    Returns:
        The original string if within length, otherwise truncated with '...'.
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _bool_icon(value: bool) -> str:
    """Return a checkmark or cross string for a boolean value.

    Args:
        value: The boolean to represent.

    Returns:
        '\u2713' (✓) for True, '\u2717' (✗) for False.
    """
    return "\u2713" if value else "\u2717"


def _format_score(score: float) -> str:
    """Format a risk score to one decimal place.

    Args:
        score: The numeric risk score.

    Returns:
        Formatted string like '7.5'.
    """
    return f"{score:.1f}"


def _nist_functions_summary(nist_functions: list[str]) -> str:
    """Join NIST AI RMF function IDs into a compact comma-separated string.

    Args:
        nist_functions: List of NIST function ID strings.

    Returns:
        Comma-separated string or '-' if empty.
    """
    if not nist_functions:
        return "-"
    return ", ".join(sorted(set(nist_functions)))


def _ctem_categories_summary(ctem_categories: list[str]) -> str:
    """Join CTEM category IDs into a compact comma-separated string.

    Args:
        ctem_categories: List of CTEM category ID strings.

    Returns:
        Comma-separated string or '-' if empty.
    """
    if not ctem_categories:
        return "-"
    return ", ".join(sorted(set(ctem_categories)))


def _severity_bar(score: float) -> str:
    """Render a compact ASCII progress bar for a risk score (0-10).

    Args:
        score: The numeric risk score in [0.0, 10.0].

    Returns:
        A string like '[========  ] 8.2'.
    """
    filled = int(round(score))
    bar = "=" * filled + " " * (10 - filled)
    return f"[{bar}] {score:.1f}/10"


# ---------------------------------------------------------------------------
# ConsoleReporter
# ---------------------------------------------------------------------------


class ConsoleReporter:
    """Renders a rich terminal summary of a ScanResult.

    Outputs are sent to a rich Console instance, which can target
    stdout, stderr, or any IO stream.

    Attributes:
        config: Report configuration options.
        console: The rich Console instance used for output.
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the ConsoleReporter.

        Args:
            config: Report configuration options. Uses defaults if not provided.
            console: A pre-configured rich Console. Creates a new one if not provided.
        """
        self.config: ReportConfig = config or ReportConfig()
        self.console: Console = console or Console(
            width=self.config.console_width,
            highlight=False,
        )

    def print(self, scan_result: ScanResult) -> None:
        """Print the full rich terminal report for a ScanResult.

        Renders the following sections in order:
          1. Header banner with scan metadata.
          2. Executive summary panel.
          3. Detected AI providers table.
          4. Risk findings table.
          5. Compliance gaps (if enabled).
          6. NIST AI RMF mapping (if enabled).
          7. CTEM exposure mapping (if enabled).
          8. Affected files list (if enabled and verbose).

        Args:
            scan_result: The ScanResult to render.
        """
        self._print_header(scan_result)
        self._print_summary_panel(scan_result)
        self._print_providers_table(scan_result)
        self._print_findings_table(scan_result)

        if self.config.show_compliance_gaps:
            self._print_compliance_gaps(scan_result)

        if self.config.show_nist_mapping:
            self._print_nist_mapping(scan_result)

        if self.config.show_ctem_mapping:
            self._print_ctem_mapping(scan_result)

        if self.config.show_evidence and self.config.verbose:
            self._print_evidence_detail(scan_result)

        self._print_footer(scan_result)

    def _print_header(self, result: ScanResult) -> None:
        """Print the scan header banner.

        Args:
            result: The ScanResult containing scan metadata.
        """
        meta = result.metadata
        title = Text()
        title.append("AI Risk Inventory Scanner", style="bold white")
        title.append("  |  ", style="dim")
        title.append("Scan Report", style="bold cyan")

        subtitle_lines = [
            f"[dim]Scan ID:[/dim]      [cyan]{meta.scan_id}[/cyan]",
            f"[dim]Target path:[/dim]  [white]{meta.scan_path}[/white]",
            f"[dim]Scanner:[/dim]      [white]v{meta.scanner_version}[/white]",
            f"[dim]Started:[/dim]      [white]{meta.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}[/white]",
        ]
        if meta.completed_at:
            duration = meta.duration_seconds or 0.0
            subtitle_lines.append(
                f"[dim]Duration:[/dim]     [white]{duration:.2f}s[/white]"
            )
        subtitle_lines.append(
            f"[dim]Files scanned:[/dim] [white]{meta.total_files_scanned}[/white] "
            f"[dim]({meta.total_files_skipped} skipped)[/dim]"
        )

        header_content = title.markup + "\n\n" + "\n".join(subtitle_lines)
        self.console.print(
            Panel(
                header_content,
                title="[bold blue]\u26a1 AI Risk Scanner[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        self.console.print()

    def _print_summary_panel(self, result: ScanResult) -> None:
        """Print the executive summary panel showing overall risk.

        Args:
            result: The ScanResult with computed summary.
        """
        summary = result.summary
        risk_colour = _RISK_COLOURS.get(summary.overall_risk_level, "white")
        risk_icon = _RISK_ICONS.get(summary.overall_risk_level, "")

        # Overall risk line
        lines = [
            f"  Overall Risk:  [{risk_colour}]{risk_icon} {summary.overall_risk_level.value.upper()}[/{risk_colour}]  "
            f"Score: [{risk_colour}]{_severity_bar(summary.overall_risk_score)}[/{risk_colour}]",
            "",
            f"  [dim]Providers detected:[/dim]   [bold white]{summary.total_providers_detected}[/bold white]",
            f"  [dim]Risk findings:[/dim]        [bold white]{summary.total_findings}[/bold white]",
            f"  [dim]Compliance gaps:[/dim]      [bold white]{summary.total_compliance_gaps}[/bold white]",
            f"  [dim]Detection matches:[/dim]    [bold white]{summary.total_detection_matches}[/bold white]",
            "",
            "  Findings by severity:",
            f"    [bold red]CRITICAL: {summary.critical_findings}[/bold red]  "
            f"[bold yellow]HIGH: {summary.high_findings}[/bold yellow]  "
            f"[yellow]MEDIUM: {summary.medium_findings}[/yellow]  "
            f"[cyan]LOW: {summary.low_findings}[/cyan]",
        ]

        if summary.providers_with_credentials:
            lines.append("")
            cred_list = ", ".join(summary.providers_with_credentials)
            lines.append(
                f"  [bold red]\u26a0 Hardcoded credentials:[/bold red] [red]{cred_list}[/red]"
            )

        if summary.providers_with_data_residency_risk:
            dr_list = ", ".join(summary.providers_with_data_residency_risk)
            lines.append(
                f"  [yellow]\u26a0 Data residency risk:[/yellow]    [yellow]{dr_list}[/yellow]"
            )

        if summary.providers_with_vendor_lock_in:
            lock_list = ", ".join(summary.providers_with_vendor_lock_in)
            lines.append(
                f"  [cyan]\u26a0 Vendor lock-in:[/cyan]          [cyan]{lock_list}[/cyan]"
            )

        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold]Executive Summary[/bold]",
                border_style=risk_colour,
                padding=(0, 1),
            )
        )
        self.console.print()

    def _print_providers_table(self, result: ScanResult) -> None:
        """Print a table of all detected AI providers.

        Args:
            result: The ScanResult containing the assets dictionary.
        """
        if not result.assets:
            self.console.print("[dim]No AI providers detected.[/dim]")
            self.console.print()
            return

        table = Table(
            title="Detected AI Providers",
            box=box.ROUNDED,
            border_style="blue",
            header_style="bold blue",
            show_lines=False,
            expand=False,
        )
        table.add_column("Provider", style="bold white", min_width=20)
        table.add_column("Category", style="cyan", min_width=10)
        table.add_column("Risk Level", min_width=12)
        table.add_column("Matches", justify="right", style="white", min_width=8)
        table.add_column("Files", justify="right", style="white", min_width=6)
        table.add_column("Lock-in", justify="center", min_width=8)
        table.add_column("Data Risk", justify="center", min_width=9)
        table.add_column("Credentials", justify="center", min_width=11)

        # Sort providers by baseline risk level descending, then name
        sorted_assets = sorted(
            result.assets.values(),
            key=lambda a: (-a.baseline_risk_level.numeric_score, a.provider_name),
        )

        for asset in sorted_assets:
            risk_colour = _RISK_COLOURS.get(asset.baseline_risk_level, "white")
            risk_text = Text(
                f"{_RISK_ICONS.get(asset.baseline_risk_level, '')} {asset.baseline_risk_level.value.upper()}",
                style=risk_colour,
            )
            category_label = _CATEGORY_LABELS.get(asset.category.value, asset.category.value)

            lock_in_text = Text(
                _bool_icon(asset.vendor_lock_in),
                style="yellow" if asset.vendor_lock_in else "dim green",
            )
            data_risk_text = Text(
                _bool_icon(asset.data_residency_concern),
                style="yellow" if asset.data_residency_concern else "dim green",
            )
            cred_text = Text(
                _bool_icon(asset.has_credential_exposure),
                style="bold red" if asset.has_credential_exposure else "dim green",
            )

            table.add_row(
                asset.provider_name,
                category_label,
                risk_text,
                str(asset.total_matches),
                str(len(asset.files_detected)),
                lock_in_text,
                data_risk_text,
                cred_text,
            )

        self.console.print(table)
        self.console.print()

    def _print_findings_table(self, result: ScanResult) -> None:
        """Print a table of risk findings sorted by severity.

        Args:
            result: The ScanResult containing risk findings.
        """
        if not result.findings:
            self.console.print("[dim]No risk findings generated.[/dim]")
            self.console.print()
            return

        table = Table(
            title="Risk Findings",
            box=box.ROUNDED,
            border_style="yellow",
            header_style="bold yellow",
            show_lines=True,
            expand=True,
        )
        table.add_column("#", justify="right", style="dim", min_width=3)
        table.add_column("Finding", style="bold white", min_width=30)
        table.add_column("Severity", min_width=12)
        table.add_column("Score", justify="right", min_width=6)
        table.add_column("NIST Functions", min_width=20)
        table.add_column("CTEM Categories", min_width=22)
        table.add_column("Gaps", justify="right", min_width=5)

        for idx, finding in enumerate(result.sorted_findings, start=1):
            risk_colour = _RISK_COLOURS.get(finding.risk_level, "white")
            severity_text = Text(
                f"{_RISK_ICONS.get(finding.risk_level, '')} {finding.risk_level.value.upper()}",
                style=risk_colour,
            )
            score_text = Text(_format_score(finding.risk_score), style=risk_colour)

            nist_str = _nist_functions_summary(finding.nist_functions)
            ctem_str = _ctem_categories_summary(finding.ctem_categories)

            # Truncate long strings for table readability
            nist_display = _truncate(nist_str, 30)
            ctem_display = _truncate(ctem_str, 35)

            title_display = _truncate(finding.title, 40)

            table.add_row(
                str(idx),
                title_display,
                severity_text,
                score_text,
                nist_display,
                ctem_display,
                str(len(finding.compliance_gaps)),
            )

        self.console.print(table)
        self.console.print()

    def _print_compliance_gaps(self, result: ScanResult) -> None:
        """Print detailed compliance gap information.

        Args:
            result: The ScanResult containing findings with compliance gaps.
        """
        all_gaps: list[tuple[str, ComplianceGap]] = []
        for finding in result.sorted_findings:
            for gap in finding.compliance_gaps:
                all_gaps.append((finding.provider_name, gap))

        if not all_gaps:
            return

        # Group critical/high gaps for the summary
        critical_high = [
            (pname, g) for pname, g in all_gaps
            if g.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]

        table = Table(
            title="Compliance Gaps",
            box=box.SIMPLE_HEAVY,
            border_style="red",
            header_style="bold red",
            show_lines=True,
            expand=True,
        )
        table.add_column("Provider", style="bold white", min_width=18)
        table.add_column("Gap Title", style="white", min_width=30)
        table.add_column("Severity", min_width=12)
        table.add_column("Description", min_width=40)

        shown = 0
        max_gaps = 25 if not self.config.verbose else 100

        for provider_name, gap in all_gaps:
            if shown >= max_gaps:
                break
            risk_colour = _RISK_COLOURS.get(gap.risk_level, "white")
            severity_text = Text(
                f"{_RISK_ICONS.get(gap.risk_level, '')} {gap.risk_level.value.upper()}",
                style=risk_colour,
            )
            desc_display = _truncate(gap.description, 80)
            table.add_row(
                provider_name,
                _truncate(gap.title, 40),
                severity_text,
                desc_display,
            )
            shown += 1

        self.console.print(table)

        if len(all_gaps) > max_gaps:
            self.console.print(
                f"[dim]  ... and {len(all_gaps) - max_gaps} more compliance gaps. "
                "Use --verbose for full list or export to JSON/Markdown.[/dim]"
            )
        self.console.print()

    def _print_nist_mapping(self, result: ScanResult) -> None:
        """Print a summary of NIST AI RMF function coverage.

        Args:
            result: The ScanResult containing findings.
        """
        if not result.findings:
            return

        # Build NIST function -> provider list mapping
        nist_map: dict[str, list[str]] = {}
        for finding in result.findings:
            for func in finding.nist_functions:
                nist_map.setdefault(func, []).append(finding.provider_name)

        if not nist_map:
            return

        table = Table(
            title="NIST AI RMF Function Coverage",
            box=box.SIMPLE,
            border_style="blue",
            header_style="bold blue",
            show_lines=False,
        )
        table.add_column("Function ID", style="bold cyan", min_width=14)
        table.add_column("Providers Affected", style="white", min_width=40)
        table.add_column("Count", justify="right", min_width=6)

        for func_id in sorted(nist_map.keys()):
            providers = sorted(set(nist_map[func_id]))
            provider_str = _truncate(", ".join(providers), 60)
            table.add_row(func_id, provider_str, str(len(providers)))

        self.console.print(table)
        self.console.print()

    def _print_ctem_mapping(self, result: ScanResult) -> None:
        """Print a summary of CTEM exposure category coverage.

        Args:
            result: The ScanResult containing findings.
        """
        if not result.findings:
            return

        # Build CTEM category -> provider list mapping
        ctem_map: dict[str, list[str]] = {}
        for finding in result.findings:
            for cat in finding.ctem_categories:
                ctem_map.setdefault(cat, []).append(finding.provider_name)

        if not ctem_map:
            return

        table = Table(
            title="CTEM Exposure Category Coverage",
            box=box.SIMPLE,
            border_style="magenta",
            header_style="bold magenta",
            show_lines=False,
        )
        table.add_column("CTEM Category", style="bold magenta", min_width=25)
        table.add_column("Providers Affected", style="white", min_width=40)
        table.add_column("Count", justify="right", min_width=6)

        for cat_id in sorted(ctem_map.keys()):
            providers = sorted(set(ctem_map[cat_id]))
            provider_str = _truncate(", ".join(providers), 60)
            table.add_row(cat_id, provider_str, str(len(providers)))

        self.console.print(table)
        self.console.print()

    def _print_evidence_detail(self, result: ScanResult) -> None:
        """Print detailed evidence for each finding (verbose mode).

        Args:
            result: The ScanResult containing findings and detection matches.
        """
        if not result.findings:
            return

        self.console.rule("[bold]Detection Evidence Detail[/bold]")
        self.console.print()

        for finding in result.sorted_findings:
            risk_colour = _RISK_COLOURS.get(finding.risk_level, "white")
            self.console.print(
                f"[{risk_colour}]\u25a0[/{risk_colour}] [{risk_colour}]{finding.title}[/{risk_colour}]"
                f"  [dim]({finding.finding_id})[/dim]"
            )

            matches_to_show = finding.detection_matches[: self.config.max_evidence_lines]
            for match in matches_to_show:
                line_display = _truncate(match.line_content, self.config.truncate_line_length)
                self.console.print(
                    f"  [dim]{match.file_path}:{match.line_number}[/dim]  "
                    f"[white]{line_display}[/white]  "
                    f"[dim]({match.detection_type.value})[/dim]"
                )

            remaining = len(finding.detection_matches) - len(matches_to_show)
            if remaining > 0:
                self.console.print(
                    f"  [dim]  ... and {remaining} more matches[/dim]"
                )
            self.console.print()

    def _print_footer(self, result: ScanResult) -> None:
        """Print the report footer with scan completion information.

        Args:
            result: The ScanResult for footer metadata.
        """
        meta = result.metadata
        summary = result.summary
        risk_colour = _RISK_COLOURS.get(summary.overall_risk_level, "white")

        footer_lines = [
            f"[dim]Scan completed. Overall risk: [/dim]"
            f"[{risk_colour}]{summary.overall_risk_level.value.upper()} "
            f"({summary.overall_risk_score:.1f}/10)[/{risk_colour}]",
        ]
        if meta.completed_at:
            footer_lines.append(
                f"[dim]Completed at: {meta.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
            )

        self.console.print(
            Panel(
                "\n".join(footer_lines),
                border_style="dim",
                padding=(0, 1),
            )
        )


# ---------------------------------------------------------------------------
# JSONReporter
# ---------------------------------------------------------------------------


class JSONReporter:
    """Serializes a ScanResult to a formatted JSON inventory export.

    The JSON output follows the ScanResult schema and includes all
    metadata, assets, findings, compliance gaps, and detection matches.

    Attributes:
        config: Report configuration options controlling JSON formatting.
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize the JSONReporter.

        Args:
            config: Report configuration options.
        """
        self.config: ReportConfig = config or ReportConfig()

    def generate(self, scan_result: ScanResult) -> str:
        """Generate a JSON string representation of the scan result.

        Args:
            scan_result: The ScanResult to serialize.

        Returns:
            A formatted JSON string with the complete scan inventory.
        """
        data = scan_result.to_dict()
        return json.dumps(data, indent=self.config.indent, default=str, ensure_ascii=False)

    def write(self, scan_result: ScanResult, output_path: Path) -> None:
        """Write the JSON report to a file.

        Args:
            scan_result: The ScanResult to serialize.
            output_path: Destination file path for the JSON output.

        Raises:
            OSError: If the file cannot be written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_content = self.generate(scan_result)
        output_path.write_text(json_content, encoding="utf-8")

    def write_stream(self, scan_result: ScanResult, stream: IO[str]) -> None:
        """Write the JSON report to an open IO stream.

        Args:
            scan_result: The ScanResult to serialize.
            stream: An open text IO stream to write to.
        """
        stream.write(self.generate(scan_result))
        stream.write("\n")


# ---------------------------------------------------------------------------
# MarkdownReporter
# ---------------------------------------------------------------------------


class MarkdownReporter:
    """Generates a Markdown audit report from a ScanResult.

    The Markdown report is structured for use in security audits and
    AI governance documentation, covering:
    - Executive summary.
    - Detected AI providers table.
    - Risk findings with full detail.
    - Compliance gaps and remediation guidance.
    - NIST AI RMF and CTEM framework mappings.
    - Appendix with raw detection evidence.

    Attributes:
        config: Report configuration options.
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize the MarkdownReporter.

        Args:
            config: Report configuration options.
        """
        self.config: ReportConfig = config or ReportConfig()

    def generate(self, scan_result: ScanResult) -> str:
        """Generate a Markdown string for the full audit report.

        Args:
            scan_result: The ScanResult to document.

        Returns:
            A Markdown-formatted string representing the complete audit report.
        """
        sections: list[str] = []

        sections.append(self._render_title(scan_result))
        sections.append(self._render_toc())
        sections.append(self._render_metadata(scan_result))
        sections.append(self._render_executive_summary(scan_result))
        sections.append(self._render_providers_table(scan_result))
        sections.append(self._render_findings_detail(scan_result))

        if self.config.show_compliance_gaps:
            sections.append(self._render_compliance_gaps(scan_result))

        if self.config.show_nist_mapping:
            sections.append(self._render_nist_mapping(scan_result))

        if self.config.show_ctem_mapping:
            sections.append(self._render_ctem_mapping(scan_result))

        if self.config.show_affected_files:
            sections.append(self._render_affected_files(scan_result))

        sections.append(self._render_appendix_evidence(scan_result))
        sections.append(self._render_footer(scan_result))

        return "\n\n".join(filter(None, sections)) + "\n"

    def write(self, scan_result: ScanResult, output_path: Path) -> None:
        """Write the Markdown report to a file.

        Args:
            scan_result: The ScanResult to document.
            output_path: Destination file path for the Markdown output.

        Raises:
            OSError: If the file cannot be written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = self.generate(scan_result)
        output_path.write_text(md_content, encoding="utf-8")

    def write_stream(self, scan_result: ScanResult, stream: IO[str]) -> None:
        """Write the Markdown report to an open IO stream.

        Args:
            scan_result: The ScanResult to document.
            stream: An open text IO stream to write to.
        """
        stream.write(self.generate(scan_result))

    # ------------------------------------------------------------------
    # Private section renderers
    # ------------------------------------------------------------------

    def _render_title(self, result: ScanResult) -> str:
        """Render the report title and scan timestamp.

        Args:
            result: The ScanResult for title metadata.

        Returns:
            Markdown string for the title section.
        """
        meta = result.metadata
        ts = meta.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "# AI Risk Inventory Report",
            "",
            f"> **Scan ID:** `{meta.scan_id}`  ",
            f"> **Target Path:** `{meta.scan_path}`  ",
            f"> **Scanner Version:** `{meta.scanner_version}`  ",
            f"> **Generated At:** {ts}  ",
        ]
        if meta.completed_at:
            duration = meta.duration_seconds or 0.0
            lines.append(f"> **Scan Duration:** {duration:.2f}s  ")
        return "\n".join(lines)

    def _render_toc(self) -> str:
        """Render the table of contents.

        Returns:
            Markdown string for the TOC.
        """
        toc_items = [
            "## Table of Contents",
            "",
            "1. [Scan Metadata](#scan-metadata)",
            "2. [Executive Summary](#executive-summary)",
            "3. [Detected AI Providers](#detected-ai-providers)",
            "4. [Risk Findings](#risk-findings)",
        ]
        if self.config.show_compliance_gaps:
            toc_items.append("5. [Compliance Gaps](#compliance-gaps)")
        if self.config.show_nist_mapping:
            toc_items.append("6. [NIST AI RMF Mapping](#nist-ai-rmf-mapping)")
        if self.config.show_ctem_mapping:
            toc_items.append("7. [CTEM Exposure Mapping](#ctem-exposure-mapping)")
        if self.config.show_affected_files:
            toc_items.append("8. [Affected Files](#affected-files)")
        toc_items.append("9. [Detection Evidence Appendix](#detection-evidence-appendix)")
        return "\n".join(toc_items)

    def _render_metadata(self, result: ScanResult) -> str:
        """Render the scan metadata section.

        Args:
            result: The ScanResult for metadata.

        Returns:
            Markdown string for the metadata section.
        """
        meta = result.metadata
        lines = [
            "## Scan Metadata",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Scan ID | `{meta.scan_id}` |",
            f"| Target Path | `{meta.scan_path}` |",
            f"| Scanner Version | `{meta.scanner_version}` |",
            f"| Started At | {meta.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        ]
        if meta.completed_at:
            lines.append(
                f"| Completed At | {meta.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |"
            )
            lines.append(f"| Duration | {(meta.duration_seconds or 0.0):.2f}s |")
        lines += [
            f"| Files Scanned | {meta.total_files_scanned} |",
            f"| Files Skipped | {meta.total_files_skipped} |",
        ]
        return "\n".join(lines)

    def _render_executive_summary(self, result: ScanResult) -> str:
        """Render the executive summary section.

        Args:
            result: The ScanResult with computed summary.

        Returns:
            Markdown string for the executive summary.
        """
        summary = result.summary
        risk_emoji = {
            RiskLevel.CRITICAL: "\U0001f534",  # 🔴
            RiskLevel.HIGH: "\U0001f7e0",      # 🟠
            RiskLevel.MEDIUM: "\U0001f7e1",    # 🟡
            RiskLevel.LOW: "\U0001f7e2",       # 🟢
            RiskLevel.INFO: "\u26aa",           # ⚪
        }.get(summary.overall_risk_level, "")

        lines = [
            "## Executive Summary",
            "",
            f"**Overall Risk Level:** {risk_emoji} `{summary.overall_risk_level.value.upper()}`  ",
            f"**Overall Risk Score:** `{summary.overall_risk_score:.1f} / 10.0`",
            "",
            "### Key Metrics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| AI Providers Detected | **{summary.total_providers_detected}** |",
            f"| Risk Findings | **{summary.total_findings}** |",
            f"| Compliance Gaps | **{summary.total_compliance_gaps}** |",
            f"| Total Detection Matches | **{summary.total_detection_matches}** |",
            f"| Critical Findings | **{summary.critical_findings}** |",
            f"| High Findings | **{summary.high_findings}** |",
            f"| Medium Findings | **{summary.medium_findings}** |",
            f"| Low Findings | **{summary.low_findings}** |",
        ]

        if summary.providers_with_credentials:
            lines += [
                "",
                "### \u26a0\ufe0f Critical: Hardcoded Credentials Detected",
                "",
                "The following providers have hardcoded API keys or tokens in source files:",
                "",
            ]
            for pid in summary.providers_with_credentials:
                lines.append(f"- `{pid}`")

        if summary.providers_with_data_residency_risk:
            lines += [
                "",
                "### \U0001f30d Data Residency Risk",
                "",
                "The following providers may transfer data outside your jurisdiction:",
                "",
            ]
            for pid in summary.providers_with_data_residency_risk:
                lines.append(f"- `{pid}`")

        if summary.providers_with_vendor_lock_in:
            lines += [
                "",
                "### \U0001f512 Vendor Lock-In Risk",
                "",
                "The following providers use proprietary APIs creating lock-in dependency:",
                "",
            ]
            for pid in summary.providers_with_vendor_lock_in:
                lines.append(f"- `{pid}`")

        return "\n".join(lines)

    def _render_providers_table(self, result: ScanResult) -> str:
        """Render the detected AI providers table.

        Args:
            result: The ScanResult containing asset data.

        Returns:
            Markdown string for the providers section.
        """
        if not result.assets:
            return "## Detected AI Providers\n\n_No AI providers detected._"

        sorted_assets = sorted(
            result.assets.values(),
            key=lambda a: (-a.baseline_risk_level.numeric_score, a.provider_name),
        )

        lines = [
            "## Detected AI Providers",
            "",
            "| Provider | Category | Risk Level | Matches | Files | Lock-in | Data Risk | Credentials |",
            "|----------|----------|------------|---------|-------|---------|-----------|-------------|",
        ]

        for asset in sorted_assets:
            category_label = _CATEGORY_LABELS.get(asset.category.value, asset.category.value)
            risk_label = asset.baseline_risk_level.value.upper()
            lock_in = "Yes" if asset.vendor_lock_in else "No"
            data_risk = "Yes" if asset.data_residency_concern else "No"
            credentials = "**YES**" if asset.has_credential_exposure else "No"

            lines.append(
                f"| {asset.provider_name} | {category_label} | {risk_label} "
                f"| {asset.total_matches} | {len(asset.files_detected)} "
                f"| {lock_in} | {data_risk} | {credentials} |"
            )

        return "\n".join(lines)

    def _render_findings_detail(self, result: ScanResult) -> str:
        """Render detailed information for each risk finding.

        Args:
            result: The ScanResult containing findings.

        Returns:
            Markdown string for the findings detail section.
        """
        if not result.findings:
            return "## Risk Findings\n\n_No risk findings generated._"

        sections = ["## Risk Findings", ""]

        for idx, finding in enumerate(result.sorted_findings, start=1):
            risk_label = finding.risk_level.value.upper()
            score_bar = _severity_bar(finding.risk_score)

            sections.append(f"### {idx}. {finding.title}")
            sections.append("")
            sections.append(
                f"- **Severity:** `{risk_label}` &nbsp; Score: `{score_bar}`"
            )
            sections.append(f"- **Finding ID:** `{finding.finding_id}`")
            sections.append(f"- **Provider:** {finding.provider_name} (`{finding.provider_id}`)")
            sections.append(
                f"- **Vendor Lock-in:** {'Yes' if finding.vendor_lock_in else 'No'}  "
                f"**Data Residency Risk:** {'Yes' if finding.data_residency_concern else 'No'}  "
                f"**Hardcoded Credentials:** {'**YES**' if finding.has_hardcoded_credentials else 'No'}"
            )
            sections.append(f"- **Evidence Count:** {finding.evidence_count}")

            if finding.nist_functions:
                sections.append(
                    f"- **NIST AI RMF:** {', '.join(f'`{f}`' for f in finding.nist_functions)}"
                )
            if finding.ctem_categories:
                sections.append(
                    f"- **CTEM Categories:** {', '.join(f'`{c}`' for c in finding.ctem_categories)}"
                )

            sections.append("")
            sections.append("**Description:**")
            sections.append("")
            sections.append(finding.description)

            if finding.affected_files and self.config.show_affected_files:
                sections.append("")
                sections.append("**Affected Files:**")
                sections.append("")
                for fp in sorted(str(p) for p in finding.affected_files):
                    sections.append(f"- `{fp}`")

            if finding.compliance_gaps and self.config.show_compliance_gaps:
                sections.append("")
                sections.append("**Compliance Gaps:**")
                sections.append("")
                for gap in finding.compliance_gaps:
                    gap_risk = gap.risk_level.value.upper()
                    sections.append(f"- **[{gap_risk}] {gap.title}**")
                    sections.append(f"  - {gap.description}")
                    if gap.remediation:
                        sections.append(f"  - *Remediation:* {gap.remediation}")

            sections.append("")
            sections.append("---")
            sections.append("")

        return "\n".join(sections)

    def _render_compliance_gaps(self, result: ScanResult) -> str:
        """Render the consolidated compliance gaps section.

        Args:
            result: The ScanResult containing findings with gaps.

        Returns:
            Markdown string for the compliance gaps section.
        """
        all_gaps: list[tuple[str, str, ComplianceGap]] = []
        for finding in result.sorted_findings:
            for gap in finding.compliance_gaps:
                all_gaps.append((finding.provider_name, finding.provider_id, gap))

        if not all_gaps:
            return ""

        # Sort by risk level descending
        level_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
            RiskLevel.INFO: 4,
        }
        all_gaps.sort(key=lambda x: level_order.get(x[2].risk_level, 5))

        lines = [
            "## Compliance Gaps",
            "",
            "| Provider | Gap Title | Severity | NIST Functions | CTEM Categories |",
            "|----------|-----------|----------|----------------|-----------------|",
        ]

        for provider_name, provider_id, gap in all_gaps:
            risk_label = gap.risk_level.value.upper()
            nist_str = ", ".join(f"`{f}`" for f in gap.nist_functions) if gap.nist_functions else "-"
            ctem_str = ", ".join(f"`{c}`" for c in gap.ctem_categories) if gap.ctem_categories else "-"
            lines.append(
                f"| {provider_name} | {gap.title} | {risk_label} | {nist_str} | {ctem_str} |"
            )

        lines.append("")
        lines.append("### Remediation Guidance")
        lines.append("")

        seen_gaps: set[str] = set()
        for _, _, gap in all_gaps:
            if gap.gap_id in seen_gaps:
                continue
            seen_gaps.add(gap.gap_id)
            if gap.remediation:
                lines.append(f"**{gap.title}**")
                lines.append("")
                lines.append(gap.remediation)
                lines.append("")

        return "\n".join(lines)

    def _render_nist_mapping(self, result: ScanResult) -> str:
        """Render the NIST AI RMF function mapping section.

        Args:
            result: The ScanResult containing findings.

        Returns:
            Markdown string for the NIST mapping section.
        """
        if not result.findings:
            return ""

        nist_map: dict[str, list[str]] = {}
        for finding in result.findings:
            for func in finding.nist_functions:
                nist_map.setdefault(func, []).append(finding.provider_name)

        if not nist_map:
            return ""

        lines = [
            "## NIST AI RMF Mapping",
            "",
            "This section maps risk findings to the NIST AI Risk Management Framework "
            "(AI RMF) core functions.",
            "",
            "| Function ID | Providers Affected | Finding Count |",
            "|-------------|-------------------|---------------|",
        ]

        for func_id in sorted(nist_map.keys()):
            providers = sorted(set(nist_map[func_id]))
            providers_str = ", ".join(providers)
            lines.append(
                f"| `{func_id}` | {providers_str} | {len(providers)} |"
            )

        lines += [
            "",
            "### NIST AI RMF Function Reference",
            "",
            "| Function ID | Core Function | Description |",
            "|-------------|---------------|-------------|",
            "| `GOVERN-1` | Govern | Establish organizational policies, roles, and accountability for AI risk management. |",
            "| `GOVERN-2` | Govern | Foster organizational culture and workforce readiness for responsible AI. |",
            "| `MAP-1` | Map | Identify and categorize the AI system's context, purpose, and intended use. |",
            "| `MAP-3` | Map | Classify AI system risks based on potential harms and deployment context. |",
            "| `MAP-4` | Map | Identify and assess risks from third-party AI components and supply chain. |",
            "| `MAP-5` | Map | Identify potential harms to individuals, groups, or society from AI system outputs. |",
            "| `MEASURE-1` | Measure | Define and collect metrics to evaluate AI risk exposure. |",
            "| `MEASURE-2` | Measure | Evaluate AI system trustworthiness characteristics including bias, robustness, and privacy. |",
            "| `MANAGE-1` | Manage | Apply risk treatment strategies including mitigation, transfer, and acceptance. |",
            "| `MANAGE-2` | Manage | Implement technical and governance controls for identified AI risks. |",
            "| `MANAGE-3` | Manage | Continuously monitor AI risks and review effectiveness of controls. |",
        ]

        return "\n".join(lines)

    def _render_ctem_mapping(self, result: ScanResult) -> str:
        """Render the CTEM exposure category mapping section.

        Args:
            result: The ScanResult containing findings.

        Returns:
            Markdown string for the CTEM mapping section.
        """
        if not result.findings:
            return ""

        ctem_map: dict[str, list[str]] = {}
        for finding in result.findings:
            for cat in finding.ctem_categories:
                ctem_map.setdefault(cat, []).append(finding.provider_name)

        if not ctem_map:
            return ""

        lines = [
            "## CTEM Exposure Mapping",
            "",
            "This section maps risk findings to Continuous Threat Exposure Management "
            "(CTEM) exposure categories.",
            "",
            "| CTEM Category | Providers Affected | Count |",
            "|---------------|--------------------|-------|",
        ]

        for cat_id in sorted(ctem_map.keys()):
            providers = sorted(set(ctem_map[cat_id]))
            providers_str = ", ".join(providers)
            lines.append(
                f"| `{cat_id}` | {providers_str} | {len(providers)} |"
            )

        lines += [
            "",
            "### CTEM Category Reference",
            "",
            "| Category | Description |",
            "|----------|-------------|",
            "| `external_exposure` | AI service APIs exposed to or called from external networks. |",
            "| `data_exfiltration` | Sensitive data transmitted to third-party AI provider infrastructure. |",
            "| `third_party_dependency` | Critical dependency on external AI provider availability and reliability. |",
            "| `cloud_dependency` | Dependency on cloud-platform-specific AI services creating lock-in risk. |",
            "| `supply_chain` | Risk from open-source models or components with unclear provenance. |",
            "| `identity_access` | Inadequate identity and access management for AI service credentials. |",
            "| `open_source_risk` | Risks associated with open-source AI models including licensing and safety. |",
            "| `biometric_data` | Processing of biometric data (voice, face, fingerprint) by AI services. |",
        ]

        return "\n".join(lines)

    def _render_affected_files(self, result: ScanResult) -> str:
        """Render a consolidated list of all files containing AI references.

        Args:
            result: The ScanResult containing asset data.

        Returns:
            Markdown string for the affected files section.
        """
        all_files = result.all_affected_files
        if not all_files:
            return ""

        lines = [
            "## Affected Files",
            "",
            f"A total of **{len(all_files)}** file(s) contain AI provider references:",
            "",
        ]

        for fp in sorted(str(p) for p in all_files):
            lines.append(f"- `{fp}`")

        return "\n".join(lines)

    def _render_appendix_evidence(self, result: ScanResult) -> str:
        """Render the detection evidence appendix.

        Includes raw pattern match details for each finding, limited by
        the max_evidence_lines configuration.

        Args:
            result: The ScanResult containing detection evidence.

        Returns:
            Markdown string for the appendix section.
        """
        if not result.findings:
            return ""

        lines = [
            "## Detection Evidence Appendix",
            "",
            "Raw detection evidence for each finding. Useful for manual verification.",
        ]

        for finding in result.sorted_findings:
            if not finding.detection_matches:
                continue

            lines.append("")
            lines.append(f"### {finding.title}")
            lines.append("")

            matches_to_show = finding.detection_matches[: self.config.max_evidence_lines]

            lines.append("| File | Line | Type | Content |")
            lines.append("|------|------|------|---------|")

            for match in matches_to_show:
                file_display = str(match.file_path)
                content_display = _truncate(
                    match.line_content.strip().replace("|", "&#124;"),
                    self.config.truncate_line_length,
                )
                lines.append(
                    f"| `{file_display}` | {match.line_number} "
                    f"| `{match.detection_type.value}` | `{content_display}` |"
                )

            remaining = len(finding.detection_matches) - len(matches_to_show)
            if remaining > 0:
                lines.append("")
                lines.append(
                    f"_... and {remaining} more match(es). Export to JSON for full details._"
                )

        return "\n".join(lines)

    def _render_footer(self, result: ScanResult) -> str:
        """Render the report footer.

        Args:
            result: The ScanResult for footer metadata.

        Returns:
            Markdown string for the footer.
        """
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "---",
            "",
            "_This report was generated by the "
            "[AI Risk Inventory Scanner](https://github.com/ai-risk-scanner/ai-risk-scanner). "
            "Review findings with your security and compliance team before making "
            "governance decisions._",
            "",
            f"_Report generated: {now}_",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporter facade
# ---------------------------------------------------------------------------


class Reporter:
    """Facade class providing a unified interface for all report formats.

    The Reporter delegates to ConsoleReporter, JSONReporter, and
    MarkdownReporter based on the requested output format. It provides
    convenience methods for the most common reporting workflows.

    Attributes:
        config: Shared report configuration used across all reporters.
        console_reporter: The ConsoleReporter instance.
        json_reporter: The JSONReporter instance.
        markdown_reporter: The MarkdownReporter instance.
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the Reporter with shared configuration.

        Args:
            config: Report configuration options. Uses defaults if not provided.
            console: Optional pre-configured rich Console for terminal output.
        """
        self.config: ReportConfig = config or ReportConfig()
        self.console_reporter: ConsoleReporter = ConsoleReporter(
            config=self.config,
            console=console,
        )
        self.json_reporter: JSONReporter = JSONReporter(config=self.config)
        self.markdown_reporter: MarkdownReporter = MarkdownReporter(config=self.config)

    @property
    def console(self) -> Console:
        """Return the rich Console instance used for terminal output."""
        return self.console_reporter.console

    def print_console(self, scan_result: ScanResult) -> None:
        """Print the rich terminal report for a ScanResult.

        Args:
            scan_result: The ScanResult to render to the terminal.
        """
        self.console_reporter.print(scan_result)

    def generate_json(self, scan_result: ScanResult) -> str:
        """Generate a JSON string for the scan result.

        Args:
            scan_result: The ScanResult to serialize.

        Returns:
            Formatted JSON string.
        """
        return self.json_reporter.generate(scan_result)

    def write_json(self, scan_result: ScanResult, output_path: Path) -> None:
        """Write the JSON report to a file.

        Args:
            scan_result: The ScanResult to serialize.
            output_path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        self.json_reporter.write(scan_result, output_path)

    def generate_markdown(self, scan_result: ScanResult) -> str:
        """Generate a Markdown string for the audit report.

        Args:
            scan_result: The ScanResult to document.

        Returns:
            Formatted Markdown string.
        """
        return self.markdown_reporter.generate(scan_result)

    def write_markdown(self, scan_result: ScanResult, output_path: Path) -> None:
        """Write the Markdown audit report to a file.

        Args:
            scan_result: The ScanResult to document.
            output_path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        self.markdown_reporter.write(scan_result, output_path)

    def report(
        self,
        scan_result: ScanResult,
        format: str = "console",
        output_path: Path | None = None,
    ) -> str | None:
        """Generate a report in the specified format.

        This is a unified entry point that dispatches to the appropriate
        reporter based on the requested format.

        Args:
            scan_result: The ScanResult to report on.
            format: Output format — one of 'console', 'json', or 'markdown'.
            output_path: Optional file path to write the output to.
                         If None, console format prints to terminal and
                         json/markdown formats return the string.

        Returns:
            The generated report string for 'json' and 'markdown' formats,
            or None for 'console' format (output goes directly to terminal).

        Raises:
            ValueError: If an unsupported format is specified.
            OSError: If output_path is specified and the file cannot be written.
        """
        fmt = format.lower().strip()

        if fmt == "console":
            self.print_console(scan_result)
            return None

        elif fmt == "json":
            content = self.generate_json(scan_result)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
            return content

        elif fmt in ("markdown", "md"):
            content = self.generate_markdown(scan_result)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
            return content

        else:
            raise ValueError(
                f"Unsupported report format: {format!r}. "
                "Choose one of: 'console', 'json', 'markdown'."
            )

    def __repr__(self) -> str:
        return (
            f"Reporter("
            f"show_evidence={self.config.show_evidence}, "
            f"verbose={self.config.verbose})"
        )
