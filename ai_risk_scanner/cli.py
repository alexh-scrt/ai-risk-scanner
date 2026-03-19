"""Click-based CLI entry point for the AI Risk Inventory Scanner.

This module wires all scanner components together into a cohesive command-line
interface. It handles:
  - Scan path arguments and validation.
  - Output format flags (console, json, markdown).
  - Report file export to disk.
  - Graceful error display using rich.
  - Verbosity and configuration flags.

Entry point::

    ai-risk-scanner scan ./my_project --format json --output report.json
    ai-risk-scanner scan ./my_project --format markdown --output report.md
    ai-risk-scanner scan ./my_project --verbose
    ai-risk-scanner info

The CLI is registered in pyproject.toml as::

    [project.scripts]
    ai-risk-scanner = "ai_risk_scanner.cli:main"
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from ai_risk_scanner import __version__
from ai_risk_scanner.reporter import ReportConfig, Reporter
from ai_risk_scanner.scanner import ScanConfig, Scanner, DEFAULT_EXCLUDED_DIRS

# Shared rich Console for CLI-level messages (errors, warnings, progress)
_err_console = Console(stderr=True, highlight=False)
_out_console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Custom Click parameter types and validators
# ---------------------------------------------------------------------------


class ExistingPath(click.Path):
    """A Click path type that resolves the path and validates existence."""

    def convert(
        self,
        value: str,
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
    ) -> Path:
        """Convert and validate the path argument.

        Args:
            value: The raw string value from the CLI.
            param: The Click parameter.
            ctx: The Click context.

        Returns:
            A resolved Path object.
        """
        raw = super().convert(value, param, ctx)
        return Path(str(raw)).resolve()


# ---------------------------------------------------------------------------
# CLI group and global options
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(
    version=__version__,
    prog_name="ai-risk-scanner",
    message="%(prog)s version %(version)s",
)
@click.pass_context
def main(ctx: click.Context) -> None:
    """AI Risk Inventory Scanner.

    Recursively scans a codebase to detect integrated AI services, APIs,
    and models, then produces a structured risk report flagging data
    sensitivity concerns, vendor lock-in risks, and compliance gaps
    against NIST AI RMF and CTEM principles.

    \b
    Examples:
        ai-risk-scanner scan ./my_project
        ai-risk-scanner scan ./my_project --format json --output report.json
        ai-risk-scanner scan ./my_project --format markdown --output report.md
        ai-risk-scanner scan ./my_project --verbose --show-evidence
        ai-risk-scanner info
    """
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


@main.command(name="scan")
@click.argument(
    "path",
    default=".",
    type=ExistingPath(exists=True, file_okay=True, dir_okay=True, readable=True),
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["console", "json", "markdown", "md"], case_sensitive=False),
    default="console",
    show_default=True,
    help=(
        "Output format for the scan report. "
        "'console' prints a rich terminal summary, "
        "'json' produces a machine-readable JSON inventory, "
        "'markdown'/'md' produces a Markdown audit report."
    ),
)
@click.option(
    "--output", "-o",
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    default=None,
    help=(
        "File path to write the report output. "
        "If not provided for json/markdown formats, output is printed to stdout. "
        "For console format this option is ignored."
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output including detection evidence detail.",
)
@click.option(
    "--show-evidence",
    is_flag=True,
    default=False,
    help="Include raw detection evidence lines in the report output.",
)
@click.option(
    "--max-evidence",
    type=click.IntRange(min=1, max=50),
    default=3,
    show_default=True,
    help="Maximum number of evidence lines to show per finding.",
)
@click.option(
    "--exclude-dir",
    "exclude_dirs",
    multiple=True,
    metavar="DIR",
    help=(
        "Additional directory name to exclude from scanning. "
        "Can be specified multiple times. "
        "Merged with the built-in exclusion list."
    ),
)
@click.option(
    "--exclude-path",
    "exclude_paths",
    multiple=True,
    metavar="PATH",
    help=(
        "Path prefix to exclude from scanning. "
        "Can be specified multiple times."
    ),
)
@click.option(
    "--max-file-size",
    type=click.FloatRange(min=0.1, max=100.0),
    default=5.0,
    show_default=True,
    metavar="MB",
    help="Maximum file size in megabytes to scan. Files larger than this are skipped.",
)
@click.option(
    "--include-hidden",
    is_flag=True,
    default=False,
    help="Include hidden files and directories (dot-prefixed) in the scan.",
)
@click.option(
    "--follow-symlinks",
    is_flag=True,
    default=False,
    help="Follow symbolic links during directory traversal.",
)
@click.option(
    "--providers-yaml",
    "providers_yaml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=None,
    help=(
        "Path to a custom providers.yaml signature database. "
        "Defaults to the bundled providers.yaml."
    ),
)
@click.option(
    "--no-compliance-gaps",
    is_flag=True,
    default=False,
    help="Omit compliance gap details from the report output.",
)
@click.option(
    "--no-nist-mapping",
    is_flag=True,
    default=False,
    help="Omit the NIST AI RMF function mapping from the report.",
)
@click.option(
    "--no-ctem-mapping",
    is_flag=True,
    default=False,
    help="Omit the CTEM exposure category mapping from the report.",
)
@click.option(
    "--scan-id",
    type=str,
    default=None,
    help="Custom scan identifier. Auto-generated UUID if not provided.",
)
@click.option(
    "--encoding",
    type=str,
    default="utf-8",
    show_default=True,
    help="Character encoding used when reading source files.",
)
@click.pass_context
def scan(
    ctx: click.Context,
    path: Path,
    output_format: str,
    output_path: Optional[str],
    verbose: bool,
    show_evidence: bool,
    max_evidence: int,
    exclude_dirs: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    max_file_size: float,
    include_hidden: bool,
    follow_symlinks: bool,
    providers_yaml_path: Optional[str],
    no_compliance_gaps: bool,
    no_nist_mapping: bool,
    no_ctem_mapping: bool,
    scan_id: Optional[str],
    encoding: str,
) -> None:
    """Scan a directory or file for AI provider references and generate a risk report.

    PATH is the target directory or file to scan. Defaults to the current
    working directory if not specified.

    \b
    Examples:
        ai-risk-scanner scan .
        ai-risk-scanner scan ./src --format json --output report.json
        ai-risk-scanner scan /path/to/project --verbose --show-evidence
        ai-risk-scanner scan ./app --exclude-dir tests --exclude-dir docs
        ai-risk-scanner scan ./app --format markdown --output audit.md
    """
    # -----------------------------------------------------------------------
    # Resolve output path early so we can fail fast on bad destinations
    # -----------------------------------------------------------------------
    resolved_output: Optional[Path] = None
    if output_path is not None:
        resolved_output = Path(output_path).resolve()
        # Validate that the parent directory is writable
        try:
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _err_console.print(
                f"[bold red]Error:[/bold red] Cannot create output directory "
                f"'{resolved_output.parent}': {exc}"
            )
            ctx.exit(1)
            return

    # -----------------------------------------------------------------------
    # Print startup banner for console format
    # -----------------------------------------------------------------------
    if output_format.lower() == "console":
        _print_startup_banner(path, __version__)

    # -----------------------------------------------------------------------
    # Build ScanConfig
    # -----------------------------------------------------------------------
    excluded_dirs_set = set(DEFAULT_EXCLUDED_DIRS)
    excluded_dirs_set.update(exclude_dirs)

    excluded_paths_set: set[str] = set()
    excluded_paths_set.update(exclude_paths)

    providers_yaml: Optional[Path] = None
    if providers_yaml_path is not None:
        providers_yaml = Path(providers_yaml_path).resolve()

    scan_config = ScanConfig(
        excluded_dirs=excluded_dirs_set,
        excluded_paths=excluded_paths_set,
        max_file_size_bytes=int(max_file_size * 1024 * 1024),
        follow_symlinks=follow_symlinks,
        encoding=encoding,
        include_hidden=include_hidden,
        providers_yaml_path=providers_yaml,
        scan_id=scan_id,
    )

    # -----------------------------------------------------------------------
    # Build ReportConfig
    # -----------------------------------------------------------------------
    report_config = ReportConfig(
        show_evidence=show_evidence or verbose,
        max_evidence_lines=max_evidence,
        show_compliance_gaps=not no_compliance_gaps,
        show_nist_mapping=not no_nist_mapping,
        show_ctem_mapping=not no_ctem_mapping,
        show_affected_files=True,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # Initialise scanner and reporter
    # -----------------------------------------------------------------------
    try:
        scanner = Scanner(config=scan_config)
    except FileNotFoundError as exc:
        _err_console.print(
            f"[bold red]Error:[/bold red] Failed to initialise scanner: {exc}"
        )
        ctx.exit(1)
        return
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Unexpected error initialising scanner:[/bold red] "
            f"{type(exc).__name__}: {exc}"
        )
        if verbose:
            _err_console.print(traceback.format_exc())
        ctx.exit(1)
        return

    # Choose the console for the reporter:
    # - console format: use stdout console
    # - json/markdown with no output file: use stdout console (but reporter
    #   will write to stdout explicitly)
    # - json/markdown with output file: suppress console output from reporter
    reporter_console = _out_console

    reporter = Reporter(config=report_config, console=reporter_console)

    # -----------------------------------------------------------------------
    # Run the scan
    # -----------------------------------------------------------------------
    if output_format.lower() == "console":
        _err_console.print(
            f"[dim]Scanning:[/dim] [cyan]{path}[/cyan] ..."
        )

    try:
        scan_result = scanner.scan(target_path=path, scan_id=scan_id)
    except ValueError as exc:
        _err_console.print(
            f"[bold red]Error:[/bold red] Invalid scan target: {exc}"
        )
        ctx.exit(1)
        return
    except PermissionError as exc:
        _err_console.print(
            f"[bold red]Permission denied:[/bold red] {exc}"
        )
        ctx.exit(1)
        return
    except KeyboardInterrupt:
        _err_console.print("\n[yellow]Scan interrupted by user.[/yellow]")
        ctx.exit(130)
        return
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Unexpected error during scan:[/bold red] "
            f"{type(exc).__name__}: {exc}"
        )
        if verbose:
            _err_console.print(traceback.format_exc())
        ctx.exit(1)
        return

    # -----------------------------------------------------------------------
    # Generate and output the report
    # -----------------------------------------------------------------------
    fmt = output_format.lower().strip()

    try:
        if fmt == "console":
            reporter.print_console(scan_result)

        elif fmt == "json":
            json_content = reporter.generate_json(scan_result)
            if resolved_output is not None:
                resolved_output.write_text(json_content, encoding="utf-8")
                _err_console.print(
                    f"[green]JSON report written to:[/green] [cyan]{resolved_output}[/cyan]"
                )
            else:
                # Write to stdout
                click.echo(json_content)

        elif fmt in ("markdown", "md"):
            md_content = reporter.generate_markdown(scan_result)
            if resolved_output is not None:
                resolved_output.write_text(md_content, encoding="utf-8")
                _err_console.print(
                    f"[green]Markdown report written to:[/green] [cyan]{resolved_output}[/cyan]"
                )
            else:
                # Write to stdout
                click.echo(md_content)

    except OSError as exc:
        _err_console.print(
            f"[bold red]Error writing report:[/bold red] {exc}"
        )
        ctx.exit(1)
        return
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Unexpected error generating report:[/bold red] "
            f"{type(exc).__name__}: {exc}"
        )
        if verbose:
            _err_console.print(traceback.format_exc())
        ctx.exit(1)
        return

    # -----------------------------------------------------------------------
    # Print scan summary to stderr for non-console formats
    # -----------------------------------------------------------------------
    if fmt != "console":
        _print_compact_summary(scan_result)

    # -----------------------------------------------------------------------
    # Exit with a non-zero code if critical findings were found
    # -----------------------------------------------------------------------
    summary = scan_result.summary
    if summary.critical_findings > 0:
        # Exit code 2 indicates critical risk findings (useful for CI pipelines)
        ctx.exit(2)
        return


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


@main.command(name="info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display information about the scanner, loaded providers, and supported file types.

    Shows the scanner version, number of loaded provider signatures, supported
    file extensions, and a summary of the built-in exclusion rules.
    """
    try:
        from ai_risk_scanner.detectors import (
            DetectionEngine,
            SUPPORTED_EXTENSIONS,
            SUPPORTED_FILENAMES,
        )
        engine = DetectionEngine()
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Error loading scanner:[/bold red] {type(exc).__name__}: {exc}"
        )
        ctx.exit(1)
        return

    _out_console.print()
    _out_console.print(
        Panel(
            f"[bold white]AI Risk Inventory Scanner[/bold white]  v[cyan]{__version__}[/cyan]\n"
            "Recursively scans codebases to detect integrated AI services, APIs, and models.\n"
            "Generates risk reports mapped to NIST AI RMF and CTEM frameworks.",
            title="[bold blue]About[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    _out_console.print()

    # Provider summary table
    prov_table = Table(
        title="Loaded Provider Signatures",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold blue",
        show_lines=False,
    )
    prov_table.add_column("Provider ID", style="cyan", min_width=20)
    prov_table.add_column("Name", style="bold white", min_width=25)
    prov_table.add_column("Category", style="white", min_width=12)
    prov_table.add_column("Risk Level", style="yellow", min_width=12)
    prov_table.add_column("Lock-in", justify="center", min_width=8)
    prov_table.add_column("Data Risk", justify="center", min_width=9)

    risk_colours = {
        "critical": "bold red",
        "high": "bold yellow",
        "medium": "yellow",
        "low": "cyan",
    }

    for sig in sorted(engine.providers, key=lambda s: s.provider_name):
        colour = risk_colours.get(sig.risk_level, "white")
        from rich.text import Text as RText
        risk_text = RText(sig.risk_level.upper(), style=colour)
        lock_text = RText("✓" if sig.vendor_lock_in else "✗",
                          style="yellow" if sig.vendor_lock_in else "dim green")
        data_text = RText("✓" if sig.data_residency_concern else "✗",
                          style="yellow" if sig.data_residency_concern else "dim green")
        prov_table.add_row(
            sig.provider_id,
            sig.provider_name,
            sig.category,
            risk_text,
            lock_text,
            data_text,
        )

    _out_console.print(prov_table)
    _out_console.print()

    # Supported extensions
    ext_list = sorted(SUPPORTED_EXTENSIONS)
    ext_display = "  ".join(ext_list)
    _out_console.print(
        Panel(
            f"[dim]Supported file extensions ({len(ext_list)}):[/dim]\n"
            f"[cyan]{ext_display}[/cyan]",
            title="[bold]Supported File Types[/bold]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    _out_console.print()

    # Credential patterns
    _out_console.print(
        f"[dim]Loaded [bold white]{engine.provider_count}[/bold white] provider signatures "
        f"and [bold white]{len(engine.credential_patterns)}[/bold white] credential patterns.[/dim]"
    )
    _out_console.print(
        f"[dim]Signature database: [cyan]{engine.yaml_path}[/cyan][/dim]"
    )
    _out_console.print()


# ---------------------------------------------------------------------------
# providers command
# ---------------------------------------------------------------------------


@main.command(name="providers")
@click.option(
    "--category",
    type=click.Choice(
        ["llm", "image_gen", "speech", "embedding", "mlops", "cloud_ai"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter providers by category.",
)
@click.option(
    "--risk",
    "risk_level",
    type=click.Choice(["critical", "high", "medium", "low"], case_sensitive=False),
    default=None,
    help="Filter providers by baseline risk level.",
)
@click.option(
    "--providers-yaml",
    "providers_yaml_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a custom providers.yaml signature database.",
)
@click.pass_context
def providers(
    ctx: click.Context,
    category: Optional[str],
    risk_level: Optional[str],
    providers_yaml_path: Optional[str],
) -> None:
    """List all known AI providers in the signature database.

    Optionally filter by category or risk level.

    \b
    Examples:
        ai-risk-scanner providers
        ai-risk-scanner providers --category llm
        ai-risk-scanner providers --risk high
    """
    try:
        from ai_risk_scanner.detectors import DetectionEngine

        yaml_path = Path(providers_yaml_path).resolve() if providers_yaml_path else None
        engine = DetectionEngine(yaml_path=yaml_path)
    except FileNotFoundError as exc:
        _err_console.print(f"[bold red]Error:[/bold red] {exc}")
        ctx.exit(1)
        return
    except Exception as exc:  # noqa: BLE001
        _err_console.print(
            f"[bold red]Error loading providers:[/bold red] {type(exc).__name__}: {exc}"
        )
        ctx.exit(1)
        return

    # Apply filters
    filtered_sigs = engine.providers
    if category:
        filtered_sigs = [s for s in filtered_sigs if s.category.lower() == category.lower()]
    if risk_level:
        filtered_sigs = [s for s in filtered_sigs if s.risk_level.lower() == risk_level.lower()]

    if not filtered_sigs:
        _out_console.print("[dim]No providers match the specified filters.[/dim]")
        return

    table = Table(
        title=f"AI Provider Signatures ({len(filtered_sigs)} providers)",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold blue",
        show_lines=True,
    )
    table.add_column("ID", style="cyan", min_width=18)
    table.add_column("Name", style="bold white", min_width=22)
    table.add_column("Category", min_width=10)
    table.add_column("Risk", min_width=10)
    table.add_column("Lock-in", justify="center", min_width=7)
    table.add_column("Data Risk", justify="center", min_width=9)
    table.add_column("Patterns", justify="right", min_width=8)
    table.add_column("Docs", style="dim", min_width=20)

    risk_colours = {
        "critical": "bold red",
        "high": "bold yellow",
        "medium": "yellow",
        "low": "cyan",
    }

    for sig in sorted(filtered_sigs, key=lambda s: s.provider_name):
        from rich.text import Text as RText

        colour = risk_colours.get(sig.risk_level.lower(), "white")
        risk_text = RText(sig.risk_level.upper(), style=colour)
        lock_text = RText("✓" if sig.vendor_lock_in else "✗",
                          style="yellow" if sig.vendor_lock_in else "dim green")
        data_text = RText("✓" if sig.data_residency_concern else "✗",
                          style="yellow" if sig.data_residency_concern else "dim green")

        pattern_count = (
            len(sig.patterns.imports)
            + len(sig.patterns.env_vars)
            + len(sig.patterns.urls)
            + len(sig.patterns.model_names)
            + len(sig.patterns.sdk_calls)
        )

        docs_display = sig.docs_url[:35] if sig.docs_url else "-"

        table.add_row(
            sig.provider_id,
            sig.provider_name,
            sig.category,
            risk_text,
            lock_text,
            data_text,
            str(pattern_count),
            docs_display,
        )

    _out_console.print()
    _out_console.print(table)
    _out_console.print()


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _print_startup_banner(target_path: Path, version: str) -> None:
    """Print a compact startup banner before the scan begins.

    Args:
        target_path: The path being scanned.
        version: The scanner version string.
    """
    _out_console.print()
    _out_console.print(
        f"[bold blue]⚡ AI Risk Scanner[/bold blue] [dim]v{version}[/dim]  "
        f"→  [cyan]{target_path}[/cyan]"
    )
    _out_console.print()


def _print_compact_summary(scan_result: object) -> None:
    """Print a compact one-line scan summary to stderr after non-console reports.

    Args:
        scan_result: The ScanResult to summarise.
    """
    from ai_risk_scanner.inventory import RiskLevel

    summary = scan_result.summary  # type: ignore[attr-defined]
    meta = scan_result.metadata  # type: ignore[attr-defined]

    risk_colours = {
        RiskLevel.CRITICAL: "bold red",
        RiskLevel.HIGH: "bold yellow",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.LOW: "cyan",
        RiskLevel.INFO: "dim",
    }
    colour = risk_colours.get(summary.overall_risk_level, "white")

    duration_str = ""
    if meta.duration_seconds is not None:
        duration_str = f" in {meta.duration_seconds:.2f}s"

    _err_console.print(
        f"[dim]Scan complete{duration_str}:[/dim] "
        f"{meta.total_files_scanned} files • "
        f"{summary.total_providers_detected} providers • "
        f"{summary.total_findings} findings • "
        f"Risk: [{colour}]{summary.overall_risk_level.value.upper()} "
        f"({summary.overall_risk_score:.1f}/10)[/{colour}]"
    )
    if summary.critical_findings > 0:
        _err_console.print(
            f"[bold red]⚠  {summary.critical_findings} CRITICAL finding(s) detected.[/bold red]"
        )
    if summary.providers_with_credentials:
        _err_console.print(
            f"[bold red]⚠  Hardcoded credentials found in providers: "
            f"{', '.join(summary.providers_with_credentials)}[/bold red]"
        )


# ---------------------------------------------------------------------------
# Allow running as python -m ai_risk_scanner.cli
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
