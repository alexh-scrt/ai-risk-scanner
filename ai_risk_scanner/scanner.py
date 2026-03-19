"""File scanner and recursive directory walker for the AI Risk Inventory Scanner.

This module implements the core file discovery and scanning pipeline that
recursively walks a directory tree, identifies supported file types, delegates
content analysis to the DetectionEngine, and feeds results into the risk engine
to produce a complete ScanResult.

Key classes:
    - ScanConfig: Configuration options controlling scanner behavior.
    - FileWalker: Recursive directory walker that discovers scannable files.
    - Scanner: Orchestrates file discovery, detection, and result aggregation.

Key functions:
    - scan_path: Top-level convenience function to run a full scan.

Usage example::

    from pathlib import Path
    from ai_risk_scanner.scanner import scan_path

    result = scan_path(
        target_path=Path('./my_project'),
        exclude_dirs={'node_modules', '.git', '__pycache__'},
        max_file_size_mb=5,
    )
    print(result.summary)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ai_risk_scanner.detectors import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILENAMES,
    DetectionEngine,
    FileDetectionResult,
)
from ai_risk_scanner.inventory import (
    AIAsset,
    ScanResult,
    make_scan_result,
)
from ai_risk_scanner.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

# Default set of directory names to always exclude from scanning
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".idea",
    ".vscode",
    "node_modules",
    ".npm",
    ".yarn",
    "bower_components",
    "dist",
    "build",
    ".build",
    "target",
    "out",
    ".next",
    ".nuxt",
    "coverage",
    ".coverage",
    "htmlcov",
    ".eggs",
    "*.egg-info",
    ".DS_Store",
    "__MACOSX",
    "Thumbs.db",
    "logs",
    "log",
    "tmp",
    "temp",
    ".terraform",
    ".serverless",
    ".webpack",
    "vendor",
    "third_party",
    "site-packages",
    "lib",
    "libs",
})

# Default set of file name patterns to always skip
DEFAULT_EXCLUDED_FILENAMES: frozenset[str] = frozenset({
    ".DS_Store",
    "Thumbs.db",
    ".gitkeep",
    ".gitattributes",
    "CHANGELOG.md",
    "CHANGES.md",
    "HISTORY.md",
    "AUTHORS",
    "CONTRIBUTORS",
    "CODEOWNERS",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
    "NOTICE.md",
})

# Default maximum file size to scan in bytes (5 MB)
DEFAULT_MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024


@dataclass
class ScanConfig:
    """Configuration options controlling scanner behavior.

    Attributes:
        excluded_dirs: Set of directory names to skip during traversal.
            These are matched against directory base names (not full paths).
        excluded_filenames: Set of exact file names to skip.
        excluded_paths: Set of path prefixes (as strings) to exclude entirely.
        max_file_size_bytes: Maximum file size in bytes; larger files are skipped.
        follow_symlinks: Whether to follow symbolic links during traversal.
        encoding: Character encoding to use when reading files.
        include_hidden: Whether to scan hidden files and directories (dot-prefixed).
        extra_extensions: Additional file extensions to include beyond the defaults.
        providers_yaml_path: Optional path to a custom providers.yaml database.
        scan_id: Optional custom scan ID; auto-generated if not provided.
    """

    excluded_dirs: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_DIRS))
    excluded_filenames: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_FILENAMES))
    excluded_paths: set[str] = field(default_factory=set)
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    follow_symlinks: bool = False
    encoding: str = "utf-8"
    include_hidden: bool = False
    extra_extensions: set[str] = field(default_factory=set)
    providers_yaml_path: Path | None = None
    scan_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize excluded_dirs and extra_extensions to lowercase sets."""
        # Normalize extra extensions to have a leading dot and be lowercase
        normalized_extras: set[str] = set()
        for ext in self.extra_extensions:
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            normalized_extras.add(ext.lower())
        self.extra_extensions = normalized_extras

    @property
    def all_supported_extensions(self) -> frozenset[str]:
        """Return the union of default supported extensions and extra extensions."""
        return SUPPORTED_EXTENSIONS | frozenset(self.extra_extensions)


@dataclass
class WalkStats:
    """Statistics collected during a directory walk.

    Attributes:
        total_files_found: Total files encountered (before filtering).
        files_accepted: Files accepted for scanning.
        files_skipped_extension: Files skipped due to unsupported extension.
        files_skipped_hidden: Files skipped because they are hidden.
        files_skipped_excluded: Files skipped due to explicit exclusion rules.
        files_skipped_size: Files skipped because they exceed the size limit.
        dirs_traversed: Total directories entered during the walk.
        dirs_skipped: Directories excluded from traversal.
    """

    total_files_found: int = 0
    files_accepted: int = 0
    files_skipped_extension: int = 0
    files_skipped_hidden: int = 0
    files_skipped_excluded: int = 0
    files_skipped_size: int = 0
    dirs_traversed: int = 0
    dirs_skipped: int = 0

    @property
    def total_files_skipped(self) -> int:
        """Return the total count of skipped files across all skip reasons."""
        return (
            self.files_skipped_extension
            + self.files_skipped_hidden
            + self.files_skipped_excluded
            + self.files_skipped_size
        )

    def __repr__(self) -> str:
        return (
            f"WalkStats("
            f"found={self.total_files_found}, "
            f"accepted={self.files_accepted}, "
            f"skipped={self.total_files_skipped}, "
            f"dirs={self.dirs_traversed})"
        )


class FileWalker:
    """Recursive directory walker that discovers files eligible for scanning.

    The FileWalker applies filtering rules from ScanConfig to determine
    which files should be passed to the detection engine. It supports:
    - Extension-based filtering against the supported extension set.
    - Directory exclusion by name.
    - Hidden file/directory handling.
    - File size pre-filtering (stat-based, before reading content).
    - Explicit path prefix exclusions.

    Attributes:
        config: The ScanConfig controlling walk behavior.
        stats: WalkStats accumulated during the most recent walk.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        """Initialize the FileWalker with the given configuration.

        Args:
            config: Scanner configuration options. Uses defaults if not provided.
        """
        self.config: ScanConfig = config or ScanConfig()
        self.stats: WalkStats = WalkStats()

    def _is_excluded_dir(self, dir_path: Path) -> bool:
        """Determine if a directory should be excluded from traversal.

        Args:
            dir_path: The directory path to check.

        Returns:
            True if the directory should be excluded, False otherwise.
        """
        dir_name = dir_path.name

        # Check if directory name is in the exclusion set
        if dir_name in self.config.excluded_dirs:
            return True

        # Check hidden directories (unless include_hidden is set)
        if not self.config.include_hidden and dir_name.startswith("."):
            # Allow .env files but not .git, etc.
            if dir_name not in SUPPORTED_FILENAMES:
                return True

        # Check explicit path prefix exclusions
        str_path = str(dir_path)
        for excluded_prefix in self.config.excluded_paths:
            if str_path.startswith(excluded_prefix):
                return True

        return False

    def _is_supported_file(self, file_path: Path) -> bool:
        """Determine if a file should be included for scanning.

        Args:
            file_path: The file path to evaluate.

        Returns:
            True if the file should be scanned, False otherwise.
        """
        name = file_path.name

        # Check against excluded filenames
        if name in self.config.excluded_filenames:
            return False

        # Check explicit path prefix exclusions
        str_path = str(file_path)
        for excluded_prefix in self.config.excluded_paths:
            if str_path.startswith(excluded_prefix):
                return False

        # Check hidden files (unless include_hidden is set)
        if not self.config.include_hidden and name.startswith("."):
            # Allow known dotfiles like .env, .env.local, etc.
            if name not in SUPPORTED_FILENAMES:
                return False

        # Check by exact filename match
        if name in SUPPORTED_FILENAMES:
            return True

        # Check by file extension
        suffix = file_path.suffix.lower()
        if suffix in self.config.all_supported_extensions:
            return True

        return False

    def _is_within_size_limit(self, file_path: Path) -> bool:
        """Check if a file is within the configured size limit.

        Args:
            file_path: The file path to check.

        Returns:
            True if the file is within the size limit, False otherwise.
        """
        try:
            return file_path.stat().st_size <= self.config.max_file_size_bytes
        except OSError as exc:
            logger.debug("Cannot stat file %s: %s", file_path, exc)
            return False

    def walk(
        self,
        root_path: Path,
    ) -> Iterator[Path]:
        """Recursively walk a directory tree and yield eligible file paths.

        This generator resets and updates self.stats as it traverses the tree.
        It handles permission errors gracefully by logging and continuing.

        Args:
            root_path: The root directory path to start walking from.

        Yields:
            Path objects for each file that passes all filtering criteria.

        Raises:
            ValueError: If root_path does not exist or is not a directory.
        """
        if not root_path.exists():
            raise ValueError(f"Target path does not exist: {root_path}")

        # Handle single-file scanning
        if root_path.is_file():
            self.stats = WalkStats(total_files_found=1)
            if self._is_supported_file(root_path) and self._is_within_size_limit(root_path):
                self.stats.files_accepted += 1
                yield root_path
            else:
                self.stats.files_skipped_extension += 1
            return

        if not root_path.is_dir():
            raise ValueError(f"Target path is neither a file nor a directory: {root_path}")

        # Reset stats for a fresh walk
        self.stats = WalkStats()

        # Use os.walk for efficient directory traversal
        for dirpath_str, dirnames, filenames in os.walk(
            str(root_path),
            followlinks=self.config.follow_symlinks,
            onerror=self._handle_walk_error,
        ):
            dirpath = Path(dirpath_str)
            self.stats.dirs_traversed += 1

            # Filter out excluded directories in-place to prevent os.walk
            # from descending into them.
            dirs_to_remove: list[str] = []
            for dirname in dirnames:
                child_dir = dirpath / dirname
                if self._is_excluded_dir(child_dir):
                    dirs_to_remove.append(dirname)
                    self.stats.dirs_skipped += 1
                    logger.debug("Excluding directory: %s", child_dir)

            for dirname in dirs_to_remove:
                dirnames.remove(dirname)

            # Sort remaining dirnames for deterministic traversal order
            dirnames.sort()

            # Process files in this directory
            for filename in sorted(filenames):
                file_path = dirpath / filename
                self.stats.total_files_found += 1

                # Apply hidden file filter
                if not self.config.include_hidden and filename.startswith("."):
                    if filename not in SUPPORTED_FILENAMES:
                        self.stats.files_skipped_hidden += 1
                        logger.debug("Skipping hidden file: %s", file_path)
                        continue

                # Apply exclusion and extension filter
                if not self._is_supported_file(file_path):
                    self.stats.files_skipped_extension += 1
                    logger.debug("Skipping unsupported file: %s", file_path)
                    continue

                # Apply size filter
                if not self._is_within_size_limit(file_path):
                    self.stats.files_skipped_size += 1
                    logger.debug(
                        "Skipping oversized file: %s (%d bytes)",
                        file_path,
                        file_path.stat().st_size if file_path.exists() else -1,
                    )
                    continue

                self.stats.files_accepted += 1
                logger.debug("Accepted file for scanning: %s", file_path)
                yield file_path

    def _handle_walk_error(self, exc: OSError) -> None:
        """Handle errors encountered during os.walk traversal.

        Args:
            exc: The OSError raised during directory traversal.
        """
        logger.warning("Permission error during directory walk: %s", exc)

    def collect(self, root_path: Path) -> list[Path]:
        """Walk root_path and return all eligible file paths as a list.

        Args:
            root_path: The root directory path to scan.

        Returns:
            Sorted list of eligible file paths.
        """
        return list(self.walk(root_path))


class Scanner:
    """Orchestrates file discovery, AI provider detection, and result aggregation.

    The Scanner combines FileWalker, DetectionEngine, and RiskEngine into a
    single cohesive scan pipeline. It accepts a target path, walks the
    directory tree, runs the detection engine on each file, aggregates
    results by provider, generates risk findings, and returns a complete
    ScanResult.

    Attributes:
        config: The ScanConfig controlling all scan behavior.
        walker: The FileWalker used for directory traversal.
        engine: The DetectionEngine used for pattern matching.
        risk_engine: The RiskEngine used for risk scoring.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        """Initialize the Scanner with configuration and sub-components.

        Args:
            config: Scanner configuration. Uses defaults if not provided.
        """
        self.config: ScanConfig = config or ScanConfig()
        self.walker: FileWalker = FileWalker(config=self.config)
        self.engine: DetectionEngine = DetectionEngine(
            yaml_path=self.config.providers_yaml_path
        )
        self.risk_engine: RiskEngine = RiskEngine(
            yaml_path=self.config.providers_yaml_path
        )

    def scan(
        self,
        target_path: Path,
        scan_id: str | None = None,
    ) -> ScanResult:
        """Run a full scan of the target path and return a structured ScanResult.

        This method:
        1. Initializes a ScanResult with metadata.
        2. Walks the directory tree to discover eligible files.
        3. Runs the DetectionEngine on each file.
        4. Aggregates detected assets across all files.
        5. Runs the RiskEngine to generate risk findings.
        6. Computes the summary statistics.
        7. Finalizes metadata (completion time, file counts).

        Args:
            target_path: The root directory or file path to scan.
            scan_id: Optional custom scan identifier.

        Returns:
            A fully populated ScanResult.

        Raises:
            ValueError: If target_path does not exist.
        """
        resolved_path = target_path.resolve()
        effective_scan_id = scan_id or self.config.scan_id

        # Create the ScanResult with initial metadata
        result = make_scan_result(
            scan_path=resolved_path,
            scan_id=effective_scan_id,
        )
        result.metadata.supported_extensions = sorted(
            self.config.all_supported_extensions
        )
        result.metadata.excluded_paths = sorted(self.config.excluded_paths)

        logger.info("Starting scan of: %s", resolved_path)

        # Step 1: Discover all eligible files
        try:
            file_paths = self.walker.collect(resolved_path)
        except ValueError as exc:
            logger.error("Cannot scan path: %s", exc)
            result.metadata.completed_at = datetime.now(tz=timezone.utc)
            return result

        logger.info(
            "Discovered %d files to scan (%d skipped).",
            len(file_paths),
            self.walker.stats.total_files_skipped,
        )

        # Update file counts from walker stats
        result.metadata.total_files_scanned = self.walker.stats.files_accepted
        result.metadata.total_files_skipped = self.walker.stats.total_files_skipped

        if not file_paths:
            logger.info("No eligible files found in: %s", resolved_path)
            result.metadata.completed_at = datetime.now(tz=timezone.utc)
            result.compute_summary()
            return result

        # Step 2: Scan each file with the detection engine
        file_results: list[FileDetectionResult] = []
        for file_path in file_paths:
            logger.debug("Scanning: %s", file_path)
            file_result = self.engine.scan_file(
                file_path=file_path,
                encoding=self.config.encoding,
            )
            file_results.append(file_result)

            if file_result.scan_errors:
                for error in file_result.scan_errors:
                    logger.warning("Scan error in %s: %s", file_path, error)

        # Step 3: Aggregate detected assets across all files
        aggregated_assets: dict[str, AIAsset] = self.engine.aggregate_results(file_results)

        logger.info(
            "Detection complete: %d providers detected across %d files.",
            len(aggregated_assets),
            len(file_paths),
        )

        # Step 4: Add all aggregated assets to the result
        for asset in aggregated_assets.values():
            result.add_asset(asset)

        # Step 5: Generate risk findings via the risk engine
        findings = self.risk_engine.analyze(aggregated_assets)
        for finding in findings:
            result.add_finding(finding)

        logger.info(
            "Risk analysis complete: %d findings generated.",
            len(findings),
        )

        # Step 6: Compute summary statistics
        result.compute_summary()

        # Step 7: Finalize metadata
        result.metadata.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            "Scan complete in %.2fs: %d providers, %d findings, overall risk=%s",
            result.metadata.duration_seconds or 0.0,
            result.summary.total_providers_detected,
            result.summary.total_findings,
            result.summary.overall_risk_level.value,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"Scanner("
            f"providers={self.engine.provider_count}, "
            f"config={self.config!r})"
        )


def scan_path(
    target_path: Path | str,
    exclude_dirs: set[str] | None = None,
    exclude_paths: set[str] | None = None,
    max_file_size_mb: float = 5.0,
    follow_symlinks: bool = False,
    encoding: str = "utf-8",
    include_hidden: bool = False,
    extra_extensions: set[str] | None = None,
    providers_yaml_path: Path | None = None,
    scan_id: str | None = None,
) -> ScanResult:
    """Convenience function to run a full AI risk scan of a target path.

    This function creates a ScanConfig from the provided arguments,
    initializes a Scanner, and runs the scan pipeline.

    Args:
        target_path: Path to the directory or file to scan.
        exclude_dirs: Additional directory names to exclude (merged with defaults).
        exclude_paths: Set of path prefix strings to exclude entirely.
        max_file_size_mb: Maximum file size to scan in megabytes (default 5 MB).
        follow_symlinks: Whether to follow symbolic links during traversal.
        encoding: Character encoding for reading files.
        include_hidden: Whether to include hidden files and directories.
        extra_extensions: Additional file extensions to scan beyond defaults.
        providers_yaml_path: Optional path to a custom providers.yaml.
        scan_id: Optional custom scan ID.

    Returns:
        A fully populated ScanResult with assets, findings, and summary.

    Raises:
        ValueError: If target_path does not exist.

    Example::

        result = scan_path(
            target_path=Path('./my_project'),
            exclude_dirs={'test_fixtures', 'docs'},
            max_file_size_mb=2.0,
        )
        print(result.summary.overall_risk_level)
    """
    target = Path(target_path) if isinstance(target_path, str) else target_path

    # Build excluded dirs set (merge with defaults)
    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    if exclude_dirs:
        excluded_dirs.update(exclude_dirs)

    config = ScanConfig(
        excluded_dirs=excluded_dirs,
        excluded_paths=exclude_paths or set(),
        max_file_size_bytes=int(max_file_size_mb * 1024 * 1024),
        follow_symlinks=follow_symlinks,
        encoding=encoding,
        include_hidden=include_hidden,
        extra_extensions=extra_extensions or set(),
        providers_yaml_path=providers_yaml_path,
        scan_id=scan_id,
    )

    scanner = Scanner(config=config)
    return scanner.scan(target_path=target, scan_id=scan_id)


def discover_files(
    target_path: Path | str,
    config: ScanConfig | None = None,
) -> tuple[list[Path], WalkStats]:
    """Discover all eligible files in target_path without running detection.

    This function is useful for previewing which files would be scanned
    without actually running the full detection pipeline.

    Args:
        target_path: Path to the directory or file to walk.
        config: Optional scanner configuration. Uses defaults if not provided.

    Returns:
        A tuple of (list of eligible file paths, WalkStats from the walk).

    Raises:
        ValueError: If target_path does not exist.
    """
    target = Path(target_path) if isinstance(target_path, str) else target_path
    walker = FileWalker(config=config or ScanConfig())
    files = walker.collect(target)
    return files, walker.stats
