"""AI Risk Inventory Scanner package.

This package provides tools to recursively scan codebases and configuration
files to detect integrated AI services, APIs, and models. It generates
structured risk reports flagging data sensitivity concerns, vendor lock-in
risks, and compliance gaps against NIST AI RMF and CTEM principles.

Typical usage::

    from ai_risk_scanner import __version__
    print(__version__)

Or via the CLI entry point::

    ai-risk-scanner scan ./my_project --format json --output report.json
"""

__version__ = "0.1.0"
__author__ = "AI Risk Scanner Contributors"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
