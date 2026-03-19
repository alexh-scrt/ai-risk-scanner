# AI Risk Scanner 🔍

> **Instantly audit your codebase for AI service integrations, compliance gaps, and vendor risk — before your security team asks.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NIST AI RMF](https://img.shields.io/badge/Framework-NIST%20AI%20RMF-green.svg)](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%20Playbook.pdf)

---

## What It Does

`ai-risk-scanner` recursively walks your codebase and configuration files to detect every AI service, API, and model your project touches — from OpenAI and Anthropic to AWS Bedrock and Hugging Face. It scores each finding against the **NIST AI RMF** and **CTEM** frameworks, flagging hardcoded credentials, data residency concerns, and vendor lock-in risks. The result is a clear console summary plus exportable JSON or Markdown reports ready for security audits and AI governance reviews.

---

## Quick Start

**Install:**

```bash
pip install ai-risk-scanner
```

Or install from source:

```bash
git clone https://github.com/your-org/ai-risk-scanner.git
cd ai-risk-scanner
pip install -e .
```

**Run a scan:**

```bash
# Scan current directory and print a rich console summary
ai-risk-scanner scan .

# Export a JSON inventory report
ai-risk-scanner scan ./my_project --format json --output report.json

# Export a Markdown audit report
ai-risk-scanner scan ./my_project --format markdown --output report.md
```

That's it. A risk-scored inventory of every AI integration in your codebase will be waiting for you.

---

## Features

- **20+ AI Provider Signatures** — Detects OpenAI, Anthropic, Cohere, AWS Bedrock, Google Vertex AI, Azure OpenAI, Hugging Face, Mistral, Replicate, and more via an extensible YAML signature database.
- **NIST AI RMF & CTEM Mapping** — Every finding is mapped to NIST AI RMF Govern/Map/Measure/Manage functions and CTEM exposure categories with numeric severity scores.
- **Compliance Gap Analysis** — Automatically flags hardcoded credentials, missing rate-limit controls, data residency concerns, and vendor lock-in indicators.
- **Multi-Format Reporting** — Rich terminal summary table, machine-readable JSON inventory, and Markdown audit report suitable for governance documentation.
- **Broad File Type Support** — Scans Python, JavaScript, YAML, TOML, JSON, `.env`, and Dockerfiles with no extra configuration required.

---

## Usage Examples

### Console Summary (default)

```bash
ai-risk-scanner scan ./my_project
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
║              AI Risk Inventory Scanner — Scan Report               ║
╠══════════════════╦═══════════╦══════════════╦════════════╦══════════╣
║ Provider         ║ Category  ║ Risk Level   ║ CTEM       ║ Score    ║
╠══════════════════╬═══════════╬══════════════╬════════════╬══════════╣
║ OpenAI           ║ LLM       ║ HIGH         ║ EXPOSED    ║ 8.4      ║
║ Hugging Face     ║ MLOps     ║ MEDIUM       ║ MANAGED    ║ 5.1      ║
║ AWS Bedrock      ║ Cloud AI  ║ HIGH         ║ EXPOSED    ║ 7.9      ║
╚══════════════════╩═══════════╩══════════════╩════════════╩══════════╝

⚠  3 compliance gaps detected: hardcoded credentials (1), data residency (2)
✔  Scan complete — 42 files scanned in 0.8s
```

### JSON Export

```bash
ai-risk-scanner scan ./my_project --format json --output report.json
```

```json
{
  "metadata": {
    "scanned_path": "./my_project",
    "timestamp": "2024-11-15T10:32:00Z",
    "files_scanned": 42,
    "tool_version": "0.1.0"
  },
  "assets": [
    {
      "provider": "openai",
      "display_name": "OpenAI",
      "category": "llm",
      "risk_level": "high",
      "vendor_lock_in": true,
      "data_residency_concern": true,
      "nist_rmf_functions": ["GOVERN", "MAP", "MEASURE"],
      "ctem_category": "EXPOSED",
      "score": 8.4,
      "detected_in": ["app.py", ".env"]
    }
  ],
  "compliance_gaps": [
    {
      "type": "hardcoded_credential",
      "severity": "critical",
      "file": ".env",
      "description": "API key assigned to OPENAI_API_KEY appears hardcoded."
    }
  ]
}
```

### Markdown Audit Report

```bash
ai-risk-scanner scan ./my_project --format markdown --output report.md
```

Generates a structured Markdown document with an executive summary, per-provider risk tables, compliance gap details, and NIST AI RMF framework mapping — ready to paste into a Confluence page or GitHub issue.

### Verbose Mode

```bash
ai-risk-scanner scan ./my_project --verbose
```

Prints per-file detection details including matched patterns, line references, and detection type (import, env var, URL, model name, SDK call).

### Provider Database Info

```bash
# List all supported providers and their baseline risk levels
ai-risk-scanner providers

# Show tool version and detection engine info
ai-risk-scanner info
```

---

## Project Structure

```
ai-risk-scanner/
├── pyproject.toml                          # Project metadata, deps, CLI entry point
├── README.md
├── ai_risk_scanner/
│   ├── __init__.py                         # Package init and version export
│   ├── cli.py                              # Click CLI — arguments, flags, output routing
│   ├── scanner.py                          # Recursive file walker and scan orchestrator
│   ├── detectors.py                        # Pattern-matching detection engine
│   ├── inventory.py                        # Data models: assets, findings, scan results
│   ├── risk_engine.py                      # NIST AI RMF / CTEM risk scoring engine
│   ├── reporter.py                         # Console, JSON, and Markdown report generators
│   └── providers.yaml                      # AI provider signature database
└── tests/
    ├── __init__.py
    ├── test_detectors.py                   # Unit tests for pattern detection
    ├── test_risk_engine.py                 # Unit tests for risk scoring & framework mapping
    ├── test_scanner.py                     # Integration tests for the file walker
    ├── test_inventory.py                   # Unit tests for data models
    ├── test_reporter.py                    # Unit tests for report generation
    └── fixtures/
        └── sample_project/
            ├── app.py                      # Sample Python file with multi-provider AI usage
            └── .env                        # Sample env file with AI API key patterns
```

---

## Configuration

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--format` | `console` | Output format: `console`, `json`, or `markdown` |
| `--output` | _(stdout)_ | File path to write the report |
| `--verbose` / `-v` | `False` | Show per-file detection details |
| `--exclude` | _(none)_ | Glob pattern(s) for paths to skip (repeatable) |
| `--min-risk` | `low` | Minimum risk level to include: `low`, `medium`, `high`, `critical` |

**Examples:**

```bash
# Only report high and critical findings
ai-risk-scanner scan . --min-risk high

# Exclude test fixtures and vendor directories
ai-risk-scanner scan . --exclude "tests/fixtures/*" --exclude "vendor/*"

# Full verbose JSON export
ai-risk-scanner scan . --verbose --format json --output full-report.json
```

### Extending the Provider Database

The provider signature database lives in `ai_risk_scanner/providers.yaml`. Add a new entry to detect any AI service not already covered:

```yaml
- id: my_custom_ai
  name: My Custom AI
  category: llm
  risk_level: high
  vendor_lock_in: true
  data_residency_concern: true
  patterns:
    imports:
      - "my_custom_ai"
      - "from mycustomai"
    env_vars:
      - "MYCUSTOMAI_API_KEY"
    urls:
      - "api.mycustomai.com"
    model_names:
      - "mycustom-v1"
    sdk_calls:
      - "CustomAIClient"
  nist_rmf_functions:
    - GOVERN
    - MAP
  ctem_category: EXPOSED
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
