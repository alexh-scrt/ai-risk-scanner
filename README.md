# AI Risk Inventory Scanner

> **Recursively scan codebases to detect, catalog, and risk-score all integrated AI services, APIs, and models.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NIST AI RMF](https://img.shields.io/badge/Framework-NIST%20AI%20RMF-green.svg)](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%20Playbook.pdf)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
   - [scan](#scan-command)
   - [info](#info-command)
   - [providers](#providers-command)
6. [Output Formats](#output-formats)
   - [Console (Rich Terminal)](#console-rich-terminal)
   - [JSON Inventory](#json-inventory)
   - [Markdown Audit Report](#markdown-audit-report)
7. [Detected AI Providers](#detected-ai-providers)
8. [Risk Scoring Methodology](#risk-scoring-methodology)
9. [NIST AI RMF Mapping](#nist-ai-rmf-mapping)
10. [CTEM Exposure Categories](#ctem-exposure-categories)
11. [Compliance Gap Analysis](#compliance-gap-analysis)
12. [Configuration Reference](#configuration-reference)
13. [Custom Provider Signatures](#custom-provider-signatures)
14. [CI/CD Integration](#cicd-integration)
15. [Development](#development)
16. [Architecture](#architecture)
17. [License](#license)

---

## Overview

**AI Risk Inventory Scanner** is a command-line security and governance tool that recursively walks a codebase and configuration directory tree to detect every reference to AI services, APIs, and models. It identifies imports, SDK calls, model name references, API endpoint URLs, environment variable keys, and hardcoded credentials belonging to 20+ AI providers.

For each detected provider, the scanner computes a **numeric risk score** (0–10), maps findings to the **[NIST AI Risk Management Framework (AI RMF)](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%20Playbook.pdf)** Govern / Map / Measure / Manage functions, and categorises exposures using **CTEM (Continuous Threat Exposure Management)** principles.

The output is delivered as:
- A **rich terminal summary** with colour-coded tables and severity badges.
- A **machine-readable JSON inventory** suitable for SIEM ingestion, dashboards, or further processing.
- A **Markdown audit report** ready for security reviews, board briefings, or compliance documentation.

---

## Features

| Feature | Detail |
|---------|--------|
| **Recursive scanning** | Python, JS/TS, YAML, TOML, JSON, `.env`, Dockerfile, shell scripts, Terraform, Go, Rust, Java, Ruby, and 30+ more file types |
| **20+ AI providers** | OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, Azure OpenAI, Hugging Face, Cohere, Mistral, Replicate, Groq, Together AI, LangChain, LlamaIndex, Ollama, Pinecone, Weaviate, ElevenLabs, Stability AI, SageMaker, Databricks, Perplexity AI |
| **Detection types** | SDK imports, SDK method calls, model name references, API endpoint URLs, environment variables, hardcoded credentials |
| **Risk scoring** | Numeric 0–10 score per provider with CTEM severity modifiers, credential escalation, vendor lock-in, and data residency factors |
| **NIST AI RMF mapping** | Findings mapped to GOVERN-1/2, MAP-1/3/4/5, MEASURE-1/2, MANAGE-1/2/3 |
| **CTEM categories** | External exposure, data exfiltration, third-party dependency, cloud dependency, supply chain, identity access, open source risk, biometric data |
| **Compliance gaps** | Hardcoded credentials, data residency risk, vendor lock-in, missing key management evidence |
| **Multi-format output** | Rich console, JSON, Markdown |
| **Extensible signatures** | Override or extend `providers.yaml` with custom provider signatures |
| **CI/CD integration** | Non-zero exit codes on critical findings enable pipeline gating |

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ai-risk-scanner/ai-risk-scanner.git
cd ai-risk-scanner

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package with development dependencies
pip install -e ".[dev]"
```

### Using pip (once published)

```bash
pip install ai-risk-scanner
```

### Requirements

- Python **3.11** or later
- Dependencies: `click>=8.1`, `rich>=13.0`, `pyyaml>=6.0`

---

## Quick Start

```bash
# Scan the current directory and display a rich terminal report
ai-risk-scanner scan .

# Scan a specific project directory
ai-risk-scanner scan /path/to/my_project

# Export a JSON inventory
ai-risk-scanner scan ./my_project --format json --output ai_inventory.json

# Export a Markdown audit report
ai-risk-scanner scan ./my_project --format markdown --output audit_report.md

# Verbose output with raw detection evidence
ai-risk-scanner scan ./my_project --verbose --show-evidence

# Scan with custom exclusions
ai-risk-scanner scan ./my_project \
    --exclude-dir tests \
    --exclude-dir docs \
    --max-file-size 2.0

# List all known provider signatures
ai-risk-scanner info

# Filter providers by risk level
ai-risk-scanner providers --risk high
ai-risk-scanner providers --category llm
```

---

## CLI Reference

### `scan` Command

```
ai-risk-scanner scan [PATH] [OPTIONS]
```

`PATH` defaults to the current directory (`.`).

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `console` | Output format: `console`, `json`, `markdown`, `md` |
| `--output` | `-o` | — | File path to write the report. Prints to stdout if omitted (json/markdown). |
| `--verbose` | `-v` | `false` | Enable verbose output including detection evidence. |
| `--show-evidence` | | `false` | Include raw detection evidence lines in the report. |
| `--max-evidence` | | `3` | Max evidence lines per finding (1–50). |
| `--exclude-dir` | | — | Directory name to exclude. Repeatable. Merged with defaults. |
| `--exclude-path` | | — | Path prefix to exclude entirely. Repeatable. |
| `--max-file-size` | | `5.0` | Maximum file size in MB to scan. |
| `--include-hidden` | | `false` | Include hidden files and directories (dot-prefixed). |
| `--follow-symlinks` | | `false` | Follow symbolic links during traversal. |
| `--providers-yaml` | | bundled | Path to a custom `providers.yaml` signature database. |
| `--no-compliance-gaps` | | `false` | Omit compliance gap details from output. |
| `--no-nist-mapping` | | `false` | Omit NIST AI RMF function mapping from output. |
| `--no-ctem-mapping` | | `false` | Omit CTEM exposure category mapping from output. |
| `--scan-id` | | auto UUID | Custom scan identifier. |
| `--encoding` | | `utf-8` | Character encoding for reading source files. |

**Exit Codes:**

| Code | Meaning |
|------|---------|
| `0` | Scan completed successfully, no critical findings. |
| `1` | Scan error (bad path, permission denied, etc.). |
| `2` | Scan completed but **CRITICAL** findings were detected. Use this to gate CI pipelines. |
| `130` | Scan interrupted by user (Ctrl+C). |

### `info` Command

```
ai-risk-scanner info
```

Displays:
- Scanner version and description.
- Table of all loaded provider signatures with category, risk level, lock-in, and data risk flags.
- Complete list of supported file extensions.
- Credential pattern count.

### `providers` Command

```
ai-risk-scanner providers [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--category` | Filter by category: `llm`, `image_gen`, `speech`, `embedding`, `mlops`, `cloud_ai` |
| `--risk` | Filter by baseline risk: `critical`, `high`, `medium`, `low` |
| `--providers-yaml` | Path to a custom `providers.yaml` |

---

## Output Formats

### Console (Rich Terminal)

The default output renders a multi-section rich terminal report:

```
╭──────────────────────────────────────────────────────────────╮
│ ⚡ AI Risk Scanner  |  Scan Report                           │
│                                                              │
│ Scan ID:      a3f8c1d2-...                                   │
│ Target path:  /path/to/my_project                            │
│ Scanner:      v0.1.0                                         │
│ Started:      2024-06-01 10:00:00 UTC                        │
│ Duration:     1.42s                                          │
│ Files scanned: 47  (12 skipped)                              │
╰──────────────────────────────────────────────────────────────╯

╭── Executive Summary ─────────────────────────────────────────╮
│  Overall Risk:  ⚠ HIGH   Score: [=======   ] 7.2/10          │
│                                                              │
│  Providers detected:    4                                    │
│  Risk findings:         5                                    │
│  Compliance gaps:       18                                   │
│  Detection matches:     32                                   │
│                                                              │
│  Findings by severity:                                       │
│    CRITICAL: 1  HIGH: 2  MEDIUM: 2  LOW: 0                   │
│                                                              │
│  ⚠ Hardcoded credentials: openai                             │
│  ⚠ Data residency risk:   openai, anthropic, cohere          │
╰──────────────────────────────────────────────────────────────╯

┌─ Detected AI Providers ──────────────────────────────────────┐
│ Provider        │ Category │ Risk     │ Matches │ Lock-in    │
│ OpenAI          │ LLM      │ ⚠ HIGH   │ 12      │ ✓          │
│ Anthropic       │ LLM      │ ⚠ HIGH   │ 8       │ ✓          │
│ LangChain       │ LLM      │ ● MEDIUM │ 6       │ ✗          │
│ Hugging Face    │ MLOps    │ ● MEDIUM │ 6       │ ✗          │
└──────────────────────────────────────────────────────────────┘
```

### JSON Inventory

The JSON output follows a structured schema suitable for machine processing:

```bash
ai-risk-scanner scan ./my_project --format json --output inventory.json
```

```json
{
  "schema_version": "1.0",
  "metadata": {
    "scan_id": "a3f8c1d2-4b5e-6f7a-8b9c-0d1e2f3a4b5c",
    "scan_path": "/path/to/my_project",
    "scanner_version": "0.1.0",
    "started_at": "2024-06-01T10:00:00+00:00",
    "completed_at": "2024-06-01T10:00:01.42+00:00",
    "duration_seconds": 1.42,
    "total_files_scanned": 47,
    "total_files_skipped": 12
  },
  "summary": {
    "total_providers_detected": 4,
    "total_findings": 5,
    "total_compliance_gaps": 18,
    "total_detection_matches": 32,
    "findings_by_severity": {
      "critical": 1,
      "high": 2,
      "medium": 2,
      "low": 0
    },
    "providers_with_credentials": ["openai"],
    "providers_with_data_residency_risk": ["openai", "anthropic", "cohere"],
    "providers_with_vendor_lock_in": ["openai", "anthropic", "cohere"],
    "overall_risk_level": "high",
    "overall_risk_score": 7.2
  },
  "assets": {
    "openai": {
      "provider_id": "openai",
      "provider_name": "OpenAI",
      "category": "llm",
      "baseline_risk_level": "high",
      "vendor_lock_in": true,
      "data_residency_concern": true,
      "total_matches": 12,
      "files_detected": ["src/app.py", ".env", "config.yaml"],
      "detection_types_found": ["import", "env_var", "model_name", "sdk_call"],
      "has_credential_exposure": true
    }
  },
  "findings": [
    {
      "finding_id": "finding_openai_llm",
      "title": "OpenAI Integration Detected",
      "risk_level": "high",
      "risk_score": 9.7,
      "provider_id": "openai",
      "vendor_lock_in": true,
      "data_residency_concern": true,
      "has_hardcoded_credentials": true,
      "nist_functions": ["GOVERN-1", "MAP-1", "MAP-5", "MEASURE-2", "MANAGE-1"],
      "ctem_categories": ["external_exposure", "data_exfiltration", "third_party_dependency"],
      "compliance_gaps": [ "..." ]
    }
  ]
}
```

### Markdown Audit Report

```bash
ai-risk-scanner scan ./my_project --format markdown --output audit_report.md
```

The Markdown report is structured for compliance documentation and includes:
- Scan metadata table
- Executive summary with risk level and key metrics
- Detected AI providers table
- Per-finding detail with NIST and CTEM references
- Consolidated compliance gaps with remediation guidance
- NIST AI RMF function coverage table
- CTEM exposure category coverage table
- Affected files list
- Detection evidence appendix

---

## Detected AI Providers

The scanner ships with a built-in signature database (`providers.yaml`) covering **21 providers**:

| Provider | Category | Baseline Risk | Vendor Lock-in | Data Residency |
|----------|----------|--------------|----------------|----------------|
| OpenAI | LLM | High | Yes | Yes |
| Anthropic | LLM | High | Yes | Yes |
| Cohere | LLM | High | Yes | Yes |
| Mistral AI | LLM | High | Yes | Yes |
| Groq | LLM | High | No | Yes |
| Together AI | LLM | High | No | Yes |
| Perplexity AI | LLM | High | Yes | Yes |
| LangChain | LLM | Medium | No | Yes |
| LlamaIndex | LLM | Medium | No | Yes |
| Ollama (Local) | LLM | Low | No | No |
| AWS Bedrock | Cloud AI | Medium | Yes | No |
| Google Vertex AI | Cloud AI | Medium | Yes | Yes |
| Azure OpenAI | Cloud AI | Medium | Yes | No |
| Hugging Face | MLOps | Medium | No | Yes |
| Replicate | MLOps | High | No | Yes |
| Pinecone | MLOps | Medium | Yes | Yes |
| Weaviate | MLOps | Low | No | No |
| AWS SageMaker | MLOps | Medium | Yes | No |
| Databricks / MLflow | MLOps | Medium | Yes | No |
| ElevenLabs | Speech | High | Yes | Yes |
| Stability AI | Image Gen | Medium | Yes | Yes |
| OpenAI Whisper | Speech | High | Yes | Yes |

**Detection patterns per provider include:**

- `imports` — Python/JS/TS import and require statements
- `env_vars` — Environment variable name references
- `urls` — API endpoint URL patterns
- `model_names` — Specific model identifier strings (e.g., `gpt-4o`, `claude-3-5-sonnet`)
- `sdk_calls` — SDK method and class instantiation patterns (e.g., `OpenAI()`, `client.chat.completions`)

**Cross-provider credential patterns** detect hardcoded secrets:

| Pattern | Example |
|---------|---------|
| OpenAI Key | `sk-[A-Za-z0-9]{20,}` |
| Anthropic Key | `sk-ant-[A-Za-z0-9-_]{20,}` |
| HuggingFace Token | `hf_[A-Za-z0-9]{20,}` |
| Bearer Token | `Bearer [A-Za-z0-9...]{20,}` |
| Generic API Key | `api_key = "..."` |

---

## Risk Scoring Methodology

Each detected AI provider asset receives a **numeric risk score** from `0.0` to `10.0` computed as follows:

### 1. Base Score (from baseline risk level)

| Risk Level | Base Score |
|------------|------------|
| CRITICAL | 9.0 |
| HIGH | 7.0 |
| MEDIUM | 5.0 |
| LOW | 2.5 |
| INFO | 1.0 |

### 2. CTEM Severity Modifier

The highest CTEM category severity modifier among applicable categories is applied **multiplicatively**:

| CTEM Category | Multiplier |
|---------------|------------|
| `biometric_data` | ×1.8 |
| `data_exfiltration` | ×1.5 |
| `identity_access` | ×1.3 |
| `external_exposure` | ×1.2 |
| `supply_chain` | ×1.2 |
| `third_party_dependency` | ×1.1 |
| `cloud_dependency` | ×1.0 |
| `open_source_risk` | ×0.9 |

### 3. Aggravating Factors (additive)

| Factor | Score Increment |
|--------|-----------------|
| Hardcoded credentials detected | +2.0 |
| Data residency concern | +0.4 |
| Vendor lock-in | +0.3 |
| Multi-file spread (per extra file, capped at +0.5) | +0.1 |

### 4. Final Severity Classification

| Score Range | Risk Level |
|-------------|------------|
| ≥ 8.5 | CRITICAL |
| ≥ 6.5 | HIGH |
| ≥ 4.0 | MEDIUM |
| > 0.0 | LOW |
| 0.0 | INFO |

**Example:** OpenAI with `data_exfiltration` CTEM modifier (×1.5), hardcoded credentials (+2.0), vendor lock-in (+0.3), and data residency (+0.4):

```
base=7.0 × ctem_modifier=1.5 = 10.5 → clamped to 10.0
+ credentials=2.0 → 10.0 (already at max)
Final: 10.0 / CRITICAL
```

---

## NIST AI RMF Mapping

Findings are mapped to the [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence) core functions:

| Function ID | Core Function | Description |
|-------------|---------------|-------------|
| `GOVERN-1` | Govern | Establish organizational policies, roles, and accountability for AI risk management. |
| `GOVERN-2` | Govern | Foster organizational culture and workforce readiness for responsible AI. |
| `MAP-1` | Map | Identify and categorize the AI system's context, purpose, and intended use. |
| `MAP-3` | Map | Classify AI system risks based on potential harms and deployment context. |
| `MAP-4` | Map | Identify and assess risks from third-party AI components and supply chain. |
| `MAP-5` | Map | Identify potential harms to individuals, groups, or society from AI system outputs. |
| `MEASURE-1` | Measure | Define and collect metrics to evaluate AI risk exposure. |
| `MEASURE-2` | Measure | Evaluate AI system trustworthiness characteristics including bias, robustness, and privacy. |
| `MANAGE-1` | Manage | Apply risk treatment strategies including mitigation, transfer, and acceptance. |
| `MANAGE-2` | Manage | Implement technical and governance controls for identified AI risks. |
| `MANAGE-3` | Manage | Continuously monitor AI risks and review effectiveness of controls. |

### Provider-to-Function Mapping Reference

| Provider Type | GOVERN-1 | GOVERN-2 | MAP-1 | MAP-3 | MAP-4 | MAP-5 | MEASURE-1 | MEASURE-2 | MANAGE-1 | MANAGE-2 | MANAGE-3 |
|---------------|:--------:|:--------:|:-----:|:-----:|:-----:|:-----:|:---------:|:---------:|:--------:|:--------:|:--------:|
| External LLM APIs (OpenAI, Anthropic, Cohere) | ✓ | | ✓ | | | ✓ | | ✓ | ✓ | | |
| Cloud AI (Bedrock, Vertex AI, Azure OpenAI) | | ✓ | | ✓ | | | ✓ | | | ✓ | |
| ML Orchestration (LangChain, LlamaIndex) | | | ✓ | | ✓ | | | ✓ | | | |
| MLOps Platforms (HuggingFace, SageMaker, Databricks) | | ✓ | | ✓ | ✓ | | ✓ | | | ✓ | ✓ |
| Local/Open Models (Ollama) | | | ✓ | | | | | ✓ | | | |
| Hardcoded Credentials | ✓ | | | | | | | | | ✓ | |

---

## CTEM Exposure Categories

The scanner classifies findings against **Continuous Threat Exposure Management** exposure categories:

| Category | Severity Modifier | Description |
|----------|:-----------------:|-------------|
| `biometric_data` | ×1.8 | Processing of biometric data (voice, face) by AI services. ElevenLabs voice cloning is a primary example. |
| `data_exfiltration` | ×1.5 | Sensitive data transmitted to third-party AI provider infrastructure. Any external LLM API call is a potential vector. |
| `identity_access` | ×1.3 | Inadequate identity and access management for AI service credentials, including hardcoded API keys. |
| `external_exposure` | ×1.2 | AI service APIs exposed to or called from external networks. |
| `supply_chain` | ×1.2 | Risk from open-source models or components with unclear provenance. Common with Hugging Face and Replicate. |
| `third_party_dependency` | ×1.1 | Critical dependency on external AI provider availability and reliability. |
| `cloud_dependency` | ×1.0 | Dependency on cloud-platform-specific AI services creating lock-in risk. |
| `open_source_risk` | ×0.9 | Risks associated with open-source AI models including licensing and safety. |

---

## Compliance Gap Analysis

For each detected provider, the scanner automatically identifies compliance gaps:

### Gap Types Generated

| Gap Type | Severity | Trigger Condition |
|----------|----------|-----------------|
| Hardcoded Credentials | **CRITICAL** | Credential pattern matched in source file |
| Data Residency Risk | **HIGH** | Provider has `data_residency_concern: true` |
| Vendor Lock-In Risk | **MEDIUM** | Provider has `vendor_lock_in: true` |
| API Key Management Not Verified | **MEDIUM** | SDK import/call detected but no env var reference found |
| Provider Compliance Notes | **LOW** | One gap per compliance note in `providers.yaml` |

### Remediation Guidance

Each compliance gap includes specific remediation steps. Examples:

**Hardcoded Credentials:**
> Remove hardcoded API keys immediately. Store secrets in environment variables, a dedicated secrets manager (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault), or a `.env` file excluded from version control. Rotate all exposed credentials.

**Data Residency Risk:**
> Review the provider's data processing agreement and privacy policy. Configure region-specific endpoints where available. Avoid sending PII, PHI, or sensitive business data to this API without appropriate legal safeguards (GDPR Article 46, CCPA §1798.100).

**Vendor Lock-In Risk:**
> Abstract provider API calls behind an interface or adapter layer to allow provider substitution. Consider multi-provider strategies or open standards (e.g., OpenAI-compatible endpoints) where feasible.

---

## Configuration Reference

### Default Excluded Directories

The following directories are excluded from scanning by default:

```
.git  .hg  .svn  .tox  .venv  venv  env  __pycache__  .mypy_cache
.pytest_cache  .ruff_cache  .cache  .idea  .vscode  node_modules
.npm  .yarn  bower_components  dist  build  .build  target  out
.next  .nuxt  coverage  .coverage  htmlcov  .eggs  logs  log
tmp  temp  .terraform  .serverless  .webpack  vendor  third_party
site-packages  lib  libs
```

Add custom exclusions with `--exclude-dir`:

```bash
ai-risk-scanner scan . --exclude-dir my_fixtures --exclude-dir legacy_code
```

### Supported File Extensions

The scanner processes files with these extensions:

```
.py .js .ts .jsx .tsx .mjs .cjs        # JavaScript / TypeScript
.yaml .yml .json .toml .cfg .ini .conf  # Configuration
.env .properties                        # Environment / Properties
.sh .bash .zsh                          # Shell scripts
.dockerfile .tf .hcl                    # Infrastructure
.rb .go .java .kt .rs .php .cs .cpp .c # Other languages
.txt .md .rst .xml .gradle .lock        # Documentation / Build
```

Plus named files: `Dockerfile`, `.env`, `.env.local`, `.env.production`, `docker-compose.yml`, `requirements.txt`, `Pipfile`, `package.json`, `pyproject.toml`, and more.

Add custom extensions:

```bash
# Via CLI flag
ai-risk-scanner scan . --extra-extension .custom

# Via ScanConfig in Python
from ai_risk_scanner.scanner import ScanConfig, Scanner
config = ScanConfig(extra_extensions={".custom", ".myext"})
scanner = Scanner(config=config)
```

---

## Custom Provider Signatures

You can extend or replace the built-in provider database by providing a custom `providers.yaml`.

### Schema

```yaml
providers:
  - id: my_custom_ai           # Unique snake_case identifier
    name: My Custom AI          # Human-readable display name
    category: llm               # llm | image_gen | speech | embedding | mlops | cloud_ai
    risk_level: high            # critical | high | medium | low
    vendor_lock_in: true        # Creates proprietary lock-in?
    data_residency_concern: true # Data leaves your jurisdiction?
    patterns:
      imports:                  # Python/JS import patterns
        - "import my_custom_ai"
        - "from my_custom_ai"
      env_vars:                 # Environment variable names
        - "MY_CUSTOM_AI_API_KEY"
        - "MY_CUSTOM_AI_TOKEN"
      urls:                     # API endpoint URL patterns
        - "api.mycustomai.com"
      model_names:              # Model name references
        - "custom-model-v1"
        - "custom-model-v2"
      sdk_calls:                # SDK method / class patterns
        - "MyCustomAI("
        - "client.custom_call"
    nist_rmf_functions:
      - "GOVERN-1"
      - "MAP-1"
    ctem_categories:
      - "external_exposure"
      - "data_exfiltration"
    compliance_notes:
      - "Ensure API key rotation policy is in place."
      - "Review data retention policy with provider."
    docs_url: "https://docs.mycustomai.com"

credential_patterns:
  - name: "My Custom AI Key"
    pattern: "mca-[A-Za-z0-9]{20,}"
    risk_level: critical
    description: "Hardcoded My Custom AI API key detected"
```

### Usage

```bash
ai-risk-scanner scan ./my_project --providers-yaml ./custom_providers.yaml
```

Or via the Python API:

```python
from pathlib import Path
from ai_risk_scanner.scanner import scan_path

result = scan_path(
    target_path=Path("./my_project"),
    providers_yaml_path=Path("./custom_providers.yaml"),
)
```

---

## CI/CD Integration

The scanner returns exit code `2` when **CRITICAL** findings are detected, making it easy to gate CI/CD pipelines.

### GitHub Actions

```yaml
# .github/workflows/ai-risk-scan.yml
name: AI Risk Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ai-risk-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install AI Risk Scanner
        run: pip install ai-risk-scanner

      - name: Run AI Risk Scan
        run: |
          ai-risk-scanner scan . \
            --format json \
            --output ai-risk-report.json \
            --exclude-dir tests \
            --exclude-dir docs
        # Exit code 2 = critical findings; allow but capture
        continue-on-error: true

      - name: Upload Risk Report
        uses: actions/upload-artifact@v4
        with:
          name: ai-risk-report
          path: ai-risk-report.json

      - name: Gate on Critical Findings
        run: |
          ai-risk-scanner scan . --format json | \
          python -c "
import json, sys
data = json.load(sys.stdin)
critical = data['summary']['findings_by_severity']['critical']
if critical > 0:
    print(f'FAILED: {critical} critical AI risk finding(s) detected!')
    sys.exit(1)
print('PASSED: No critical AI risk findings.')
"
```

### GitLab CI

```yaml
# .gitlab-ci.yml
ai-risk-scan:
  stage: security
  image: python:3.11-slim
  script:
    - pip install ai-risk-scanner
    - ai-risk-scanner scan . --format json --output gl-ai-risk.json
  artifacts:
    reports:
      security: gl-ai-risk.json
    paths:
      - gl-ai-risk.json
    expire_in: 30 days
  allow_failure: false  # Block pipeline on critical findings (exit code 2)
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ai-risk-scan
        name: AI Risk Inventory Scan
        language: python
        additional_dependencies: [ai-risk-scanner]
        entry: ai-risk-scanner scan
        args: ["--format", "console", "--no-nist-mapping", "--no-ctem-mapping"]
        pass_filenames: false
        always_run: true
```

---

## Development

### Setup

```bash
git clone https://github.com/ai-risk-scanner/ai-risk-scanner.git
cd ai-risk-scanner
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_risk_scanner --cov-report=html

# Run a specific test module
pytest tests/test_detectors.py -v
pytest tests/test_risk_engine.py -v
pytest tests/test_scanner.py -v
pytest tests/test_reporter.py -v

# Run integration tests only
pytest tests/test_scanner.py -v -k "integration"
```

### Project Structure

```
ai_risk_scanner/
├── __init__.py          # Package init and version export
├── cli.py               # Click-based CLI entry point
├── scanner.py           # FileWalker + Scanner orchestrator
├── detectors.py         # Pattern-matching detection engine
├── inventory.py         # Data models (AIAsset, RiskFinding, ScanResult, etc.)
├── risk_engine.py       # Risk scoring + NIST/CTEM mapping
├── reporter.py          # Console, JSON, and Markdown report generators
└── providers.yaml       # Provider signature database

tests/
├── __init__.py
├── test_detectors.py    # Unit tests for detection engine
├── test_risk_engine.py  # Unit tests for risk scoring
├── test_scanner.py      # Integration tests for file walker + scanner
├── test_reporter.py     # Unit tests for report generation
├── test_inventory.py    # Unit tests for data models
└── fixtures/
    └── sample_project/
        ├── app.py       # Sample file with 20+ AI provider integrations
        └── .env         # Sample env file with API key references
```

### Python API

You can use the scanner programmatically:

```python
from pathlib import Path
from ai_risk_scanner.scanner import scan_path, ScanConfig, Scanner
from ai_risk_scanner.reporter import Reporter, ReportConfig

# Simple one-call scan
result = scan_path(
    target_path=Path("./my_project"),
    exclude_dirs={"tests", "docs"},
    max_file_size_mb=5.0,
)

print(f"Risk Level: {result.summary.overall_risk_level.value}")
print(f"Risk Score: {result.summary.overall_risk_score:.1f}/10")
print(f"Providers:  {result.summary.total_providers_detected}")
print(f"Findings:   {result.summary.total_findings}")

# Print to console
reporter = Reporter(config=ReportConfig(show_evidence=True, verbose=True))
reporter.print_console(result)

# Export JSON
reporter.write_json(result, Path("inventory.json"))

# Export Markdown
reporter.write_markdown(result, Path("audit_report.md"))

# Iterate findings
for finding in result.sorted_findings:
    print(f"[{finding.risk_level.value.upper()}] {finding.title} — score={finding.risk_score}")
    for gap in finding.compliance_gaps:
        print(f"  Gap: {gap.title} ({gap.risk_level.value})")
        print(f"  Remediation: {gap.remediation}")

# Access raw assets
if "openai" in result.assets:
    openai_asset = result.assets["openai"]
    print(f"OpenAI detected in {len(openai_asset.files_detected)} file(s)")
    print(f"Total matches: {openai_asset.total_matches}")
    print(f"Hardcoded credentials: {openai_asset.has_credential_exposure}")
```

### Advanced Configuration

```python
from pathlib import Path
from ai_risk_scanner.scanner import Scanner, ScanConfig
from ai_risk_scanner.risk_engine import RiskEngine, RiskScoringConfig

# Custom scoring weights
scoring_config = RiskScoringConfig(
    credential_score_increment=3.0,   # Increase credential penalty
    data_residency_increment=0.8,     # Increase data residency penalty
    vendor_lock_in_increment=0.5,
    apply_ctem_modifiers=True,
)

# Custom scan configuration
scan_config = ScanConfig(
    excluded_dirs={"tests", "docs", "node_modules", ".git"},
    excluded_paths={"/path/to/vendor"},
    max_file_size_bytes=2 * 1024 * 1024,  # 2 MB
    follow_symlinks=False,
    include_hidden=False,
    extra_extensions={".custom"},
    providers_yaml_path=Path("./custom_providers.yaml"),
    scan_id="my-audit-2024-q4",
)

scanner = Scanner(config=scan_config)
result = scanner.scan(Path("./my_project"))

# Apply custom risk scoring separately
risk_engine = RiskEngine(scoring_config=scoring_config)
findings = risk_engine.analyze(result.assets)
for f in findings:
    print(f"{f.provider_name}: {f.risk_score:.2f} ({f.risk_level.value})")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                     │
│  Click-based entry point, argument parsing, error        │
│  handling, exit code management                          │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                    Scanner (scanner.py)                   │
│  FileWalker: recursive directory traversal +              │
│  filtering. Scanner: orchestrates the full pipeline.      │
└──────┬───────────────────┬────────────────────┬──────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌────────────┐   ┌─────────────────┐   ┌──────────────────┐
│  Detectors │   │   Risk Engine   │   │    Reporter       │
│ (detectors)│   │ (risk_engine.py)│   │  (reporter.py)   │
│            │   │                 │   │                   │
│ Loads YAML │   │ Scores assets,  │   │ Console (rich),   │
│ signatures,│   │ maps to NIST    │   │ JSON, Markdown    │
│ scans files│   │ AI RMF & CTEM,  │   │ output formats    │
│ line by    │   │ generates       │   │                   │
│ line for   │   │ compliance gaps │   │                   │
│ 5 pattern  │   │                 │   │                   │
│ types      │   │                 │   │                   │
└─────┬──────┘   └────────┬────────┘   └──────────────────┘
      │                   │
      ▼                   ▼
┌─────────────────────────────────────────────────────────┐
│               Inventory Data Models (inventory.py)        │
│  DetectionMatch, AIAsset, RiskFinding, ComplianceGap,    │
│  ScanResult, ScanMetadata, ScanSummary                   │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                  providers.yaml                           │
│  Provider signature database: patterns, risk metadata,   │
│  NIST/CTEM mappings, compliance notes                    │
└─────────────────────────────────────────────────────────┘
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **[NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework)** — the governance framework used for finding classification.
- **[CTEM (Continuous Threat Exposure Management)](https://www.gartner.com/en/security/glossary/continuous-threat-exposure-management)** — the exposure management framework used for category classification.
- **[Rich](https://github.com/Textualize/rich)** — for beautiful terminal output.
- **[Click](https://click.palletsprojects.com/)** — for the CLI framework.
- **[PyYAML](https://pyyaml.org/)** — for parsing the provider signature database.

---

*Built for AI governance, security audits, and responsible AI deployment.*
