"""Recognising secrets — both in file names and in content.

Two different jobs, both documented at docs/safety-and-tool-use.md §10:

* **File patterns** name things that must never be indexed, read, or auto-edited. This
  lives here rather than in ``indexing/`` because three layers need it (the indexer skips
  these files, ``read_file`` refuses them, and the policy engine keeps them out of the
  auto-edit allowance) and ``safety`` is the lowest of the three.
* **Content patterns** catch a secret that ends up in text anyway — a write, a commit, an
  audit record — so it can be flagged or redacted before it is ever written to a permanent
  log.
"""

from __future__ import annotations

import re

#: Filename patterns that must never be indexed, read by a tool, or auto-edited. Gitignore
#: syntax, matched against both the full relative path and the bare filename.
SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "env.local",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*.ppk",
    "credentials",
    "credentials.*",
    "*credentials.json",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    ".htpasswd",
    "*.tfstate",
    "*.tfstate.*",
    "secrets.*",
    "*.secrets",
    "service-account*.json",
    ".aws/credentials",
    ".ssh/*",
)

#: Content patterns, each labelled with what it caught (docs/safety-and-tool-use.md §10).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9\-_]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("jwt", re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "generic_assignment",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|APIKEY|API_KEY|CREDENTIAL)"
            r"[A-Z0-9_]*)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
        ),
    ),
)


def find_secrets(text: str) -> list[str]:
    """Which patterns matched, by label. Empty when nothing looks like a secret."""
    return [label for label, pattern in _PATTERNS if pattern.search(text)]


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for _label, pattern in _PATTERNS)


def redact(text: str) -> str:
    """Replace every matched secret with a labelled placeholder.

    Applied before anything reaches the audit log or a debug dump — a credential that
    reaches a permanent log is a leak with a timestamp on it.
    """
    redacted = text
    for label, pattern in _PATTERNS:
        redacted = pattern.sub(f"<redacted:{label}>", redacted)
    return redacted


def redact_value(value: object) -> object:
    """Recursively redact strings inside dicts/lists/tuples, for structured audit args."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
