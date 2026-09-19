"""Noticing when repository content is trying to give the model instructions.

Repository content is **data, not instructions** (docs/safety-and-tool-use.md §11). This
module is layer 2 of five, and it is worth being precise about how little it is asked to
do: it does not stop an injected instruction. Policy and approval do that. What this
provides is a *badge* — when a tool result in the last few steps looked like it was
addressing the model, the next approval says so and cites where it came from, so the
person reading the prompt knows why the proposal in front of them might be strange.

That framing sets the design priority, which is **not** catching every payload:

* **A false positive costs more than a miss.** A badge that fires on ordinary
  documentation is a badge people learn to scroll past, and then it is worth nothing on
  the day it matters (T2, approval fatigue). So the patterns below are narrow and
  anchored: "run the following" alone is ordinary prose, and only earns a finding when it
  arrives with a pipe-to-shell payload.
* **A finding must cite its source.** "Recent output contained instruction-like text" is
  unactionable; `README.md:12` can be looked at.

The heuristics can be evaded, and the doc says so. Nothing here is load-bearing for
safety; it is load-bearing for the user's ability to understand what they are approving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: How many steps a finding keeps badging approvals. Injected text and the proposal it
#: provoked are rarely in the same step: the model reads a file, then acts two steps later.
DEFAULT_WINDOW = 5

#: Cap on an excerpt. These go into approval panels and audit records, not log files.
_EXCERPT = 200

#: At most this many reasons per approval, so the panel stays readable.
_MAX_REASONS = 3


def _pattern(source: str) -> re.Pattern[str]:
    return re.compile(source, re.IGNORECASE)


#: Text that addresses the model rather than describing code.
#:
#: Each of these is anchored on a *directive* aimed at a reader who is an agent. The
#: near-misses they must not catch are real: "See the previous section for instructions"
#: and "The tests ignore deprecation warnings" both contain the obvious keywords.
_ADDRESSING: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override", _pattern(r"\bignore\s+(?:all\s+)?(?:the\s+)?previous\s+instructions?\b")),
    ("override", _pattern(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:above|previous|prior|earlier)\b")),
    ("override", _pattern(r"\bforget\s+(?:everything|all)\b.{0,30}\b(?:above|before|told)\b")),
    ("persona", _pattern(r"\byou\s+are\s+now\b")),
    ("persona", _pattern(r"\bas\s+an?\s+(?:ai|llm|language\s+model|assistant)\b.{0,40}\byou\s+should\b")),
    (
        "addressed",
        _pattern(r"\b(?:new|updated)\s+instructions?\s+for\s+(?:the\s+)?(?:ai|agent|assistant|model)\b"),
    ),
    ("addressed", _pattern(r"\bsystem\s+prompt\s*:")),
    ("addressed", _pattern(r"\b(?:ai|agent|assistant)\s+reading\s+this\b")),
)

#: Setup instructions that fetch and execute code. The pipe is what makes it a finding:
#: "run the migration before deploying" is a perfectly ordinary TODO.
_PIPE_TO_SHELL = _pattern(
    r"\b(?:curl|wget|iwr|invoke-webrequest)\b[^\n|]{0,200}\|\s*(?:sudo\s+)?"
    r"(?:ba|z|k|da)?sh\b|\b(?:curl|wget)\b[^\n|]{0,200}\|\s*(?:python|python3|perl|ruby|node)\b"
)

#: Characters that exist to be invisible. In a code file they are only ever there so that
#: what a person reads differs from what a machine reads.
_HIDDEN = _pattern("[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069\ufeff]")

#: A base64 run long enough to hide a script. Short ones are everywhere (hashes, keys in
#: fixtures), so the threshold is deliberately high.
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")


@dataclass(frozen=True)
class InjectionFinding:
    """One piece of tool output that looked like an instruction."""

    #: override | persona | addressed | pipe-to-shell | hidden-text | encoded-blob
    kind: str
    source: str
    line: int
    excerpt: str

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line}"

    def describe(self) -> str:
        return f"{self.location} contains instruction-like text ({self.kind})"


def scan_tool_result(text: str, *, source: str) -> list[InjectionFinding]:
    """Scan one tool result for instruction-like content.

    Args:
        text: The tool's output, as the model will see it.
        source: Where it came from, e.g. a workspace-relative path. Reported verbatim,
            because a finding nobody can locate is a finding nobody can act on.
    """
    if not text.strip():
        return []

    findings: list[InjectionFinding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        for kind, pattern in _ADDRESSING:
            if pattern.search(line):
                findings.append(_finding(kind, source, number, line))
                break
        else:
            if _PIPE_TO_SHELL.search(line):
                findings.append(_finding("pipe-to-shell", source, number, line))
            elif _HIDDEN.search(line):
                findings.append(_finding("hidden-text", source, number, line))
            elif _BASE64_BLOB.search(line):
                findings.append(_finding("encoded-blob", source, number, line))

    return findings


def _finding(kind: str, source: str, line: int, text: str) -> InjectionFinding:
    return InjectionFinding(kind=kind, source=source, line=line, excerpt=text.strip()[:_EXCERPT])


class InjectionMonitor:
    """Remembers recent findings, so approvals can carry `INJECTION?` for a few steps."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self._window = window
        self._seen: list[tuple[int, InjectionFinding]] = []

    def observe(self, *, step: int, text: str, source: str) -> list[InjectionFinding]:
        """Record what one tool result contained. Returns this result's findings."""
        findings = scan_tool_result(text, source=source)
        self._seen.extend((step, finding) for finding in findings)
        return findings

    def recent(self, step: int) -> list[InjectionFinding]:
        """Findings still inside the badging window at ``step``."""
        return [finding for seen_at, finding in self._seen if 0 <= step - seen_at < self._window]

    def badge(self, step: int) -> str | None:
        return "INJECTION?" if self.recent(step) else None

    def reasons(self, step: int) -> list[str]:
        """Lines for the approval panel, deduplicated by location."""
        lines: list[str] = []
        locations: set[str] = set()
        for finding in self.recent(step):
            if finding.location in locations:
                continue
            locations.add(finding.location)
            lines.append(finding.describe())
            if len(lines) >= _MAX_REASONS:
                break
        return lines
