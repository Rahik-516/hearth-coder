"""Reading a unified diff, so code can check what the model says about it.

Two workflows hand a diff to a model — `/commit` to describe it, `/review` to criticise it —
and both have the same weakness: the model may say things the diff does not support. The
cheapest defence is not a better prompt. It is to make the diff something code can *check
against*: parsed into files and lines, with each line's position in the new file known.

That is what this module is. It does three things:

* **parse** ``git diff`` output into files and numbered lines,
* **render** it for a model with the new-side line number beside every line, so a citation
  is something the model can copy rather than compute — a 4B model counting lines in a
  hunk header's arithmetic gets them wrong, and one copying a number it can see does not,
* **verify** ``path:line`` citations in the model's answer against what was actually shown.

The parser follows the hunk header's own line counts rather than pattern-matching the
lines. A deleted line whose text begins ``-- `` looks exactly like a ``--- a/file`` header
to a matcher and is not one to a counter, and a diff of a SQL or Lua file is where that
bites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from hearth.core.context.tokens import TokenEstimator

LineKind = Literal["add", "ctx", "del"]
FileStatus = Literal["added", "deleted", "modified", "renamed", "binary"]

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+)$")
_HUNK_HEADER = re.compile(r"^@@ -(?P<old>\d+)(?:,(?P<oldn>\d+))? \+(?P<new>\d+)(?:,(?P<newn>\d+))? @@")
#: `path/to/file.ext:12` or `:12-18`. The extension requirement keeps `foo:12` in prose
#: ("see step 3: 12 files") from being read as a citation; the lookbehind keeps a match
#: from starting in the middle of a longer token.
_CITATION = re.compile(r"(?<![\w./-])(?P<path>[\w./-]+\.\w+):(?P<start>\d+)(?:-(?P<end>\d+))?")


@dataclass(frozen=True)
class DiffLine:
    kind: LineKind
    text: str
    #: Position in the *new* file. None for a deleted line, which has no new position.
    new_line: int | None = None


@dataclass
class FileDiff:
    path: str
    status: FileStatus = "modified"
    lines: list[DiffLine] = field(default_factory=list)
    #: Where the file was before a rename, for the header only.
    old_path: str | None = None

    @property
    def added(self) -> int:
        return sum(1 for line in self.lines if line.kind == "add")

    @property
    def removed(self) -> int:
        return sum(1 for line in self.lines if line.kind == "del")

    def citable_lines(self) -> frozenset[int]:
        """New-side line numbers a model was shown, and so may legitimately cite."""
        return frozenset(line.new_line for line in self.lines if line.new_line is not None)


@dataclass
class RenderedDiff:
    """A diff laid out for a model, and what had to be left out to fit."""

    text: str
    included: list[str] = field(default_factory=list)
    #: Whole files that did not fit. Named rather than dropped silently: a review that
    #: says "looks good" about a diff it only half saw is the failure to avoid.
    omitted: list[str] = field(default_factory=list)
    #: Some file was cut part-way to fit.
    truncated: bool = False


@dataclass(frozen=True)
class Citation:
    path: str
    line: int
    verified: bool
    #: Why it is not verified, worded for the person reading the footer.
    reason: str = ""


# ------------------------------------------------------------------- parsing


def parse_diff(text: str) -> list[FileDiff]:
    """Parse ``git diff`` output. Tolerates renames, binary files, additions and deletions."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    old_left = new_left = 0
    old_no = new_no = 0

    for raw in text.splitlines():
        # Inside a hunk the header's counts decide what a line is, not its first
        # characters — see the module docstring.
        if current is not None and (old_left > 0 or new_left > 0):
            marker = raw[:1]
            body = raw[1:]
            if marker == "+":
                current.lines.append(DiffLine("add", body, new_no))
                new_no += 1
                new_left -= 1
                continue
            if marker == "-":
                current.lines.append(DiffLine("del", body))
                old_no += 1
                old_left -= 1
                continue
            if marker == " " or raw == "":
                current.lines.append(DiffLine("ctx", body, new_no))
                new_no += 1
                old_no += 1
                new_left -= 1
                old_left -= 1
                continue
            if marker == "\\":  # "\ No newline at end of file"
                continue
            # Anything else means the counts lied; fall through and treat it as a header.
            old_left = new_left = 0

        header = _DIFF_HEADER.match(raw)
        if header:
            current = FileDiff(path=header.group("new"))
            if header.group("old") != header.group("new"):
                current.old_path = header.group("old")
            files.append(current)
            old_left = new_left = 0
            continue

        if current is None:
            continue

        if raw.startswith("new file mode"):
            current.status = "added"
        elif raw.startswith("deleted file mode"):
            current.status = "deleted"
        elif raw.startswith("rename to "):
            current.status = "renamed"
            current.path = raw.removeprefix("rename to ").strip()
        elif raw.startswith(("Binary files", "GIT binary patch")):
            current.status = "binary"
        elif raw.startswith("+++ b/"):
            current.path = raw.removeprefix("+++ b/").strip()
        else:
            hunk = _HUNK_HEADER.match(raw)
            if hunk:
                old_no = int(hunk.group("old"))
                new_no = int(hunk.group("new"))
                old_left = int(hunk.group("oldn") if hunk.group("oldn") is not None else 1)
                new_left = int(hunk.group("newn") if hunk.group("newn") is not None else 1)

    return files


# ----------------------------------------------------------------- rendering


def render(
    files: list[FileDiff],
    *,
    budget_tokens: int,
    estimator: TokenEstimator,
    numbered: bool = True,
) -> RenderedDiff:
    """Lay files out for a model within a token budget.

    Files go in whole, in order, until the next would not fit. One file larger than the
    whole budget is cut line by line rather than skipped: a single enormous file is the
    common case for a generated lockfile, and dropping it entirely would hide the fact that
    it changed, while truncating it shows that and how much.

    ``numbered`` puts each surviving line's new-file position beside it. Reviews need it,
    since the model has to cite; a commit message does not, and the digits are tokens.
    """
    out: list[str] = []
    result = RenderedDiff(text="")
    used = 0

    for file in files:
        block = _render_file(file, numbered=numbered)
        cost = estimator.estimate(block)

        if used + cost <= budget_tokens:
            out.append(block)
            used += cost
            result.included.append(file.path)
            continue

        room = budget_tokens - used
        if not result.included and room > 0:
            # Nothing else has been shown, so a partial view of this file is better than
            # a promise about the rest.
            cut = _truncate_block(file, room, estimator, numbered=numbered)
            out.append(cut)
            result.included.append(file.path)
            result.truncated = True
            used = budget_tokens
            continue

        result.omitted.append(file.path)

    result.text = "\n".join(out)
    return result


def _render_file(file: FileDiff, *, numbered: bool) -> str:
    label = file.status if file.old_path is None else f"renamed from {file.old_path}"
    lines = [f"=== {file.path} ({label}, +{file.added} -{file.removed}) ==="]

    if file.status == "binary":
        lines.append("  (binary file; contents not shown)")
        return "\n".join(lines)

    previous: int | None = None
    for line in file.lines:
        if line.new_line is not None:
            if previous is not None and line.new_line > previous + 1:
                lines.append("  ...")
            previous = line.new_line
        lines.append(_render_line(line, numbered=numbered))
    return "\n".join(lines)


def _render_line(line: DiffLine, *, numbered: bool) -> str:
    sign = {"add": "+", "del": "-", "ctx": " "}[line.kind]
    if not numbered:
        return f"{sign}{line.text}"
    number = f"{line.new_line:>5}" if line.new_line is not None else "     "
    return f"{number} {sign}{line.text}"


def _truncate_block(
    file: FileDiff, budget: int, estimator: TokenEstimator, *, numbered: bool
) -> str:
    header = f"=== {file.path} ({file.status}, +{file.added} -{file.removed}) ==="
    kept: list[str] = [header]
    used = estimator.estimate(header)

    for line in file.lines:
        rendered = _render_line(line, numbered=numbered)
        cost = estimator.estimate(rendered)
        if used + cost > budget:
            kept.append(f"  ... (file cut here; {len(file.lines) - len(kept) + 1} more line(s))")
            break
        kept.append(rendered)
        used += cost
    return "\n".join(kept)


# -------------------------------------------------------------- verification


def verify_citations(answer: str, files: list[FileDiff]) -> list[Citation]:
    """Check every ``path:line`` in a model's answer against the diff it was shown.

    A citation is verified when its path is one of the diff's files and its line is a
    new-side line the model was actually given. That is deliberately strict: the model was
    handed a numbered view, so a line outside it was invented or miscounted, and either way
    the reader should be told before they go looking for it.

    A path matches by exact name, or by unique suffix — models routinely drop a leading
    directory, and refusing ``models.py`` when the diff has ``src/billing/models.py`` would
    flag correct findings. An ambiguous suffix is *not* verified, since guessing which file
    was meant is the thing this function exists to avoid.
    """
    by_path = {file.path: file for file in files}
    results: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for match in _CITATION.finditer(answer):
        cited = match.group("path")
        line = int(match.group("start"))
        if (cited, line) in seen:
            continue
        seen.add((cited, line))

        file = by_path.get(cited)
        if file is None:
            candidates = [path for path in by_path if path.endswith("/" + cited)]
            if len(candidates) == 1:
                file = by_path[candidates[0]]
            elif len(candidates) > 1:
                results.append(Citation(cited, line, False, "ambiguous: several files match"))
                continue

        if file is None:
            results.append(Citation(cited, line, False, "not a file in the reviewed diff"))
        elif line not in file.citable_lines():
            results.append(Citation(cited, line, False, "that line was not in the diff shown"))
        else:
            results.append(Citation(cited, line, True))

    return results


def unverified_footer(citations: list[Citation]) -> str:
    """The note appended to a review when some citations did not check out. Empty if none."""
    bad = [citation for citation in citations if not citation.verified]
    if not bad:
        return ""
    lines = ["", "Citations that could not be verified against the diff:"]
    lines.extend(f"  {c.path}:{c.line} — {c.reason}" for c in bad)
    return "\n".join(lines)
