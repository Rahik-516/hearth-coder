"""Applying a text edit exactly, or refusing to apply it at all.

The engine's job is to make a model's approximate idea of a file's contents land exactly
(docs/safety-and-tool-use.md §7.2). The failure to design against is not a refusal — it is
a *near miss*: an edit that matches the wrong region, where the preview the user approved
was computed from the same wrong match and therefore looks entirely reasonable. Every
choice below is biased toward refusing with a useful hint instead of guessing.

Three things are worth knowing before changing anything here.

**Matching happens in LF space.** :meth:`FileForm.decode` normalises line endings and
strips any BOM, so the text the engine and the model both see is plain ``\\n``. A model
cannot know whether a file is CRLF, and requiring it to guess would make every
Windows-authored file unmatchable. The original form is carried alongside and reapplied on
the way out, so the bytes written differ from the bytes read only where the edit says.

**The round trip is lossless for a consistently-terminated file, and deliberately not for
a mixed one.** A file that is half CRLF and half LF is normalised to its dominant ending,
so a one-line edit to it rewrites every line. That is a real consequence, accepted because
the alternative — tracking each line's ending individually — preserves a mess that is
almost always accidental.

**Strategies run in order and stop at the first unique match**: exact, then normalised
(trailing whitespace), then indentation-insensitive. Anything past exact earns the
``FUZZY-MATCH`` badge, because the user needs to know the engine interpreted rather than
found.

One narrowing from §7.2 worth stating: **UTF-16 is refused, not supported.** It arrives as
bytes containing NUL, which the binary sniffer catches, and a half-implemented UTF-16 path
that silently wrote UTF-8 back would corrupt the file. Refusing is the honest behaviour
until a real need appears.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass

from hearth.util.text import dominant_line_ending, is_probably_binary

#: `write_file` refuses content larger than this (§7.2 item 3).
MAX_WRITE_BYTES = 2 * 1024 * 1024

#: Changed lines above which a diff carries the `LARGE` badge (§6.2).
LARGE_DIFF_LINES = 300

#: How many lines of surrounding context a "closest region" hint shows.
_HINT_CONTEXT = 2

#: Below this similarity a "closest region" is not worth showing — a hint pointing at
#: unrelated code is worse than no hint, because the model will try to reconcile it.
_HINT_MIN_RATIO = 0.5

_BOM = "﻿"


@dataclass(frozen=True)
class FileForm:
    """How a file is encoded on disk, so an edit can put it back the same way.

    Frozen and made only of values, so it can be carried in ``Prepared.payload`` across
    the approval round trip and still describe the file that was previewed.
    """

    encoding: str = "utf-8"
    line_ending: str = "\n"
    had_trailing_newline: bool = True
    usable: bool = True
    reason: str = ""

    def decode(self, data: bytes) -> str:
        """The file's text, with LF line endings and no BOM."""
        text = data.decode(self.encoding)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.removeprefix(_BOM)


def read_form(data: bytes) -> FileForm:
    """Inspect a file's bytes, or report why it cannot be edited as text.

    Refusal is a first-class outcome rather than an exception: the caller turns it into a
    ``ToolResult`` the model can read, and "this is a binary file" is ordinary information
    rather than a fault.
    """
    if is_probably_binary(data):
        return FileForm(usable=False, reason="file looks binary (contains NUL bytes)")

    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        # Deliberately *not* falling back to latin-1. `util.text.decode_text` replaces
        # undecodable bytes, which is right for indexing and catastrophic for editing:
        # the write-back would not be lossless, so the file would quietly change in
        # places the diff never mentioned.
        return FileForm(usable=False, reason=f"file is not valid UTF-8 ({exc.reason})")

    return FileForm(
        encoding=encoding,
        line_ending=dominant_line_ending(text),
        had_trailing_newline=text.endswith(("\n", "\r")),
    )


def encode_with_form(text: str, form: FileForm) -> bytes:
    """Turn LF text back into the file's own form."""
    if form.line_ending != "\n":
        text = text.replace("\n", form.line_ending)
    if form.encoding == "utf-8-sig":
        text = _BOM + text
    return text.encode("utf-8")


@dataclass(frozen=True)
class EditOutcome:
    """What an edit would do, or why it cannot be done."""

    ok: bool
    new_text: str | None = None
    #: exact | normalized | indent | none
    strategy: str = "none"
    badges: tuple[str, ...] = ()
    #: How many occurrences the winning strategy found.
    replacements: int = 0
    #: A sentence for the model. Present whenever ``ok`` is False.
    error: str | None = None
    #: Concrete detail that makes the next attempt a correction, not another guess.
    hint: str | None = None


def apply_edit(
    text: str, old: str, new: str, *, replace_all: bool = False, language: str | None = None
) -> EditOutcome:
    """Compute the result of replacing ``old`` with ``new`` in ``text``.

    Pure: takes and returns text, touches no filesystem. That is what lets ``prepare()``
    produce the exact preview ``execute()`` will write.

    Args:
        text: The file's current content, LF-normalised.
        old: The text to replace. Must match exactly once unless ``replace_all``.
        new: The replacement.
        replace_all: Replace every occurrence the winning strategy finds.
        language: Used for the parse guard, when the file's language is known.
    """
    if not old:
        return EditOutcome(
            ok=False,
            error="old_string is empty. Give the exact text to replace, with enough "
            "surrounding context to be unique.",
        )

    if old == new:
        return EditOutcome(
            ok=False,
            error="old_string and new_string are identical, so this edit would change "
            "nothing. Did you mean to change something else?",
        )

    for name, strategy in _STRATEGIES:
        spans = strategy(text, old)
        if not spans:
            continue

        if len(spans) > 1 and not replace_all:
            return _ambiguous(text, spans, name)

        new_text = _splice(text, spans, old, new, strategy=name, replace_all=replace_all)
        badges = _badges(text, new_text, strategy=name, language=language)
        return EditOutcome(
            ok=True,
            new_text=new_text,
            strategy=name,
            badges=badges,
            replacements=len(spans),
        )

    return _not_found(text, old)


def diff_stats(before: str, after: str) -> tuple[int, int]:
    """Added and removed line counts, for the approval header's `+4 -2`."""
    added = removed = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def unified_diff(before: str, after: str, *, path: str, context: int = 3) -> str:
    """The diff shown in the approval panel."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context,
        )
    )


# ----------------------------------------------------------------- strategies
#
# Each returns the character spans it matched, in order. An empty list means "this
# strategy found nothing"; the caller moves on to the next one.
#
# Names are declared in the table at the bottom rather than derived from the function
# names. Deriving them silently coupled the `indent` re-indentation branch to the spelling
# of `_match_indented`, which produced "indented" and skipped the re-indent entirely.


def _match_exact(text: str, old: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = text.find(old)
    while start != -1:
        spans.append((start, start + len(old)))
        start = text.find(old, start + len(old))
    return spans


def _match_normalized(text: str, old: str) -> list[tuple[int, int]]:
    """Ignore trailing whitespace, on both sides.

    Line-oriented because trailing whitespace is a line-level phenomenon: a model that
    reproduced a line correctly but dropped the two spaces someone left at its end has
    identified the right line.
    """
    return _match_lines(text, old, key=lambda line: line.rstrip())


def _match_indented(text: str, old: str) -> list[tuple[int, int]]:
    """Ignore absolute indentation, but require the relative shape to match.

    Requiring the *shape* is what keeps this from being reckless: a two-line needle still
    has to find two lines whose contents match and whose relative indentation agrees, so
    it cannot land inside an unrelated block that happens to share one line.
    """
    return _match_lines(text, old, key=lambda line: line.strip(), check_shape=True)


def _match_lines(
    text: str,
    old: str,
    *,
    key: Callable[[str], str],
    check_shape: bool = False,
) -> list[tuple[int, int]]:
    """Slide a window of the needle's lines over the haystack under an equivalence."""
    hay_lines = text.split("\n")
    needle_lines = old.split("\n")
    # A needle ending in a newline splits to a trailing "", which would demand the
    # haystack have an empty line there too.
    if needle_lines and needle_lines[-1] == "":
        needle_lines = needle_lines[:-1]
    if not needle_lines or len(needle_lines) > len(hay_lines):
        return []

    needle_keys = [key(line) for line in needle_lines]
    offsets = _line_offsets(hay_lines)
    spans: list[tuple[int, int]] = []

    for index in range(len(hay_lines) - len(needle_lines) + 1):
        window = hay_lines[index : index + len(needle_lines)]
        if [key(line) for line in window] != needle_keys:
            continue
        if check_shape and not _same_shape(window, needle_lines):
            continue
        start = offsets[index]
        end = offsets[index + len(needle_lines) - 1] + len(window[-1])
        spans.append((start, end))

    return spans


def _same_shape(window: list[str], needle: list[str]) -> bool:
    """Whether two blocks indent their lines the same way relative to their first line."""
    window_base = len(_indent_of(window[0]))
    needle_base = len(_indent_of(needle[0]))
    return [len(_indent_of(line)) - window_base for line in window] == [
        len(_indent_of(line)) - needle_base for line in needle
    ]


# ------------------------------------------------------------------- splicing


def _splice(
    text: str,
    spans: list[tuple[int, int]],
    old: str,
    new: str,
    *,
    strategy: str,
    replace_all: bool,
) -> str:
    targets = spans if replace_all else spans[:1]
    result = text
    # Back to front, so an earlier replacement cannot move a later span.
    for start, end in reversed(targets):
        replacement = new
        if strategy == "indent":
            replacement = _reindent(new, old, matched=text[start:end])
        result = result[:start] + replacement + result[end:]
    return result


def _reindent(new: str, old: str, *, matched: str) -> str:
    """Re-indent ``new`` from the needle's indentation to the file's.

    The subtlety this handles: the replacement must land at the *file's* indent depth, not
    the model's. Pasting a correctly-matched block at the wrong depth produces
    syntactically broken Python from a match that was otherwise right.

    Rebuilding each line's indent as "the file's base indent, plus whatever this line had
    beyond the needle's base indent" keeps the file's own indent character, so a
    tab-indented file does not acquire spaces.
    """
    old_base = _indent_of(old.split("\n")[0])
    hay_base = _indent_of(matched.split("\n")[0])

    rebuilt: list[str] = []
    for line in new.split("\n"):
        if not line.strip():
            rebuilt.append("")  # never indent a blank line
            continue
        indent = _indent_of(line)
        extra = indent[len(old_base) :] if indent.startswith(old_base) else ""
        rebuilt.append(hay_base + extra + line[len(indent) :])
    return "\n".join(rebuilt)


# -------------------------------------------------------------------- failures


def _ambiguous(text: str, spans: list[tuple[int, int]], strategy: str) -> EditOutcome:
    lines = sorted(_line_number(text, start) for start, _ in spans)
    return EditOutcome(
        ok=False,
        strategy=strategy,
        replacements=len(spans),
        error=(
            f"old_string matches {len(spans)} places, so it is ambiguous. Add surrounding "
            "context to make it unique, or pass replace_all=true to change every one."
        ),
        hint="matches at lines " + ", ".join(str(number) for number in lines),
    )


def _not_found(text: str, old: str) -> EditOutcome:
    """Zero matches — and the one case where a hint changes the model's next move.

    Without it the model can only guess again. With the actual text and its line numbers,
    the next attempt is a correction.
    """
    return EditOutcome(
        ok=False,
        replacements=0,
        error="old_string was not found in the file. Read the file again and copy the "
        "exact text, including indentation.",
        hint=_closest_region(text, old),
    )


def _closest_region(text: str, old: str) -> str | None:
    """The most similar block of the file, with line numbers."""
    hay_lines = text.split("\n")
    needle_lines = [line for line in old.split("\n") if line.strip()]
    if not needle_lines or not hay_lines:
        return None

    width = len(needle_lines)
    best_ratio = 0.0
    best_index = -1
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2("\n".join(line.strip() for line in needle_lines))

    for index in range(max(1, len(hay_lines) - width + 1)):
        window = hay_lines[index : index + width]
        matcher.set_seq1("\n".join(line.strip() for line in window))
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_index = ratio, index

    if best_index < 0 or best_ratio < _HINT_MIN_RATIO:
        return None

    start = max(0, best_index - _HINT_CONTEXT)
    end = min(len(hay_lines), best_index + width + _HINT_CONTEXT)
    numbered = "\n".join(f"{number + 1:>5}| {hay_lines[number]}" for number in range(start, end))
    return f"closest match is around line {best_index + 1}:\n{numbered}"


# --------------------------------------------------------------------- badges


def _badges(text: str, new_text: str, *, strategy: str, language: str | None) -> tuple[str, ...]:
    badges: list[str] = []
    if strategy != "exact":
        badges.append("FUZZY-MATCH")

    added, removed = diff_stats(text, new_text)
    if added + removed > LARGE_DIFF_LINES:
        badges.append("LARGE")

    if language and _introduces_parse_errors(text, new_text, language):
        badges.append("PARSE-ERRORS-INTRODUCED")

    return tuple(badges)


def _introduces_parse_errors(before: str, after: str, language: str) -> bool:
    """Whether the edit increased the count of tree-sitter ERROR/MISSING nodes.

    Compared rather than absolute, because plenty of real files do not parse cleanly to
    begin with — a vendored snippet, an unsupported syntax version — and refusing to edit
    those would be worse than useless. What matters is whether *this* edit made it worse.

    Imported lazily and failing open: an unavailable grammar must not block an edit, it
    just means this particular signal is unavailable (docs/system-design.md §10.1).
    """
    try:
        from hearth.indexing.parser import count_error_nodes, parse
    except ImportError:  # pragma: no cover - tree-sitter is a hard dependency
        return False

    try:
        before_tree = parse(before.encode(), language)
        after_tree = parse(after.encode(), language)
    except Exception:
        return False

    if before_tree.root is None or after_tree.root is None:
        return False
    return count_error_nodes(after_tree.root) > count_error_nodes(before_tree.root)


# ------------------------------------------------------------------ small bits


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


#: Strategy order, which is the §7.2 order. The first one to find a unique match wins, so
#: moving a row here changes which interpretation of an ambiguous edit is applied.
_STRATEGIES: tuple[tuple[str, Callable[[str, str], list[tuple[int, int]]]], ...] = (
    ("exact", _match_exact),
    ("normalized", _match_normalized),
    ("indent", _match_indented),
)
