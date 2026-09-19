"""Text handling: identifier splitting, encoding and line-ending detection.

Identifier splitting is what makes lexical search work on code. A developer searching for
"invoice service" must find ``InvoiceService``, and one searching for "stripe webhook
secret" must find ``STRIPE_WEBHOOK_SECRET``. FTS5 tokenizes on word boundaries, so without
this step neither query matches anything (docs/system-design.md §6.5).

The same splitting is applied to queries, so the two sides agree.
"""

from __future__ import annotations

import re
import unicodedata

#: Split a run of word characters into camel/Pascal/acronym parts.
#:
#: The first alternative handles acronyms followed by a word: HTTPServer -> HTTP, Server.
#: Without it, a greedy [A-Z][a-z]* would yield H, T, T, P, Server.
_WORD_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

#: Characters that separate identifiers in source code.
_SEPARATORS = re.compile(r"[^0-9A-Za-z_]+")

#: A token worth indexing on its own. One-character fragments are noise.
_MIN_PART_LEN = 2

_BOM_UTF8 = b"\xef\xbb\xbf"


def split_identifier(identifier: str) -> list[str]:
    """Split one identifier into lowercase parts.

    ``InvoiceService`` -> ``["invoice", "service"]``;
    ``STRIPE_WEBHOOK_SECRET`` -> ``["stripe", "webhook", "secret"]``;
    ``parseURLQuery`` -> ``["parse", "url", "query"]``.
    """
    parts: list[str] = []
    for word in identifier.split("_"):
        parts.extend(match.group(0).lower() for match in _WORD_PARTS.finditer(word))
    return [p for p in parts if len(p) >= _MIN_PART_LEN or p.isdigit()]


def expand_identifier(identifier: str) -> list[str]:
    """Every searchable form of one identifier, de-duplicated and order-preserving.

    Includes the split parts plus the whole identifier lowercased, so both
    ``invoice service`` and ``invoiceservice`` find ``InvoiceService``, and
    ``stripe_webhook_secret`` still matches itself verbatim.
    """
    forms: list[str] = []
    lowered = identifier.lower()

    for part in split_identifier(identifier):
        if part not in forms:
            forms.append(part)

    # The whole thing, as written and with separators stripped. Only worth adding when it
    # differs from the parts already present.
    for whole in (lowered, lowered.replace("_", "")):
        if whole and whole not in forms:
            forms.append(whole)
    return forms


def extract_identifiers(text: str) -> list[str]:
    """Pull identifier-shaped tokens out of a blob of code, in order of appearance."""
    seen: list[str] = []
    for raw in _SEPARATORS.split(text):
        if raw and not raw.isdigit() and raw not in seen:
            seen.append(raw)
    return seen


def build_search_text(text: str) -> str:
    """The FTS5 ``search_text`` value: the original plus expanded identifier forms.

    Keeping the original means exact-phrase queries still work; appending the expansions
    means a split query matches too.
    """
    expansions: list[str] = []
    for identifier in extract_identifiers(text):
        for form in expand_identifier(identifier):
            if form not in expansions:
                expansions.append(form)

    if not expansions:
        return text
    return f"{text}\n{' '.join(expansions)}"


def build_search_query(query: str) -> list[str]:
    """Expand a user query the same way documents were expanded."""
    terms: list[str] = []
    for identifier in extract_identifiers(query):
        for form in expand_identifier(identifier):
            if form not in terms:
                terms.append(form)
    return terms


# ------------------------------------------------------------------ file text


def detect_encoding(data: bytes) -> str:
    """Best-effort text encoding. Only UTF-8 family is supported; else latin-1 fallback."""
    if data.startswith(_BOM_UTF8):
        return "utf-8-sig"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8"


def decode_text(data: bytes) -> str:
    """Decode file bytes, never raising. Undecodable bytes are replaced."""
    encoding = detect_encoding(data)
    return data.decode(encoding, errors="replace")


def dominant_line_ending(text: str) -> str:
    """The file's prevailing line ending, so edits can preserve it.

    Counts CRLF first, then bare LF, because every CRLF contains an LF.
    """
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf

    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if cr > lf and cr > 0:
        return "\r"
    return "\n"


def is_probably_binary(data: bytes, *, sniff_bytes: int = 8192) -> bool:
    """A NUL byte in the first 8 KB means binary (docs/system-design.md §6.2)."""
    return b"\x00" in data[:sniff_bytes]


def normalize_for_hash(text: str) -> str:
    """Normalize Unicode so visually identical text hashes identically."""
    return unicodedata.normalize("NFC", text)


def count_lines(text: str) -> int:
    """Number of lines, counting a final unterminated line."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def average_line_length(text: str) -> float:
    """Mean line length. A very high value suggests minified or generated output."""
    lines = text.splitlines() or [""]
    return sum(len(line) for line in lines) / len(lines)
