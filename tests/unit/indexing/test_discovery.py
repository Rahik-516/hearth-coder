"""Language detection, filtering, scanning and change detection.

The secret-file tests are an M1 acceptance criterion and are asserted against the
`malicious` fixture. The change-detection tests back the other criterion: re-indexing an
unchanged tree must do no work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.indexing.change_detector import detect_changes
from hearth.indexing.filters import (
    SECRET_FILE_PATTERNS,
    PathFilter,
    is_too_large,
    load_ignore_file,
    looks_generated,
)
from hearth.indexing.languages import (
    SupportLevel,
    detect_language,
    is_parseable,
    support_level,
)
from hearth.indexing.scanner import ScannedFile, scan

MALICIOUS = Path(__file__).resolve().parents[2] / "fixtures" / "repos" / "malicious"
PY_SMALL = Path(__file__).resolve().parents[2] / "fixtures" / "repos" / "py_small"


# ------------------------------------------------------------------- languages


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/a.py", "python"),
        ("src/a.pyi", "python"),
        ("src/app.ts", "typescript"),
        ("src/app.tsx", "tsx"),
        ("src/app.js", "javascript"),
        ("src/app.jsx", "javascript"),
        ("README.md", "markdown"),
        ("data.json", "json"),
        ("Dockerfile", "dockerfile"),
        ("Dockerfile.prod", "dockerfile"),
        ("Makefile", "make"),
        ("pyproject.toml", "toml"),
        ("mystery.zzz", None),
    ],
)
def test_detect_language(path: str, language: str | None) -> None:
    assert detect_language(path) == language


@pytest.mark.parametrize(
    ("shebang", "language"),
    [
        ("#!/usr/bin/env python3", "python"),
        ("#!/usr/bin/python", "python"),
        ("#!/bin/bash", "bash"),
        ("#!/usr/bin/env node", "javascript"),
        ("not a shebang", None),
    ],
)
def test_detect_language_from_shebang(shebang: str, language: str | None) -> None:
    assert detect_language("script", first_line=shebang) == language


def test_support_levels_match_the_mvp_scope() -> None:
    for language in ("python", "typescript", "tsx", "javascript"):
        assert support_level(language) is SupportLevel.FULL
        assert is_parseable(language)

    assert support_level("markdown") is SupportLevel.DOCUMENT
    assert support_level("json") is SupportLevel.DATA
    assert support_level(None) is SupportLevel.FALLBACK


def test_languages_without_a_bundled_grammar_degrade() -> None:
    """Go is FULL by design but has no grammar in this build, so it must not claim FULL."""
    assert support_level("go") is SupportLevel.FALLBACK
    assert is_parseable("go") is False


# --------------------------------------------------------------------- filters


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        ".env.local",
        "server.pem",
        "certs/private.key",
        "id_rsa",
        "id_ed25519",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "terraform.tfstate",
        "config/credentials",
        "deploy/service-account-prod.json",
    ],
)
def test_secret_files_are_excluded(path: str) -> None:
    assert PathFilter().decide(path).reason == "secret-file"


def test_secret_exclusion_cannot_be_overridden_by_include() -> None:
    """An index.include entry must not be able to pull a .env into the index.

    A credential reaching the index would flow into retrieval and from there into a
    prompt. That is one-way, so it is an invariant rather than a default.
    """
    permissive = PathFilter(include=["**"], exclude=[], respect_defaults=False)

    assert permissive.decide(".env").include is False
    assert permissive.decide("server.pem").include is False


def test_ordinary_files_are_kept() -> None:
    filter_ = PathFilter()
    assert filter_.decide("src/billing/invoice_service.py").include is True
    assert filter_.decide("README.md").include is True


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/react/index.js",
        ".git/config",
        "src/__pycache__/mod.cpython-312.pyc",
        ".venv/lib/python3.12/site-packages/x.py",
        "dist/bundle.js",
        "target/debug/build.rs",
    ],
)
def test_default_ignored_directories(path: str) -> None:
    assert PathFilter().decide(path).include is False


@pytest.mark.parametrize(
    "path",
    ["app.min.js", "styles.min.css", "bundle.js.map", "uv.lock", "logo.png", "archive.tar.gz"],
)
def test_default_ignored_globs(path: str) -> None:
    assert PathFilter().decide(path).include is False


def test_gitignore_is_respected() -> None:
    spec = load_ignore_file("build/\n*.tmp\n")
    filter_ = PathFilter(gitignore=spec)

    assert filter_.decide("build/out.js").include is False
    assert filter_.decide("scratch.tmp").include is False
    assert filter_.decide("src/main.py").include is True


def test_config_exclude_beats_include() -> None:
    filter_ = PathFilter(exclude=["data/**"], include=["**"])
    assert filter_.decide("data/big.csv").include is False


def test_explicit_include_overrides_gitignore() -> None:
    """This is what lets a project deliberately index a vendored subtree."""
    filter_ = PathFilter(gitignore=load_ignore_file("vendor/\n"), include=["vendor/important/**"])

    assert filter_.decide("vendor/important/lib.py").include is True
    assert filter_.decide("vendor/other/lib.py").include is False


def test_size_limit() -> None:
    assert is_too_large(2_000_000, max_bytes=1_000_000) is True
    assert is_too_large(500, max_bytes=1_000_000) is False


@pytest.mark.parametrize(
    "text",
    [
        "# @generated by tooling\nx = 1",
        "// Code generated by protoc. DO NOT EDIT.\n",
        "/* autogenerated */\n",
    ],
)
def test_generated_markers_detected(text: str) -> None:
    assert looks_generated(text) is True


def test_minified_shape_detected_by_line_length() -> None:
    assert looks_generated("\n".join(["x" * 400] * 10)) is True


def test_ordinary_source_is_not_generated() -> None:
    assert looks_generated("def add(a, b):\n    return a + b\n") is False


def test_secret_patterns_cover_the_documented_set() -> None:
    """docs/system-design.md §6.2 names these explicitly."""
    joined = " ".join(SECRET_FILE_PATTERNS)
    for required in (".env", "*.pem", "*.key", "id_rsa*", "*.p12", ".npmrc", ".pypirc", "*.tfstate"):
        assert required in joined


# --------------------------------------------------------------------- scanner


def test_scan_finds_source_and_skips_secrets() -> None:
    """M1 acceptance: secret files never reach the index, asserted on `malicious`."""
    result = scan(MALICIOUS)
    found = {f.relative_path for f in result.files}

    assert "src/safe.py" in found
    assert "src/helpers.py" in found

    for secret in (".env", ".env.production", "server.pem", "id_rsa", ".npmrc", "terraform.tfstate"):
        assert secret not in found, f"{secret} must never be scanned"

    assert result.skipped.get("secret-file", 0) >= 6


def test_scan_reports_stat_data(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = scan(tmp_path)

    assert len(result.files) == 1
    assert result.files[0].size_bytes == 6
    assert result.files[0].mtime_ns > 0


def test_scan_prunes_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1", encoding="utf-8")

    found = {f.relative_path for f in scan(tmp_path).files}

    assert found == {"src/main.py"}


def test_scan_results_are_sorted(tmp_path: Path) -> None:
    for name in ("c.py", "a.py", "b.py"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    paths = [f.relative_path for f in scan(tmp_path).files]

    assert paths == sorted(paths)


# ------------------------------------------------------------- change detection


def scanned(path: str, size: int, mtime: int, tmp_path: Path) -> ScannedFile:
    absolute = tmp_path / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if not absolute.exists():
        absolute.write_text("x" * size, encoding="utf-8")
    return ScannedFile(relative_path=path, absolute_path=absolute, size_bytes=size, mtime_ns=mtime)


def test_unchanged_tree_needs_no_work(tmp_path: Path) -> None:
    """M1 acceptance: re-running index with no changes performs zero parses."""
    files = [scanned("a.py", 10, 111, tmp_path), scanned("b.py", 20, 222, tmp_path)]
    indexed = {"a.py": (10, 111, "hash-a"), "b.py": (20, 222, "hash-b")}

    changes = detect_changes(files, indexed)

    assert changes.needs_indexing == []
    assert changes.has_changes is False
    assert changes.hashed == 0, "the stat fast path must avoid opening files"
    assert len(changes.unchanged) == 2


def test_new_file_is_added(tmp_path: Path) -> None:
    changes = detect_changes([scanned("new.py", 5, 1, tmp_path)], {})
    assert [f.relative_path for f in changes.added] == ["new.py"]


def test_deleted_file_is_detected(tmp_path: Path) -> None:
    changes = detect_changes([], {"gone.py": (1, 1, "h")})
    assert changes.deleted == ["gone.py"]


def test_touched_but_unchanged_file_is_not_reindexed(tmp_path: Path) -> None:
    """A checkout rewriting identical bytes must not trigger a reindex of the world."""
    path = tmp_path / "a.py"
    path.write_text("stable content", encoding="utf-8")
    from hearth.util.hashing import file_hash

    real_hash = file_hash(path)
    file = ScannedFile("a.py", path, path.stat().st_size, 999_999)
    indexed = {"a.py": (path.stat().st_size, 111, real_hash)}

    changes = detect_changes([file], indexed)

    assert changes.needs_indexing == []
    assert changes.hashed == 1, "stat differed, so it had to hash"
    assert changes.stat_changed_content_same == 1


def test_genuinely_modified_file_is_reindexed(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("new content", encoding="utf-8")
    file = ScannedFile("a.py", path, path.stat().st_size, 999)
    indexed = {"a.py": (999, 111, "a-stale-hash")}

    changes = detect_changes([file], indexed)

    assert [f.relative_path for f in changes.modified] == ["a.py"]


def test_force_reindexes_everything(tmp_path: Path) -> None:
    files = [scanned("a.py", 10, 111, tmp_path)]
    indexed = {"a.py": (10, 111, "hash-a")}

    changes = detect_changes(files, indexed, force=True)

    assert len(changes.modified) == 1
    assert changes.hashed == 0, "force should not bother hashing"
