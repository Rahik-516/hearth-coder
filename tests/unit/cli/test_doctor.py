"""Doctor checks.

The two that earn their place are the WSL ones: a repo under /mnt/<drive> and NAT-mode
networking are the failure modes the roadmap names explicitly, and both are silent
otherwise — things just feel slow or refuse to connect for no visible reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.cli import doctor as doc
from hearth.cli.doctor import (
    CheckResult,
    DoctorReport,
    Status,
    check_sqlite_fts5,
    check_workspace_location,
    check_wsl_networking,
    format_report,
    run_doctor,
)
from hearth.config.loader import load_config
from hearth.llm.scripted_provider import ScriptedProvider
from hearth.llm.types import ModelInfo, RunningModel


@pytest.fixture
def loaded(tmp_path: Path):
    return load_config(global_config_path=tmp_path / "absent.toml")


# ------------------------------------------------------------------- environment


def test_fts5_is_available_in_this_build() -> None:
    """If this fails, lexical search cannot work at all."""
    assert check_sqlite_fts5().status is Status.PASS


def test_report_exit_code_reflects_failures() -> None:
    report = DoctorReport()
    report.add(CheckResult("a", Status.PASS))
    assert report.exit_code == 0

    report.add(CheckResult("b", Status.WARN))
    assert report.exit_code == 0, "warnings must not fail the run"

    report.add(CheckResult("c", Status.FAIL))
    assert report.exit_code == 1


def test_format_report_includes_fixes_for_problems_only() -> None:
    report = DoctorReport()
    report.add(CheckResult("fine", Status.PASS, "ok", fix="should not appear"))
    report.add(CheckResult("broken", Status.FAIL, "bad", fix="do this"))

    text = format_report(report)

    assert "do this" in text
    assert "should not appear" not in text


# --------------------------------------------------------------------------- WSL


def test_workspace_in_wsl_filesystem_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A repo on the Linux filesystem is the layout we want — no warning."""
    monkeypatch.setattr(doc, "is_wsl", lambda: True)

    assert check_workspace_location(tmp_path).status is Status.PASS


def test_workspace_under_mnt_drive_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doc, "is_wsl", lambda: True)

    result = check_workspace_location(Path("/mnt/c/Users/dev/project"))

    assert result.status is Status.WARN
    assert "/mnt/c" in result.detail
    assert result.fix is not None
    assert "~/code" in result.fix


def test_workspace_check_ignores_mnt_outside_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    """On native Linux, /mnt is just a directory and means nothing."""
    monkeypatch.setattr(doc, "is_wsl", lambda: False)

    assert check_workspace_location(Path("/mnt/c/project")).status is Status.PASS


def test_multi_character_mount_is_not_a_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    """/mnt/data is an ordinary mount, not a Windows drive letter."""
    monkeypatch.setattr(doc, "is_wsl", lambda: True)

    assert check_workspace_location(Path("/mnt/data/project")).status is Status.PASS


def test_nat_mode_networking_fails_with_a_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The WSL2 NAT case: Windows host on a private IP, which the guard refuses."""
    monkeypatch.setattr(doc, "is_wsl", lambda: True)

    result = check_wsl_networking("http://172.30.112.1:11434")

    assert result.status is Status.FAIL
    assert result.fix is not None
    assert "mirrored" in result.fix


def test_loopback_under_wsl_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doc, "is_wsl", lambda: True)

    assert check_wsl_networking("http://127.0.0.1:11434").status is Status.PASS


# ------------------------------------------------------------------------ ollama


async def test_doctor_reports_model_capabilities(loaded, tmp_path: Path) -> None:
    provider = ScriptedProvider(
        models={
            "qwen3.5:4b": ModelInfo(
                name="qwen3.5:4b",
                parameter_size="4.0B",
                quantization="Q4_K_M",
                capabilities=["completion", "tools", "thinking"],
            ),
            "qwen3-embedding:0.6b": ModelInfo(
                name="qwen3-embedding:0.6b",
                capabilities=["embedding"],
            ),
        }
    )

    report = await run_doctor(loaded=loaded, workspace=tmp_path, provider=provider)
    by_name = {r.name: r for r in report.results}

    assert by_name["Chat model"].status is Status.PASS
    assert "4.0B" in by_name["Chat model"].detail
    assert by_name["Embedding model"].status is Status.PASS


async def test_chat_model_without_tools_capability_fails(loaded, tmp_path: Path) -> None:
    """Agent mode needs tool calling; discovering that at runtime is far worse."""
    provider = ScriptedProvider(
        models={
            "qwen3.5:4b": ModelInfo(name="qwen3.5:4b", capabilities=["completion"]),
            "qwen3-embedding:0.6b": ModelInfo(name="qwen3-embedding:0.6b", capabilities=["embedding"]),
        }
    )

    report = await run_doctor(loaded=loaded, workspace=tmp_path, provider=provider)
    chat = next(r for r in report.results if r.name == "Chat model")

    assert chat.status is Status.FAIL
    assert "tools" in chat.detail


async def test_missing_model_suggests_pull(loaded, tmp_path: Path) -> None:
    provider = ScriptedProvider(models={"something-else:1b": ModelInfo(name="something-else:1b")})

    report = await run_doctor(loaded=loaded, workspace=tmp_path, provider=provider)
    chat = next(r for r in report.results if r.name == "Chat model")

    assert chat.status is Status.FAIL
    assert chat.fix is not None
    assert "ollama pull" in chat.fix


async def test_partially_offloaded_model_warns(loaded, tmp_path: Path, monkeypatch) -> None:
    """A model split to CPU is the usual explanation for a session feeling slow."""
    provider = ScriptedProvider()

    async def running() -> list[RunningModel]:
        return [RunningModel(name="qwen3.5:9b", size=1000, size_vram=400)]

    monkeypatch.setattr(provider, "running", running)

    report = await run_doctor(loaded=loaded, workspace=tmp_path, provider=provider)
    entry = next(r for r in report.results if r.name.startswith("Loaded:"))

    assert entry.status is Status.WARN
    assert "40%" in entry.detail


async def test_doctor_runs_without_a_provider(loaded, tmp_path: Path) -> None:
    """--skip-ollama still reports everything that doesn't need a server."""
    report = await run_doctor(loaded=loaded, workspace=tmp_path, provider=None)

    names = {r.name for r in report.results}
    assert "SQLite FTS5" in names
    assert "Ollama" not in names


async def test_untrusted_allow_rules_are_surfaced(tmp_path: Path) -> None:
    """The user should learn why their project rules aren't taking effect."""
    root = tmp_path / "repo"
    (root / ".hearth").mkdir(parents=True)
    (root / ".hearth" / "config.toml").write_text(
        '[[permissions.allow]]\ntool = "run_command"\nargv = ["curl", "*"]\n',
        encoding="utf-8",
    )
    loaded = load_config(project_root=root, global_config_path=tmp_path / "absent.toml")

    report = await run_doctor(loaded=loaded, workspace=root, provider=None)
    trust = next(r for r in report.results if r.name == "Project trust")

    assert trust.status is Status.WARN
    assert trust.fix is not None
    assert "hearth trust" in trust.fix


# --------------------------------------------------------------- GPU placement advice


def test_a_full_card_is_told_to_shrink_the_model() -> None:
    """The ordinary case: the model did not fit, so make it smaller."""
    fix = doc._placement_fix(40, free_vram_mib=120)

    assert "smaller model" in fix


def test_an_idle_card_with_zero_percent_is_told_ollama_missed_the_gpu() -> None:
    """The observed failure on this machine, which the generic advice sends nowhere.

    Ollama's `llama-server --list-devices` crashes during discovery, so it registers no
    VRAM and runs on CPU while nvidia-smi reports the card healthy and idle. Telling that
    user to close GPU applications or pick a smaller model cannot help.
    """
    fix = doc._placement_fix(0, free_vram_mib=5996)

    assert "did not detect one" in fix
    assert "will not help" in fix
    assert "smaller model" not in fix


def test_unknown_vram_falls_back_to_the_generic_advice() -> None:
    """With no nvidia-smi there is nothing to distinguish the two causes."""
    assert "smaller model" in doc._placement_fix(0, free_vram_mib=None)


async def test_a_cpu_pinned_embedding_model_is_not_a_warning(loaded, tmp_path: Path, monkeypatch) -> None:
    """The embedding model sits on CPU by design; warning about it is noise with wrong advice."""
    provider = ScriptedProvider()

    async def running() -> list[RunningModel]:
        return [RunningModel(name="qwen3-embedding:0.6b", size=1000, size_vram=0)]

    monkeypatch.setattr(provider, "running", running)

    results = await doc._check_loaded_models(provider, cpu_pinned=frozenset({"qwen3-embedding:0.6b"}))

    assert results[0].status is Status.PASS
    assert "by design" in results[0].detail


async def test_a_chat_model_on_cpu_still_warns(loaded, tmp_path: Path, monkeypatch) -> None:
    """Only the pinned model is exempt — the chat model on CPU is the real problem."""
    provider = ScriptedProvider()

    async def running() -> list[RunningModel]:
        return [RunningModel(name="qwen3.5:4b", size=1000, size_vram=0)]

    monkeypatch.setattr(provider, "running", running)

    results = await doc._check_loaded_models(provider, cpu_pinned=frozenset({"qwen3-embedding:0.6b"}))

    assert results[0].status is Status.WARN


# ------------------------------------------------------------- language grammars


def test_grammar_check_lists_what_is_available() -> None:
    """A missing grammar is invisible from outside: the language still indexes, still
    searches, and silently has no symbols. doctor is where that becomes visible."""
    result = doc.check_grammars()

    assert "python" in result.detail
    assert result.status in (Status.PASS, Status.WARN)


def test_languages_hearth_ships_no_grammar_for_do_not_warn(monkeypatch) -> None:
    """A warning nobody can act on is noise, and noise is how a report stops being read.

    C, Ruby and the rest have no grammar in any install, so they are stated in the detail
    line and left at PASS. Only a grammar one `uv sync` away raises the status.
    """
    from hearth.indexing import languages

    present = frozenset(
        {"python", "typescript", "tsx", "javascript", "go", "rust", "java"}
    )
    monkeypatch.setattr(languages, "grammar_available", lambda: present)

    result = doc.check_grammars()

    assert result.status is Status.PASS
    assert "ruby" in result.detail
    assert result.fix is None


def test_a_missing_optional_grammar_warns_with_the_install_command(monkeypatch) -> None:
    monkeypatch.setattr(doc, "check_grammars", doc.check_grammars)
    from hearth.indexing import languages

    monkeypatch.setattr(languages, "grammar_available", lambda: frozenset({"python"}))

    result = doc.check_grammars()

    assert result.status is Status.WARN
    assert "go" in result.detail
    assert result.fix is not None
    assert "langs-extra" in result.fix


def test_document_and_data_languages_are_not_reported_missing(monkeypatch) -> None:
    """Markdown has no grammar by design; listing it would invent a problem."""
    from hearth.indexing import languages

    monkeypatch.setattr(languages, "grammar_available", lambda: frozenset({"python"}))

    result = doc.check_grammars()

    assert "markdown" not in result.detail
    assert "json" not in result.detail


def test_every_grammar_present_is_a_pass(monkeypatch) -> None:
    from hearth.indexing import languages

    every = {
        language
        for language in languages.known_languages()
        if languages.declared_support_level(language)
        in (languages.SupportLevel.FULL, languages.SupportLevel.STRUCTURAL)
    }
    monkeypatch.setattr(languages, "grammar_available", lambda: frozenset(every))

    assert doc.check_grammars().status is Status.PASS
