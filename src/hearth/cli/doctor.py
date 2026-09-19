"""Environment checks — ``hearth doctor``.

The acceptance bar from docs/implementation-roadmap.md Phase 0: a clear pass/warn/fail
report that guides a fresh machine to a working setup, and that correctly flags NAT-mode
WSL networking and a repository under ``/mnt/<drive>``.

Every check returns rather than raises, so one failure never hides the rest — a user with
three problems should see three problems, not discover them one run at a time.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sqlite3
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from hearth.config.loader import LoadedConfig
from hearth.config.schema import HearthConfig
from hearth.llm.errors import LLMError
from hearth.llm.guards import is_loopback_host
from hearth.llm.provider import LLMProvider


class Status(StrEnum):
    PASS = "pass"  # noqa: S105 — a check outcome, not a credential
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """One check's outcome. ``fix`` is the next action, not a restatement of the problem."""

    name: str
    status: Status
    detail: str = ""
    fix: str | None = None


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    @property
    def warned(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


# --------------------------------------------------------------- environment checks


def check_python() -> CheckResult:
    version = platform.python_version()
    if sys_version_tuple() < (3, 12):
        return CheckResult(
            "Python",
            Status.FAIL,
            f"{version} (3.12+ required)",
            fix="Install Python 3.12 or newer, then re-run `uv sync`.",
        )
    return CheckResult("Python", Status.PASS, version)


def sys_version_tuple() -> tuple[int, int]:
    import sys

    return sys.version_info[:2]


def check_sqlite_fts5() -> CheckResult:
    """FTS5 is not compiled into every SQLite build, and lexical search depends on it."""
    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return CheckResult(
            "SQLite FTS5",
            Status.FAIL,
            f"unavailable: {exc}",
            fix="Install a Python whose SQLite has FTS5 (most do). uv-managed CPython does.",
        )
    return CheckResult("SQLite FTS5", Status.PASS, f"available (SQLite {sqlite3.sqlite_version})")


def check_external_tool(name: str, *, required: bool, purpose: str, install: str) -> CheckResult:
    path = shutil.which(name)
    if path:
        return CheckResult(name, Status.PASS, path)
    return CheckResult(
        name,
        Status.FAIL if required else Status.WARN,
        f"not found — {purpose}",
        fix=install,
    )


def check_git() -> CheckResult:
    return check_external_tool(
        "git",
        required=True,
        purpose="used for file discovery and every git tool",
        install="Install git (apt install git).",
    )


def check_ripgrep() -> CheckResult:
    return check_external_tool(
        "rg",
        required=False,
        purpose="grep falls back to a slower pure-Python scan",
        install="Install ripgrep (apt install ripgrep) for faster search.",
    )


# ---------------------------------------------------------------------- WSL checks


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def check_workspace_location(root: Path) -> CheckResult:
    """A repo under /mnt/<drive> is a Windows filesystem seen through a translation layer.

    Indexing crawls it slowly, file watching is unreliable, and path-safety semantics stop
    matching what the policy engine assumes (docs/model-recommendations.md §0.3).
    """
    resolved = root.resolve()
    parts = resolved.parts
    on_windows_mount = len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1

    if not is_wsl():
        return CheckResult("Workspace location", Status.PASS, str(resolved))

    if on_windows_mount:
        drive = parts[2]
        return CheckResult(
            "Workspace location",
            Status.WARN,
            f"{resolved} is on the Windows filesystem (/mnt/{drive})",
            fix=(
                "Move the repository into the WSL filesystem, e.g. ~/code/<name>. "
                "Indexing is far faster there and file watching actually works."
            ),
        )
    return CheckResult("Workspace location", Status.PASS, f"{resolved} (WSL filesystem)")


def check_wsl_networking(host: str) -> CheckResult:
    """Under WSL2, a non-loopback Ollama host almost always means NAT-mode networking."""
    if not is_wsl():
        return CheckResult("WSL networking", Status.PASS, "not running under WSL")

    if is_loopback_host(host):
        # Only the shape of the host is verified here. Whether anything answers is the
        # Ollama check's job — claiming reachability from a string would be a lie the
        # user then has to debug.
        return CheckResult("WSL networking", Status.PASS, f"host {host} is loopback")

    return CheckResult(
        "WSL networking",
        Status.FAIL,
        f"Ollama host {host} is not loopback",
        fix=(
            "WSL is likely in NAT mode, where the Windows host is a private IP that Hearth "
            "refuses. Either enable mirrored networking (add "
            "networkingMode=mirrored under [wsl2] in %UserProfile%\\.wslconfig, then "
            "`wsl --shutdown`), or run Ollama inside WSL2."
        ),
    )


# ------------------------------------------------------------------- Ollama checks


async def check_ollama(
    provider: LLMProvider,
    config: HearthConfig,
) -> list[CheckResult]:
    """Reachability, version, and per-model capability checks.

    Returns early if the server is unreachable: every later check would fail for the same
    reason, and repeating it just buries the actual fix.
    """
    results: list[CheckResult] = []

    try:
        version = await provider.version()
    except LLMError as exc:
        results.append(
            CheckResult(
                "Ollama",
                Status.FAIL,
                str(exc),
                fix=(
                    "Start Ollama (`ollama serve`), and confirm "
                    f"ollama.host in your config matches it (currently {config.ollama.host})."
                ),
            )
        )
        return results

    results.append(CheckResult("Ollama", Status.PASS, f"version {version} at {config.ollama.host}"))

    if _version_tuple(version) < _version_tuple(config.ollama.min_version):
        results.append(
            CheckResult(
                "Ollama version",
                Status.WARN,
                f"{version} is below the configured minimum {config.ollama.min_version}",
                fix="Upgrade Ollama, or lower ollama.min_version if you have validated this build.",
            )
        )

    results.append(_check_host(config.ollama.host))

    results.append(await _check_model(provider, config.models.chat, needs="tools", label="Chat model"))
    results.append(
        await _check_model(provider, config.models.embed, needs="embedding", label="Embedding model")
    )
    # The embedding model runs on CPU on purpose unless the user asked for the GPU:
    # pinning it there is what stops a query embedding from evicting the chat model out
    # of a small card mid-session (docs/system-design.md §6.7).
    cpu_pinned = frozenset() if config.models.embed_placement == "gpu" else frozenset({config.models.embed})
    results.extend(
        await _check_loaded_models(provider, free_vram_mib=free_vram_mib(), cpu_pinned=cpu_pinned)
    )
    return results


def _check_host(host: str) -> CheckResult:
    """Report whether inference stays on this machine.

    A remote host is a WARN rather than a FAIL: reaching here means the user explicitly
    set ``allow_remote_host``, and overriding their stated choice would be presumptuous.
    Saying plainly what it costs them is not.
    """
    if is_loopback_host(host):
        return CheckResult("Ollama host", Status.PASS, host)

    return CheckResult(
        "Ollama host",
        Status.WARN,
        f"{host} is remote (allow_remote_host is set)",
        fix=(
            "Prompts and code leave this machine. Unset ollama.allow_remote_host to "
            "restore the offline guarantee."
        ),
    )


async def _check_model(provider: LLMProvider, model: str, *, needs: str, label: str) -> CheckResult:
    try:
        info = await provider.show(model)
    except LLMError as exc:
        return CheckResult(label, Status.FAIL, str(exc), fix=f"ollama pull {model}")

    if needs not in info.capabilities:
        has = ", ".join(info.capabilities) or "none"
        return CheckResult(
            label,
            Status.FAIL,
            f"{model} does not advertise the '{needs}' capability (has: {has})",
            fix=f"Choose a model with '{needs}' support — see docs/model-recommendations.md.",
        )

    detail = model
    if info.parameter_size:
        detail += f" ({info.parameter_size}"
        detail += f", {info.quantization})" if info.quantization else ")"
    return CheckResult(label, Status.PASS, detail)


async def _check_loaded_models(
    provider: LLMProvider,
    *,
    free_vram_mib: int | None = None,
    cpu_pinned: frozenset[str] = frozenset(),
) -> list[CheckResult]:
    """Report GPU placement for anything currently loaded.

    On a 6 GB card this is the check that explains why a session feels slow: a model split
    across CPU prefills many times slower (docs/model-recommendations.md §0.2).
    """
    try:
        running = await provider.running()
    except LLMError:
        return []

    results: list[CheckResult] = []
    for entry in running:
        fraction = entry.gpu_fraction
        if fraction is None:
            continue
        percent = round(fraction * 100)
        if entry.name in cpu_pinned and percent == 0:
            # Deliberate, so not a warning — and the generic advice ("close GPU
            # applications") would send the user to fix something that is working.
            results.append(
                CheckResult(f"Loaded: {entry.name}", Status.PASS, "on CPU by design (embed_placement)")
            )
        elif entry.fully_on_gpu:
            results.append(CheckResult(f"Loaded: {entry.name}", Status.PASS, f"{percent}% GPU"))
        else:
            results.append(
                CheckResult(
                    f"Loaded: {entry.name}",
                    Status.WARN,
                    f"only {percent}% on GPU — prefill will be slow",
                    fix=_placement_fix(percent, free_vram_mib),
                )
            )
    return results


def _placement_fix(percent: int, free_vram_mib: int | None) -> str:
    """What to actually do about a model that is not on the GPU.

    The two causes need opposite advice, and telling them apart is the point of this
    function. If the card is *full*, the model did not fit and the answer is to make it
    smaller. If the card is nearly empty and the model is on CPU anyway, nothing fit
    because Ollama never saw the GPU — and suggesting a smaller model sends the user down
    a dead end that cannot work.

    The observed form of the second case: Ollama's `llama-server --list-devices` crashes
    during discovery (exit 0xc0000005 on Windows) for every backend, so it registers
    `total_vram=0` and runs on CPU while `nvidia-smi` reports the card as healthy and idle.
    """
    plenty_free = free_vram_mib is not None and free_vram_mib >= _GPU_IDLE_MIB
    if percent == 0 and plenty_free:
        return (
            f"Ollama is not using the GPU at all, though {free_vram_mib} MiB are free — so "
            "it did not detect one. Check the Ollama server log for 'GPU discovery' errors, "
            "then reinstall or update Ollama and the GPU driver. Shrinking the model will "
            "not help."
        )
    return "Lower models.num_ctx, use a smaller model, or close other GPU applications."


#: Free VRAM above which "the model is on CPU" cannot be explained by a full card.
_GPU_IDLE_MIB = 2048


def free_vram_mib() -> int | None:
    """Free VRAM in MiB per nvidia-smi, or None when there is no NVIDIA GPU to ask."""
    reading = _nvidia_memory()
    return None if reading is None else reading[1]


def _nvidia_memory() -> tuple[int, int] | None:
    """``(total_mib, free_mib)`` from nvidia-smi, or None when there is no card to ask.

    Shared by the VRAM check and the placement advice, which need the same reading to
    answer different questions — how much is there, and whether "on CPU" is explained by
    the card being full.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None

    try:
        output = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [smi, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if output.returncode != 0 or not output.stdout.strip():
        return None

    first = output.stdout.strip().splitlines()[0]
    try:
        total_mb, free_mb = (int(part.strip()) for part in first.split(",")[:2])
    except ValueError:
        return None
    return total_mb, free_mb


def check_vram(config: HearthConfig) -> CheckResult | None:
    """Compare free VRAM against the configured model, when nvidia-smi is present."""
    reading = _nvidia_memory()
    if reading is None:
        return None
    total_mb, free_mb = reading

    detail = f"{free_mb} MiB free of {total_mb} MiB"
    if total_mb < 6000:
        return CheckResult(
            "GPU memory",
            Status.WARN,
            f"{detail} — tight for {config.models.chat} at num_ctx {config.models.num_ctx}",
            fix="Keep num_ctx modest and verify `ollama ps` still reports 100% GPU.",
        )
    return CheckResult("GPU memory", Status.PASS, detail)


# ------------------------------------------------------------------------- runner


async def run_doctor(
    *,
    loaded: LoadedConfig,
    workspace: Path,
    provider: LLMProvider | None,
) -> DoctorReport:
    """Run every check. ``provider=None`` skips the Ollama section (offline self-test)."""
    report = DoctorReport()
    config = loaded.config

    report.add(check_python())
    report.add(check_sqlite_fts5())
    report.add(check_git())
    report.add(check_ripgrep())
    report.add(check_workspace_location(workspace))
    report.add(check_wsl_networking(config.ollama.host))

    vram = check_vram(config)
    if vram is not None:
        report.add(vram)

    for source in loaded.sources:
        if source.existed:
            report.add(CheckResult(f"Config ({source.layer})", Status.PASS, str(source.path)))

    if loaded.dropped_allow_rules:
        report.add(
            CheckResult(
                "Project trust",
                Status.WARN,
                f"{loaded.dropped_allow_rules} allow rule(s) from project config are ignored",
                fix="Review them, then run `hearth trust` if you want them to apply.",
            )
        )

    if provider is not None:
        for result in await check_ollama(provider, config):
            report.add(result)

    return report


def run_doctor_sync(
    *,
    loaded: LoadedConfig,
    workspace: Path,
    provider: LLMProvider | None,
) -> DoctorReport:
    return asyncio.run(run_doctor(loaded=loaded, workspace=workspace, provider=provider))


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version leniently; unparseable parts sort as 0."""
    parts: list[int] = []
    for chunk in version.split("-")[0].split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts) or (0,)


def format_report(report: DoctorReport, *, use_color: bool = True) -> str:
    """Render the report as plain text. Rich formatting lives in the CLI layer."""
    symbols = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL"}
    lines: list[str] = []
    width = max((len(r.name) for r in report.results), default=0)

    for result in report.results:
        lines.append(f"[{symbols[result.status]}] {result.name.ljust(width)}  {result.detail}")
        if result.fix and result.status is not Status.PASS:
            lines.append(f"{'':>7}{'':<{width}}  -> {result.fix}")

    lines.append("")
    if report.failed:
        lines.append(f"{len(report.failed)} check(s) failed, {len(report.warned)} warning(s).")
    elif report.warned:
        lines.append(f"All checks passed, with {len(report.warned)} warning(s).")
    else:
        lines.append("All checks passed.")
    return "\n".join(lines)


def summarize(results: Sequence[CheckResult]) -> dict[Status, int]:
    counts = {Status.PASS: 0, Status.WARN: 0, Status.FAIL: 0}
    for result in results:
        counts[result.status] += 1
    return counts
