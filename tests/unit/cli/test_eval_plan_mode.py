"""The eval harness's plan-mode variant.

`hearth eval tasks --plan` runs the `/plan` -> approve -> `/execute` pipeline with the
harness as the approver. What matters is that it really is that pipeline — two phases, the
second one carrying the plan — and that a model which cannot produce a plan is scored as a
failure rather than quietly falling back to the plain loop, which would make the two modes'
numbers incomparable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hearth.cli import eval_commands
from hearth.core.session import Mode
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"

PLAN = json.dumps(
    {
        "goal": "Reword a docstring",
        "steps": [
            {
                "description": "edit the docstring",
                "files": ["src/billing/errors.py"],
                "change_type": "modify",
            }
        ],
    }
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


def install(monkeypatch: pytest.MonkeyPatch, script: list[ScriptedResponse]) -> ScriptedProvider:
    provider = ScriptedProvider(script)
    monkeypatch.setattr(
        "hearth.cli.chat_commands._build_runtime",
        lambda root, loaded: (provider, None, None, None),
    )
    return provider


def test_plan_mode_runs_two_phases_and_executes_the_plan(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = install(
        monkeypatch,
        [
            ScriptedResponse("I looked at the file."),  # plan exploration
            ScriptedResponse(PLAN),  # plan extraction
            ScriptedResponse(
                "",
                tool_calls=[
                    ToolCall(call_id="r", name="read_file", arguments={"path": "src/billing/errors.py"})
                ],
            ),
            ScriptedResponse(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="e",
                        name="edit_file",
                        arguments={
                            "path": "src/billing/errors.py",
                            "old_string": "Base class for billing failures.",
                            "new_string": "Base class for all billing failures.",
                        },
                    )
                ],
            ),
            ScriptedResponse("Done."),
        ],
    )

    result = eval_commands._run_task(workspace, "scripted", "reword it", max_steps=None, plan=True)

    assert result.ok
    assert "all billing failures" in (workspace / "src/billing/errors.py").read_text(encoding="utf-8")

    # Phase 1 is read-only with no schema until extraction; phase 2 carries the plan.
    exploration, extraction, *execution = provider.requests
    assert exploration.format is None
    assert extraction.format is not None
    user_messages = [m.content for m in execution[0].messages if m.role == "user"]
    assert any("Carry out this plan" in text and "Reword a docstring" in text for text in user_messages)


def test_plan_phase_tools_are_the_read_only_set(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposed tools are the safety property: in plan mode nothing can edit."""
    provider = install(
        monkeypatch, [ScriptedResponse("looked"), ScriptedResponse("not a plan"), ScriptedResponse("x")]
    )

    eval_commands._run_task(workspace, "scripted", "reword it", max_steps=None, plan=True)

    names = {tool["function"]["name"] for tool in provider.requests[0].tools}
    assert "read_file" in names
    assert not names & {"edit_file", "write_file", "run_command", "git_commit"}


def test_a_model_that_cannot_plan_scores_as_a_failure_not_a_fallback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back to the plain loop would make plan-mode and plain numbers incomparable
    while looking like a success."""
    provider = install(monkeypatch, [ScriptedResponse("looked"), ScriptedResponse("I would start by...")])
    before = (workspace / "src/billing/errors.py").read_bytes()

    result = eval_commands._run_task(workspace, "scripted", "reword it", max_steps=None, plan=True)

    assert not result.ok
    assert result.reason == "no_plan"
    assert len(provider.requests) == 2, "no execute phase was started"
    assert (workspace / "src/billing/errors.py").read_bytes() == before


def test_plain_mode_is_unchanged_by_the_new_option(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = install(monkeypatch, [ScriptedResponse("Nothing to do.")])

    result = eval_commands._run_task(workspace, "scripted", "look around", max_steps=None)

    assert result.ok
    assert len(provider.requests) == 1
    assert provider.requests[0].format is None


def test_the_schemas_follow_the_requested_mode() -> None:
    from hearth.config.loader import LoadedConfig
    from hearth.config.schema import HearthConfig

    loaded = LoadedConfig(config=HearthConfig())

    plan_tools = {s["function"]["name"] for s in eval_commands.gateway_schemas(loaded, object(), mode="plan")}
    agent_tools = {s["function"]["name"] for s in eval_commands.gateway_schemas(loaded, object())}

    assert "edit_file" not in plan_tools
    assert "edit_file" in agent_tools
    assert Mode.PLAN.value == "plan"
