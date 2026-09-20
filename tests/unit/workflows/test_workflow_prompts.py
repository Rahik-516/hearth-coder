"""Workflow prompts are snapshot-guarded (CLAUDE.md rule 6).

A prompt change is a behaviour change, so it has to be a deliberate act: update the snapshot
on purpose and re-run the eval before merging.
"""

from __future__ import annotations

import pytest

from hearth.prompts import load

WORKFLOW_PROMPTS = [
    "workflows/commit",
    "workflows/review",
    "workflows/test",
    "workflows/test_fix",
    "workflows/doc_architecture",
    "workflows/doc_readme",
    "workflows/doc_component",
    "plan_extract",
    "mode_plan",
]


@pytest.mark.parametrize("name", WORKFLOW_PROMPTS)
def test_prompt_snapshot(name: str, snapshot) -> None:
    assert load(name) == snapshot


@pytest.mark.parametrize("name", ["workflows/commit", "workflows/review"])
def test_workflow_templates_have_the_placeholders_the_code_fills(name: str) -> None:
    """The code replaces `{{omitted}}` and `{{diff}}`; a template missing one silently
    drops what the code worked to gather."""
    text = load(name)

    assert "{{diff}}" in text
    assert "{{omitted}}" in text
