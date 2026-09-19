"""What a tool can do — the axis the policy engine switches on.

This lives in ``safety`` rather than ``tools`` for one structural reason: the
``policy-pure`` contract in ``.importlinter`` forbids ``hearth.safety.policy`` from
importing ``hearth.tools``, and risk is the primary input to every decision. Defining it
here lets the engine stay pure while ``tools.base`` re-exports the name, so the ~six
modules that say ``from hearth.tools.base import Risk`` keep working.
"""

from __future__ import annotations

from enum import StrEnum


class Risk(StrEnum):
    """Ordered from least to most consequential (docs/safety-and-tool-use.md §2.1)."""

    READ = "READ"
    META = "META"
    WRITE = "WRITE"
    EXEC = "EXEC"
    VCS_WRITE = "VCS_WRITE"


#: Risks with no side effects outside the session. Always allowed inside the workspace.
SIDE_EFFECT_FREE: frozenset[Risk] = frozenset({Risk.READ, Risk.META})

#: Risks that only exist in agent mode (docs/safety-and-tool-use.md §4.1).
AGENT_ONLY: frozenset[Risk] = frozenset({Risk.WRITE, Risk.EXEC, Risk.VCS_WRITE})
