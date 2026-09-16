"""The policy engine — a pure decision function.

**Invariant: this module performs no I/O.** No filesystem, no network, no clock, no
database. It takes (tool, prepared plan, session view, config view) and returns a
decision. That is what makes it exhaustively testable with tables and Hypothesis, and it
is enforced by the ``policy-pure`` contract in ``.importlinter``: importing
``hearth.storage``, ``hearth.llm``, ``hearth.tools`` or ``hearth.git`` from here fails CI.

Evaluation order, first decisive result wins (docs/safety-and-tool-use.md §5.1):

1. Hard invariants          -> Deny (never configurable)
2. Mode restrictions        -> Deny
3. User DENY rules          -> Deny
4. Classification overlays  -> Destructive: Ask+typed, or Deny when headless
5. Session grants           -> Allow (exact grant keys only)
6. User ALLOW rules         -> Allow (project rules only when the project is trusted)
7. User ASK rules           -> Ask
8. Permission-level default -> per the table in docs/safety-and-tool-use.md §4.1

Two rules govern every change here: it fails closed, and the agent can never raise its
own privileges. Implemented in M5 (docs/implementation-roadmap.md) — tests first,
including adversarial cases.
"""

from __future__ import annotations
