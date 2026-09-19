"""Domain exceptions for the safety layer."""

from __future__ import annotations


class SafetyError(Exception):
    """Base class for every error raised by ``hearth.safety``."""


class PathError(SafetyError):
    """A path was refused by the workspace jail.

    Carries the *reason* rather than the resolved path: the message is shown to the model
    as a corrective error, and echoing back a resolved system path would leak filesystem
    layout for no benefit.
    """

    def __init__(self, reason: str, *, requested: str | None = None) -> None:
        self.reason = reason
        self.requested = requested
        super().__init__(f"{reason}: {requested!r}" if requested else reason)


class PolicyError(SafetyError):
    """The policy engine could not reach a decision.

    Callers must treat this as a denial. A policy that cannot decide has failed, and
    failing open would be the worst possible reading of it.
    """


class AuditError(SafetyError):
    """The audit log could not be written."""
