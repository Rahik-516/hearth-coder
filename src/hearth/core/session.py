"""Session state.

Holds what a conversation needs in memory and mirrors it to ``state.db`` so
``hearth resume`` can rebuild it (docs/system-design.md §5.2).

``num_ctx`` is fixed at session start and never changed afterwards. Changing it forces
Ollama to reload the model — seconds of stall — and invalidates the KV cache, so a
mid-session change would silently make every later turn slow
(docs/system-design.md §1.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from hearth.core.context.tokens import TokenEstimator
from hearth.llm.types import Message
from hearth.storage.state_repo import SessionRecord, StateRepository


class Mode(StrEnum):
    """Interaction mode. Decides which tools exist, separately from permissions."""

    CHAT = "chat"
    PLAN = "plan"
    AGENT = "agent"


class PermissionLevel(StrEnum):
    SUPERVISED = "supervised"
    AUTO_EDIT = "auto-edit"
    HEADLESS = "headless"


@dataclass
class Session:
    """One conversation, in memory."""

    id: str
    workspace: Path
    model: str
    num_ctx: int
    mode: Mode = Mode.CHAT
    permission_level: PermissionLevel = PermissionLevel.SUPERVISED

    history: list[Message] = field(default_factory=list)
    estimator: TokenEstimator = field(default_factory=TokenEstimator)

    #: Cache epoch. Bumped by compaction or a mode switch, never mid-turn
    #: (docs/system-design.md §9.2).
    epoch: int = 0

    #: Slowest prefill rate seen this session, in tokens/sec. This is the cold-prefill
    #: baseline: Ollama does not report how many prompt tokens came from the KV cache, but
    #: a cached prefix shows up as a dramatically higher prefill rate for the same prompt
    #: size. Measured on the reference machine: 51 tok/s cold versus 1470 tok/s with the
    #: prefix reused — the same 28x that vanishes if the prefix changes by one word.
    prefill_baseline_tps: float | None = None

    #: Files pinned with `@path` for the next turn only.
    pinned_paths: list[str] = field(default_factory=list)
    #: Paths mentioned or edited this session, used to bias retrieval.
    touched_paths: set[str] = field(default_factory=set)

    title: str | None = None

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.history if m.role == "user")

    def add_user(self, text: str) -> Message:
        message = Message(role="user", content=text)
        self.history.append(message)
        return message

    def add_assistant(self, text: str, *, thinking: str | None = None) -> Message:
        message = Message(role="assistant", content=text, thinking=thinking)
        self.history.append(message)
        return message

    def clear(self) -> None:
        """Drop history but keep the session and its identity.

        Starts a new epoch: the prefix is about to change completely, so pretending the
        old cache still applies would be wrong.
        """
        self.history.clear()
        self.epoch += 1
        self.pinned_paths.clear()

    def replace_history(self, messages: list[Message]) -> None:
        """Swap history for a compacted version, starting a new epoch.

        The epoch bump is the whole cost of compaction: everything before the current
        message is the cached prefix, and rewriting history invalidates it, so the next
        turn pays one full prefill (docs/system-design.md §9.2). Not bumping would be
        worse than the cost — the server would be told a prefix it no longer holds is
        still valid.
        """
        self.history = list(messages)
        self.epoch += 1

    def observe_prefill(self, *, prompt_tokens: int | None, prefill_ms: float | None) -> int | None:
        """Record a prefill measurement and estimate how much of it was cached.

        Returns an estimated cached-token count, or None when there is nothing to compare
        against yet.

        This is **inferred, not reported**. Ollama gives prompt token counts and prefill
        duration but not a cache-hit count, so the estimate comes from how far this turn's
        prefill rate exceeds the session's slowest observed rate. It is a diagnostic for
        the stats line, deliberately not used for budgeting.
        """
        if not prompt_tokens or not prefill_ms or prefill_ms <= 0:
            return None

        rate = prompt_tokens / (prefill_ms / 1000)
        if self.prefill_baseline_tps is None or rate < self.prefill_baseline_tps:
            self.prefill_baseline_tps = rate
            return None

        baseline = self.prefill_baseline_tps
        if rate <= baseline * 1.5:
            return 0  # measurably the same speed: nothing was reused

        return int(prompt_tokens * (1 - baseline / rate))

    def pin(self, path: str) -> None:
        if path not in self.pinned_paths:
            self.pinned_paths.append(path)

    def take_pins(self) -> list[str]:
        """Consume pins for this turn. Pins apply once, not forever."""
        pins = list(self.pinned_paths)
        self.pinned_paths.clear()
        return pins

    def switch_mode(self, mode: Mode) -> None:
        """Change mode, starting a new epoch.

        Tool schemas are part of the cached prefix, so a mode change cannot reuse it
        (docs/system-design.md §9.2, rule 4).
        """
        if mode is not self.mode:
            self.mode = mode
            self.epoch += 1

    def switch_model(self, model: str) -> None:
        """Change model, resetting calibration and starting a new epoch.

        The estimator's calibration is per-model — a new tokenizer makes the old ratio
        meaningless — and the cache belongs to the previously loaded model.
        """
        if model != self.model:
            self.model = model
            self.estimator.reset()
            self.epoch += 1


class SessionStore:
    """Persists sessions to ``state.db`` and restores them."""

    def __init__(self, repository: StateRepository) -> None:
        self._repo = repository

    @property
    def repo(self) -> StateRepository:
        """The underlying repository.

        Exposed because ``state.db`` holds more than sessions — checkpoints, grants and
        trust live there too, and the frontend needs the same connection rather than a
        second one. Two connections to one SQLite file is how you get a writer lock
        fighting itself.
        """
        return self._repo

    def create(
        self,
        *,
        workspace: Path,
        model: str,
        num_ctx: int,
        mode: Mode = Mode.CHAT,
    ) -> Session:
        record = self._repo.create_session(
            mode=mode.value,
            model=model,
            num_ctx=num_ctx,
            workspace=str(workspace),
        )
        return Session(
            id=record.id,
            workspace=workspace,
            model=model,
            num_ctx=num_ctx,
            mode=mode,
        )

    def resume(self, session_id: str) -> Session | None:
        """Rebuild a session from storage, including its history and epoch."""
        record = self._repo.get_session(session_id)
        if record is None:
            return None

        session = Session(
            id=record.id,
            workspace=Path(record.workspace or "."),
            model=record.model or "",
            num_ctx=record.num_ctx or 12_288,
            mode=Mode(record.mode),
            permission_level=PermissionLevel(record.permission_level),
            title=record.title,
            epoch=self._repo.current_epoch(record.id),
        )
        # storage hands back JSON; converting here is what keeps `storage` free of any
        # dependency on `llm` (docs/project-structure.md §2).
        session.history = [
            Message.model_validate_json(stored.content_json) for stored in self._repo.load_messages(record.id)
        ]
        return session

    def resume_latest(self, *, workspace: Path | None = None) -> Session | None:
        record = self._repo.latest_session(workspace=str(workspace) if workspace else None)
        return None if record is None else self.resume(record.id)

    def save_message(self, session: Session, message: Message) -> None:
        self._repo.append_message(
            session.id,
            role=message.role,
            content_json=message.model_dump_json(),
            token_estimate=session.estimator.estimate(message.content),
            epoch=session.epoch,
        )

    def set_title_from(self, session: Session, first_message: str) -> None:
        """Name a session after its opening question, once.

        A session list of timestamps is unusable; the first question is what someone
        actually remembers about a conversation.
        """
        if session.title:
            return
        title = " ".join(first_message.split())[:60]
        session.title = title
        self._repo.touch_session(session.id, title=title)

    def clear(self, session: Session) -> None:
        self._repo.clear_messages(session.id)
        session.clear()

    def list_recent(self, limit: int = 20) -> list[SessionRecord]:
        return self._repo.list_sessions(limit=limit)
