"""Compaction: trading old turns for a structured summary when the window fills.

docs/system-design.md §9.4. A long session eventually has more history than `num_ctx`
holds. The naive answers are both bad: dropping the oldest turns loses the decisions that
explain the current state, and letting history grow makes Ollama silently truncate the
prompt — taking the *system* prompt with it, because truncation happens from the front.

So the old turns are replaced by one summary message with a fixed shape. The shape is the
point. A free-form "summarise this conversation" produces prose that reads well and omits
the file paths, the commands and the user's stated preferences — exactly the things a
coding session cannot continue without. Asking for named fields makes the model fill them
in, and makes a missing one visible.

**Compaction starts a new cache epoch**, and that cost is deliberate. Everything before
the current message is the cached prefix; rewriting history invalidates it, so the next
turn pays one full prefill (§9.2). Doing it at 75% of the usable window rather than at
100% means that happens once, at a moment of Hearth's choosing, instead of mid-answer.

The summary is produced by the model, so it can be wrong. What survives compaction
verbatim — the last K turns, and anything pinned — is chosen so that being wrong costs
context rather than correctness.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from hearth.core.context.budget import ContextBudget, Segment
from hearth.core.context.tokens import TokenEstimator
from hearth.llm.types import Message

#: Fraction of the usable window (everything but the output reserve) that triggers
#: compaction. Below 1.0 on purpose: at 1.0 the first over-budget turn is already the one
#: being truncated, and compaction would run too late to help it.
COMPACT_AT = 0.75

#: Turns kept verbatim, by tier. A summary is a lossy account of what happened; the most
#: recent exchanges are the ones the next message actually refers to ("that file", "the
#: error above"), so they are kept as they were.
KEEP_RECENT_TURNS_SMALL = 2
KEEP_RECENT_TURNS_LARGE = 4

#: num_ctx at or above which the larger keep-window applies.
_LARGE_CONTEXT = 16_384

#: Marks the summary so a later compaction can recognise and replace its own output
#: rather than nesting summaries of summaries.
SUMMARY_MARKER = "## Session summary"

#: The fields the summary must contain (§9.4 step 2). Named rather than free-form: prose
#: about a coding session reliably drops the paths and commands it depends on.
SUMMARY_SECTIONS: tuple[str, ...] = (
    "Goal",
    "Decisions made",
    "Files read and modified",
    "Current state",
    "Open todos",
    "Important facts",
    "User preferences",
)

#: Produces the summary text from a prompt. Injected so the compactor stays testable
#: without a model, and so `core` keeps its single path to the LLM.
Summarizer = Callable[[str], Awaitable[str]]


@dataclass
class CompactionResult:
    """What compaction did, and what it cost."""

    history: list[Message]
    compacted: bool = False
    #: Messages folded into the summary.
    replaced: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    #: Set when the model could not produce a usable summary.
    degraded: str | None = None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CompactionPolicy:
    """When to compact, and how much to keep."""

    budget: ContextBudget
    estimator: TokenEstimator = field(default_factory=TokenEstimator)
    threshold: float = COMPACT_AT

    @property
    def keep_recent(self) -> int:
        return (
            KEEP_RECENT_TURNS_LARGE
            if self.budget.num_ctx >= _LARGE_CONTEXT
            else KEEP_RECENT_TURNS_SMALL
        )

    @property
    def usable_tokens(self) -> int:
        """Everything the prompt may occupy, excluding the output reserve.

        The reserve is never borrowed from: spending it on context buys a fuller prompt
        and a truncated answer, which is the worse trade.
        """
        return max(1, self.budget.num_ctx - self.budget.limit(Segment.OUTPUT))

    def trigger_tokens(self) -> int:
        return int(self.usable_tokens * self.threshold)

    def should_compact(self, history: Sequence[Message], *, overhead_tokens: int = 0) -> bool:
        """Whether this history plus the fixed parts of the prompt is close enough to full.

        ``overhead_tokens`` is the system prompt, project instructions and repo map — the
        parts compaction cannot shrink. Ignoring them would let a session with a large repo
        map sail past the trigger and hit the real ceiling instead.
        """
        return self.measure(history) + overhead_tokens >= self.trigger_tokens()

    def measure(self, history: Sequence[Message]) -> int:
        return sum(self.estimator.estimate(_message_text(message)) for message in history)


async def compact(
    history: Sequence[Message],
    *,
    policy: CompactionPolicy,
    summarize: Summarizer,
    pinned_facts: Sequence[str] = (),
) -> CompactionResult:
    """Replace old turns with one structured summary, keeping the recent ones verbatim.

    Args:
        history: The full conversation so far.
        policy: When to compact and how many turns to keep.
        summarize: Produces the summary. Given the prompt, returns the text.
        pinned_facts: Statements that must survive verbatim regardless of what the model
            writes — the acceptance criterion's "keeps pinned facts after compaction".
            They are appended to the summary rather than handed to the model to
            paraphrase, because a fact that survives only if the model chooses to repeat
            it is not pinned.

    Returns:
        The rewritten history. On failure the original history is returned unchanged with
        ``degraded`` set: a session that cannot compact is still a working session, and
        raising here would end a turn the user was in the middle of.
    """
    before = policy.measure(history)
    older, recent = _split(history, keep_recent=policy.keep_recent)

    if not older:
        # Nothing to fold. Happens when the recent turns alone exceed the budget, which
        # compaction cannot fix — the context builder's own truncation handles it.
        return CompactionResult(history=list(history), tokens_before=before, tokens_after=before)

    try:
        summary_text = await summarize(build_summary_prompt(older))
    except Exception as exc:  # any provider failure degrades the same way
        return CompactionResult(
            history=list(history),
            tokens_before=before,
            tokens_after=before,
            degraded=f"summarization failed: {exc}",
        )

    summary_text = summary_text.strip()
    if not summary_text:
        return CompactionResult(
            history=list(history),
            tokens_before=before,
            tokens_after=before,
            degraded="the model returned an empty summary",
        )

    summary = Message(role="user", content=render_summary(summary_text, pinned_facts))
    rewritten = [summary, *recent]
    after = policy.measure(rewritten)

    if after >= before:
        # A summary longer than what it replaced is not a saving, and swapping real turns
        # for a worse account of them would be a strict loss.
        return CompactionResult(
            history=list(history),
            tokens_before=before,
            tokens_after=before,
            degraded="the summary was no smaller than the history it replaced",
        )

    return CompactionResult(
        history=rewritten,
        compacted=True,
        replaced=len(older),
        tokens_before=before,
        tokens_after=after,
    )


def build_summary_prompt(older: Sequence[Message]) -> str:
    """The structured prompt from §9.4 step 2.

    Every section is requested by name, and the model is told to write "none" rather than
    omit one: a missing heading is indistinguishable from a heading the model decided was
    empty, and the next turn cannot tell which.
    """
    transcript = "\n\n".join(
        f"[{message.role}] {_message_text(message).strip()}"
        for message in older
        if _message_text(message).strip()
    )
    sections = "\n".join(f"### {name}" for name in SUMMARY_SECTIONS)

    return (
        "Summarise the conversation below so it can be continued after the original "
        "messages are discarded. This is a coding session: paths, commands and exact "
        "names matter more than prose.\n\n"
        "Use exactly these headings, in this order. Write 'none' under any that does not "
        "apply — do not omit a heading.\n\n"
        f"{sections}\n\n"
        "Keep it under 400 words. Quote file paths and commands exactly.\n\n"
        "--- conversation ---\n"
        f"{transcript}"
    )


def render_summary(summary_text: str, pinned_facts: Sequence[str] = ()) -> str:
    """The message that replaces the folded turns.

    Pinned facts are appended *after* the model's text, under their own heading, so they
    are not competing with it for the model's attention and cannot be paraphrased away.
    """
    parts = [SUMMARY_MARKER, "", summary_text.strip()]

    facts = [fact.strip() for fact in pinned_facts if fact.strip()]
    if facts:
        parts.extend(["", "### Pinned", *(f"- {fact}" for fact in facts)])

    parts.extend(
        [
            "",
            "The messages this summarises have been discarded. Continue from here, and "
            "re-read any file you need rather than recalling its contents.",
        ]
    )
    return "\n".join(parts)


def is_summary(message: Message) -> bool:
    """Whether a message is a previous compaction's output."""
    return message.role == "user" and message.content.lstrip().startswith(SUMMARY_MARKER)


def _split(
    history: Sequence[Message], *, keep_recent: int
) -> tuple[list[Message], list[Message]]:
    """Divide into (folded, kept), counting *user turns* rather than messages.

    A turn is a user message and everything that answered it — assistant replies and tool
    results. Counting raw messages would keep two tool outputs and call that two turns,
    cutting the user's actual question away from its answer.
    """
    boundaries = [index for index, message in enumerate(history) if message.role == "user"]
    # A previous summary is not a turn to keep; it is a candidate for re-folding, so that
    # compacting twice produces one summary rather than a summary of a summary.
    boundaries = [index for index in boundaries if not is_summary(history[index])]

    if len(boundaries) <= keep_recent:
        return [], list(history)

    cut = boundaries[-keep_recent]
    return list(history[:cut]), list(history[cut:])


def _message_text(message: Message) -> str:
    """Everything in a message that costs tokens in the prompt."""
    parts = [message.content]
    if message.thinking:
        parts.append(message.thinking)
    parts.extend(f"{call.name} {call.arguments}" for call in message.tool_calls)
    return "\n".join(part for part in parts if part)
