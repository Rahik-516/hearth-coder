"""Request assembly, in prefix-cache-stable order.

Ollama reuses the KV cache for a **byte-identical prompt prefix** on the same loaded model.
On a GPU that saves a little; on a CPU-split prefill it is the difference between a
follow-up question answering in seconds and answering in minutes. So the layout is a
correctness concern, not a style one (docs/system-design.md §9.2):

    [system]  core rules + mode guidance        <- stable for hours
    [system]  project instructions + repo map   <- stable until compaction
    [user]/[assistant] ... history              <- append-only within an epoch
    [user]   current message + <context>        <- new each turn

Four rules, each of which has a specific failure behind it:

1. **No volatile content in early messages.** A timestamp or a token countdown in the
   system prompt invalidates the cache on every single request.
2. **Retrieved context goes inside the user message**, never a mid-conversation system
   message: chat templates handle non-leading system messages inconsistently across model
   families, and a silently-dropped block is hard to notice.
3. **History is append-only within an epoch.** Trimming an old turn rewrites the prefix and
   throws the cache away. Oversized content is truncated *at insertion*, once.
4. **Tool schemas are stable.** Adding or removing a tool mid-session breaks the prefix, so
   a mode switch starts a new epoch instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hearth.core.context.budget import BudgetUsage, ContextBudget, Segment
from hearth.core.context.tokens import TokenEstimator
from hearth.llm.types import Message
from hearth.retrieval.types import FusedResult

#: Wrapper for retrieved code. The framing is load-bearing: it tells the model this text is
#: data rather than instructions, which is the first layer of prompt-injection defence
#: (docs/safety-and-tool-use.md §11).
CONTEXT_OPEN = '<context source="retrieval" trust="untrusted-data">'
CONTEXT_CLOSE = "</context>"

#: Languages whose fenced blocks get a syntax hint.
_FENCE_LANGUAGE = {
    "python": "python",
    "typescript": "typescript",
    "tsx": "tsx",
    "javascript": "javascript",
    "markdown": "markdown",
}


@dataclass
class BuiltRequest:
    """An assembled request, plus the accounting behind it."""

    messages: list[Message]
    usage: BudgetUsage
    #: Citations offered to the model, in the order they appear.
    citations: list[str] = field(default_factory=list)
    #: Results that did not fit the retrieval budget.
    dropped: list[FusedResult] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    @property
    def estimated_prompt_tokens(self) -> int:
        return self.usage.total_input

    @property
    def stable_prefix(self) -> list[Message]:
        """Messages that must stay byte-identical for the cache to be reused."""
        return [m for m in self.messages if m.role == "system"]


class ContextBuilder:
    """Assembles chat requests within a budget, in cache-stable order."""

    def __init__(
        self,
        *,
        budget: ContextBudget,
        estimator: TokenEstimator,
    ) -> None:
        self._budget = budget
        self._estimator = estimator

    def build(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history: list[Message] | None = None,
        project_instructions: str | None = None,
        repo_map: str | None = None,
        retrieved: list[FusedResult] | None = None,
        pinned: list[tuple[str, str]] | None = None,
    ) -> BuiltRequest:
        """Assemble one request.

        Args:
            system_prompt: Core rules and mode guidance. Must not vary within a session.
            user_message: What the user just asked.
            history: Prior turns, already truncated at insertion time.
            project_instructions: ``AGENTS.md`` content, capped.
            repo_map: Rendered repo map for this cache epoch.
            retrieved: Fused retrieval results, best first.
            pinned: ``(path, content)`` for ``@path`` mentions, which outrank retrieval.
        """
        notices: list[str] = []
        used: dict[Segment, int] = {}
        messages: list[Message] = []

        # --- stable prefix -------------------------------------------------
        messages.append(Message(role="system", content=system_prompt))
        used[Segment.SYSTEM] = self._estimator.estimate(system_prompt)

        epoch_parts: list[str] = []
        if project_instructions:
            capped, truncated = self._cap(project_instructions, self._budget.limit(Segment.PROJECT))
            epoch_parts.append(capped)
            used[Segment.PROJECT] = self._estimator.estimate(capped)
            if truncated:
                notices.append("Project instructions were truncated to fit the budget.")

        if repo_map:
            capped, truncated = self._cap(repo_map, self._budget.limit(Segment.REPO_MAP))
            epoch_parts.append(capped)
            used[Segment.REPO_MAP] = self._estimator.estimate(capped)
            if truncated:
                notices.append("The repo map was truncated to fit the budget.")

        if epoch_parts:
            messages.append(Message(role="system", content="\n\n".join(epoch_parts)))

        # --- history -------------------------------------------------------
        kept_history, history_tokens, dropped_turns = self._fit_history(history or [])
        messages.extend(kept_history)
        used[Segment.HISTORY] = history_tokens
        if dropped_turns:
            notices.append(f"{dropped_turns} older turn(s) were left out; run /compact to summarise them.")

        # --- current turn --------------------------------------------------
        # Slack from the stable prefix flows into retrieval, so a short system prompt buys
        # more context rather than being wasted.
        effective = self._budget.with_slack_from(used)
        block, citations, dropped, retrieval_tokens = self._build_context_block(
            retrieved or [],
            pinned or [],
            limit=effective.limit(Segment.RETRIEVAL),
        )

        content = f"{block}\n\n{user_message}" if block else user_message
        messages.append(Message(role="user", content=content))
        used[Segment.RETRIEVAL] = retrieval_tokens

        if dropped:
            notices.append(f"{len(dropped)} retrieved chunk(s) did not fit the context budget.")

        usage = BudgetUsage(budget=effective, used=used)
        return BuiltRequest(
            messages=messages,
            usage=usage,
            citations=citations,
            dropped=dropped,
            notices=notices,
        )

    # -------------------------------------------------------------- internals

    def _fit_history(self, history: list[Message]) -> tuple[list[Message], int, int]:
        """Keep the most recent turns that fit.

        Walks backwards: recent turns matter more, and dropping from the front preserves
        the append-only shape of what remains.
        """
        limit = self._budget.limit(Segment.HISTORY)
        kept: list[Message] = []
        total = 0

        for message in reversed(history):
            cost = self._estimator.estimate(message.content) + 4
            if total + cost > limit:
                break
            kept.append(message)
            total += cost

        kept.reverse()
        return kept, total, len(history) - len(kept)

    def _build_context_block(
        self,
        retrieved: list[FusedResult],
        pinned: list[tuple[str, str]],
        *,
        limit: int,
    ) -> tuple[str, list[str], list[FusedResult], int]:
        """Render pinned files and retrieved chunks into one framed block.

        Pinned files come first and are never dropped for retrieval's benefit: the user
        named them explicitly. They are still capped individually, so one enormous `@file`
        cannot consume the whole window — the budgeter trims and the caller notifies.
        """
        if not retrieved and not pinned:
            return "", [], [], 0

        parts: list[str] = [CONTEXT_OPEN]
        citations: list[str] = []
        dropped: list[FusedResult] = []
        spent = self._estimator.estimate(CONTEXT_OPEN) + self._estimator.estimate(CONTEXT_CLOSE)
        index = 1

        for path, content in pinned:
            share = max(1, limit // max(1, len(pinned) + 1))
            capped, was_capped = self._cap(content, share)
            rendered = self._render_entry(index, f"{path} (pinned)", capped, None)
            cost = self._estimator.estimate(rendered)

            parts.append(rendered)
            citations.append(path)
            spent += cost
            index += 1
            if was_capped:
                dropped.append(  # recorded so the caller can say the pin was trimmed
                    FusedResult(
                        chunk_id=-1,
                        path=path,
                        kind="pinned",
                        symbol_path=None,
                        start_line=0,
                        end_line=0,
                        text="",
                    )
                )

        for result in retrieved:
            rendered = self._render_entry(
                index,
                result.citation,
                result.text,
                result.symbol_path,
                language=result.language,
            )
            cost = self._estimator.estimate(rendered)
            if spent + cost > limit:
                dropped.append(result)
                continue

            parts.append(rendered)
            citations.append(result.citation)
            spent += cost
            index += 1

        parts.append(CONTEXT_CLOSE)
        return "\n".join(parts), citations, dropped, spent

    @staticmethod
    def _render_entry(
        index: int,
        citation: str,
        body: str,
        symbol_path: str | None,
        *,
        language: str | None = None,
    ) -> str:
        """One numbered, cited context entry.

        The citation header is what the model is asked to quote back, so it carries the
        exact `path:start-end` form the answer should use.
        """
        scope = f"  ({symbol_path})" if symbol_path else ""
        fence = _FENCE_LANGUAGE.get(language or "", "")
        return f"[{index}] {citation}{scope}\n```{fence}\n{body}\n```"

    def _cap(self, text: str, limit: int) -> tuple[str, bool]:
        """Trim text to a token limit, keeping head and tail.

        Head and tail rather than head alone: the end of a file usually carries as much
        meaning as its imports, and a hard cut at the top loses it entirely.
        """
        if limit <= 0:
            return "", bool(text)

        estimated = self._estimator.estimate(text)
        if estimated <= limit:
            return text, False

        keep_chars = int(limit * self._estimator.chars_per_token)
        head = int(keep_chars * 0.7)
        tail = keep_chars - head
        omitted = estimated - limit

        return (
            f"{text[:head]}\n\n… {omitted} tokens omitted …\n\n{text[-tail:]}" if tail > 0 else text[:head],
            True,
        )
