# 0002 — A custom agent loop, not a framework

| | |
|---|---|
| Status | Accepted |
| Date | 2026-09-16 |
| Deciders | Project owner |

## Context

Every agent framework assumes a capable hosted model with reliable tool calling. Hearth cannot make that
assumption: it targets 4B–30B models on consumer hardware, where usable context is 12K–32K, prefill is
often the latency bottleneck, and tool calls are sometimes emitted as plain text with invented arguments
(`docs/system-design.md` §1.1).

## Decision

Write the loop — about 400 lines over the native Ollama API, with Pydantic v2 validating every tool call.
No agent framework.

Full analysis and the rejected alternatives: `docs/tech-stack.md` §9.

## Why the loop cannot be delegated

Four properties are the product, and each one lives in the loop:

1. **Prefix-cache-stable prompt layout.** Static system prompt → project instructions → append-only
   history → current user message carrying retrieved context. A framework that reorders or re-renders
   messages destroys the KV cache, and on a CPU-split prefill that is the difference between seconds and
   minutes.
2. **Approvals are mid-loop, asynchronous and blocking.** `ApprovalRequested` suspends execution until a
   response arrives, and edited arguments are re-prepared and re-evaluated by policy so a user edit
   cannot bypass a deny rule.
3. **Tolerant tool-call parsing.** Native `tool_calls` first, then text fallbacks, with corrective error
   messages against a retry budget — and the fallback strategy is per model profile.
4. **Step limits, loop detection and context compaction** scale with a per-model reliability profile.

## Alternatives considered

| Alternative | Why not |
|---|---|
| **LangGraph** | Real graph semantics and checkpointing, but a large dependency tree, and its message handling fights the byte-identical-prefix requirement. |
| **PydanticAI** | The closest philosophical fit, sharing the Pydantic validation approach. Rejected because approvals-as-events, per-profile fallback parsing and cache-stable layout would each require bypassing it. |
| **smolagents** | Code-execution-centric agents are the wrong safety model: Hearth previews and approves each discrete tool call. |
| **AutoGen / CrewAI** | Multi-agent orchestration, when one model at a time fits in 6 GB of VRAM. |

## Consequences

- Full control over prompts, cache layout, approval flow and failure recovery.
- The loop must be kept small enough to hold in one head — it is the thing most often debugged.
- Model-family differences live in `profiles.toml`, not in code: adding a family means adding a profile.
- Everything is testable offline through `ScriptedProvider` with scripted approval responses.

## Revisit when

A target model family has poor native tool support, requiring a per-profile text protocol — that is a
profile change, not a framework change. A framework becomes worth reconsidering only if Hearth ever
drops the local-model constraint, which would make it a different product.
