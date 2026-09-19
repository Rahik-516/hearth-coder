# M6 task eval — results

**Status: harness landed, numbers outstanding.** The three tasks, the disposable-workspace
setup and the structural scorers are implemented in `src/hearth/evals/task_eval.py` and
covered by `tests/unit/test_task_eval.py`. No live figures have been recorded yet.

## Why there are no numbers

No chat model is present on this machine. `GET /api/tags` on the loopback Ollama returns
`{"models":[]}` — the models were lost with the WSL home directory (see CLAUDE.md) and have
not been pulled again. Pulling one is several GB and a deliberate decision for whoever owns
the machine, not something the agent should do on its own.

The eval is therefore **not** blocked on code. It is blocked on:

```bash
ollama pull qwen3.5:4b
```

## The acceptance criterion, unchanged

From [docs/implementation-roadmap.md](../implementation-roadmap.md) M6:

> With `qwen3.5:4b`, ≥1 of 3 tasks succeeds within step limits. With `qwen3.5:9b`
> (eval-only, partial offload), ≥2 of 3 succeed. Record results with timings.

Both halves are outstanding. The 9b half was already expected to be: only the 4B and
embedding models were ever on disk, and with the GPU unavailable a 9B model runs entirely
on CPU. The 4b half is newly outstanding for the reason above.

Treat these as starting targets and recalibrate after the first real run, as the roadmap
says.

## The tasks

| Task | Fixture | Passes when |
|---|---|---|
| `add-unit-test` | `py_small` | a test mentioning `apply_discount` exists **and** the suite is green |
| `rename-symbol` | `py_small` | `compute_total` appears nowhere, `calculate_total` does, suite green |
| `fix-failing-test` | `py_small` | the seeded failing test passes without having been edited |

Each runs in a `shutil.copytree` of the fixture, never the fixture itself (CLAUDE.md rule
8). `fix-failing-test` seeds its own failure, and a test asserts the task **starts red** —
without that, passing it would prove nothing.

## Scoring is structural, never model-graded

Every predicate is a check over the resulting working tree, and each one ends by running
the fixture's suite. A task whose suite is red scores zero however good the diff looks:
an unverified change is not a completed task, which is the same standard
`prompts/system_tools.md` sets for the agent itself.

A model grading another model's diff would make the eval's numbers depend on the thing
being measured, and a 4B judge is not a reliable grader of a 4B worker.

## What to record when it runs

Per task: pass/fail, steps, tool calls, wall time. Plus machine state — which model,
`num_ctx`, and whether `ollama ps` showed it fully on the GPU, since a partially offloaded
model changes the timings by an order of magnitude and a number without that context is
not comparable to the next one.
