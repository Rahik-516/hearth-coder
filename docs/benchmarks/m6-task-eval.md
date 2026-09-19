# M6 task eval — results

**3 of 3 tasks passed** on `qwen3.5:4b`, in one run, on the reference machine.

The M6 criterion asked for ≥1 of 3. Read that result with the two caveats below before
treating it as a benchmark: inference was **CPU-only**, and retrieval was **lexical-only**.

## Run

| | |
|---|---|
| Date | 2026-09-19 |
| Model | `qwen3.5:4b` (4.7B, Q4_K_M, `qwen35` family) |
| `num_ctx` | 12288, sent explicitly on every request |
| Placement | **100% CPU** — `ollama ps` reported no GPU offload |
| Retrieval | lexical only; `qwen3-embedding:0.6b` not installed, so 0/56 chunks embedded |
| Ollama | 0.34.2 |
| Machine | ASUS ROG Strix G15, Ryzen 7 4800H, RTX 3060 6 GB (unused), 16 GB RAM, WSL2 |
| Command | `hearth eval tasks --repo-root ~/code/hearth` |

## Results

| Task | Result | Steps | Tool calls | Wall time |
|---|---|---|---|---|
| `add-unit-test` | PASS | 10 | 9 | 242.3 s |
| `rename-symbol` | PASS | 10 | 10 | 374.0 s |
| `fix-failing-test` | PASS | 8 | 7 | 246.9 s |

Every task ends by running the fixture's suite, and each predicate additionally checks the
change itself: the new test names `compute_tax`, the old symbol is gone everywhere and the
new one present, the seeded failing test comes back byte-identical.

An earlier run of the same three tasks (before the single-run table above) scored
PASS/PASS/PASS at 283 s, 342 s and 480 s, so the spread on wall time is wide — roughly
±40% on the same task and model. Treat any single timing here as indicative, not precise.

## Caveat 1: this is CPU-only, and the GPU is not the reason you think

`nvidia-smi` reports the RTX 3060 healthy with 5996 of 6144 MiB free and idle, yet the
model loads at 100% CPU. The cause is in Ollama's own log
(`%LOCALAPPDATA%\Ollama\server.log`):

```
msg="failure during llama-server GPU discovery"
error="llama-server --list-devices failed: exit status 0xc0000005: The instruction at 0xp
referenced memory at 0xp. The memory could not be s."
```

`0xc0000005` is an access violation, and it repeats for every backend Ollama tries —
vulkan, cuda_v12, cuda_v13, rocm_v7_1. Ollama therefore registers `total_vram="0 B"` and
runs on CPU. The card and driver are fine; Ollama's device-discovery helper crashes.

**Every timing above is a CPU number and is not comparable to a GPU run.** Expect a large
improvement once discovery works — but do not guess the factor; re-run this eval.

`hearth doctor` now distinguishes this case from a genuinely full card, because the advice
is opposite: a full card wants a smaller model, an idle one wants Ollama fixed.

## Caveat 2: retrieval was lexical only

No embedding model is installed, so dense retrieval contributed nothing and every run
printed `Embeddings are 0/56 complete`. These tasks are small and lexical search found the
right files, but a repository where the answer needs semantic search would behave
differently. To close this:

```bash
ollama pull qwen3-embedding:0.6b
```

## Still outstanding: the 9b half

> With `qwen3.5:9b` (eval-only, partial offload), ≥2 of 3 succeed.

Not run. Only the 4B model is installed, and with GPU discovery broken a 9B model would run
entirely on CPU at roughly twice the per-token cost — hours for the suite, measuring the
CPU rather than the model. Worth doing after the GPU issue is resolved.

## What the first run taught us, and what changed because of it

The first full run scored **1/3**, and that number was wrong in a way worth recording.

Two of the three tasks named functions — `compute_total` and `apply_discount` — that
`py_small` does not contain. They were invented when the tasks were written and never
checked against the fixture. On `rename-symbol` the model searched, reported that the
symbol was absent, listed the functions that do exist, and asked which was meant. That is
exactly what `prompts/mode_agent.md` asks for, and the scorer recorded it as a failure.

The run measured the tasks, not the model. Two changes followed:

* `TaskSpec.required_symbols`, with `prepare_workspace` **refusing** a task whose symbols
  the fixture lacks rather than running it. An unanswerable task is not a hard task.
* The `fix-failing-test` scorer now requires the seeded test back byte-identical. A green
  suite alone is not enough, because deleting the failing test also turns the suite green.

Both are covered by unit tests. The general lesson: a scorer that can be satisfied by the
model doing nothing, or that punishes it for being right, produces numbers that look like
measurements and are not.
