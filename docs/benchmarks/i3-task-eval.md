# I3 task eval — ten tasks

**8 of 10 passed** on `qwen3.5:4b` in a single run (5 m 44 s wall, GPU, `num_ctx` 12288,
lexical retrieval only — the fixtures are indexed without embeddings). The I3 bar was
≥25 % at 4b; this is comfortably above it.

| Task | Result | Steps | Tools | Time | Note |
|---|---|---|---|---|---|
| `add-unit-test` | PASS | 9 | 9 | 49.8 s | |
| `rename-symbol` | **FAIL** | 10 | 9 | 49.7 s | old name still in `invoice_service.py` |
| `fix-failing-test` | PASS | 4 | 3 | 14.0 s | |
| `add-model-property` | PASS | 7 | 6 | 20.1 s | behavioural probe |
| `write-error-tests` | PASS | 9 | 8 | 31.4 s | |
| `extract-quantize` | PASS | 10 | 9 | 47.1 s | two-file refactor |
| `generate-architecture-doc` | **FAIL** | 10 | 13 | 46.5 s | ran out of steps before writing the file |
| `fix-injected-bug` | PASS | 7 | 6 | 30.2 s | |
| `add-validation` | PASS | 10 | 9 | 42.8 s | see caveat 2 |
| `readme-usage-section` | PASS | 3 | 2 | 11.2 s | |

## How to read this

**1. One run, on tasks written to be answerable.** The earlier three-task runs varied by
roughly ±40 % on wall time and the M6 pass count moved between runs. Treat 8/10 as
"well above the bar", not as a stable rate. Failures are the interesting rows: both are
step-limit or thoroughness failures, not wrong answers, and `rename-symbol` has been the
least reliable task since the eval began.

**2. A pass means the tree is right, not that the model verified it.** `add-validation`
ended at the step limit with an answer saying "the loop is complete — no further
verification needed", having run no tests. The scorer then ran the suite and the behavioural
probe itself and both passed, so it is a genuine pass — but the model's own claim carried no
weight and should not be read as evidence it checked its work. This is the same distinction
that produced the M6 correction: what the harness measures is the working tree.

**3. Plan mode was not enabled.** The roadmap's starting target reads "with plan mode
enabled". These runs use the plain agent loop (`hearth eval tasks`), so they measure the
*baseline* that plan mode has to beat, not plan mode. A plan-mode variant of the harness is
a small follow-up and would make the comparison direct.

## Scorers

Each new scorer is proved in `tests/unit/test_task_eval_breadth.py` before its number is
trusted: an untouched workspace scores zero and says why, a hand-written solution passes a
real pytest run, and the obvious shortcuts fail (deleting or rewriting the seeded test, a
property that always returns `True`, copying a function out and leaving the original,
changing the rounding, a document that names a file that does not exist, replacing the
README instead of extending it). Where a text match could be satisfied by wrong behaviour the
check runs a snippet against the workspace instead.

## Still outstanding

* **The 9b half** (≥40 % at `qwen3.5:9b`). The model is not installed
  (`ollama pull qwen3.5:9b`, several GB); with the GPU working it is now runnable.
* **A plan-mode run** of the same ten tasks (see caveat 3).
* **Repeat runs** to put a spread on the 4b number.
