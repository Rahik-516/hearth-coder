# py_small

A deliberately small billing service, used as an indexing and retrieval fixture.

## Layout

| Path | Why it exists |
|---|---|
| `src/billing/invoice_service.py` | `InvoiceService.finalize` — the M1 retrieval target. Also covers nested classes and decorated methods. |
| `src/billing/models.py` | Frozen dataclasses, enums, properties |
| `src/billing/payments.py` | `TokenBucket` — the vocabulary-gap case for "where do we throttle API calls?" |
| `src/billing/reporting.py` | `build_monthly_statement` is oversized on purpose, to force chunk splitting |
| `src/billing/broken_syntax.py` | Invalid Python. Must still chunk. **Do not fix.** |
| `src/billing/errors.py` | Exception hierarchy |
| `tests/` | Real tests, so `run_tests` and the test-writer workflow have something to work with |

Nothing here is meant to be production code. It is meant to be *representative* — the
structures a chunker and a symbol extractor have to get right.
