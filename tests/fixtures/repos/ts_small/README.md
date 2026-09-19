# ts_small

A small TypeScript shopping-cart fixture, mirroring `py_small` so retrieval and chunking
can be compared across languages.

| Path | Why it exists |
|---|---|
| `src/cart.ts` | `ShoppingCart.finalize` — the TS counterpart of `InvoiceService.finalize`. Class with methods, a getter, a nested arrow function, and a subclassed Error |
| `src/types.ts` | Interfaces, a type alias, a union type and an enum — the type-level declarations that carry most of the meaning in TS |
| `src/rateLimiter.ts` | `RateLimiter` — the vocabulary-gap case, mirroring `TokenBucket` |
| `src/components/CartBadge.tsx` | Exercises the `tsx` grammar, which is a separate parser from `typescript` |
| `src/index.js` | Plain JavaScript with `require`, so the `javascript` grammar is covered too |
| `tests/cart.test.ts` | Vitest conventions, for the test-writer workflow |
