"""Sandboxing, by levels (docs/safety-and-tool-use.md §12).

L0 — what ships now: policy, approval, checkpoints, a scrubbed environment, timeouts and
process groups. Nothing runs unapproved, files are revertible, no credentials reach a
child, and nothing hangs. **This is not containment.** An approved command runs with the
user's privileges.

L1+ — Phase 3: `bwrap` on Linux/WSL2 and `sandbox-exec` on macOS, so an approved command
cannot reach the network or write outside the workspace. The WSL2 caveat is real and
recorded in §16.2: Windows interop escapes a Linux sandbox, so sandboxed runs require
interop disabled.

Structured as a package from the start so the L1 backends land beside the L0 runner rather
than replacing it. Defence in depth means adding layers, never swapping them.
"""
