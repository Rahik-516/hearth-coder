Write tests for {{target}}.

### Where and how

- Write the tests to `{{test_path}}`, using the `{{framework}}` conventions of this project.
- Notes about that location:
{{notes}}

{{sample}}

### What to do

1. Read `{{source_path}}` first. Test what the code *does*, which you can only know by reading
   it — not what its name suggests.
2. Write the test file with `write_file` (or `edit_file` if it already exists). Cover the normal
   case, the edges the code visibly handles (empty input, zero, boundaries) and the failures it
   raises or returns.
3. Stop. **Do not run the tests yourself.** They are run for you after you finish, and if any
   fail you will be shown exactly which and why.

### Rules

- **Do not edit `{{source_path}}` or any other non-test file.** Your task is to test the code as
  it is. If a test you believe is correct fails because the code looks wrong, say so in your
  final message instead of changing the code.
- Assert on behaviour, not on implementation. A test that mirrors the code line for line proves
  nothing and breaks on every refactor.
- Tests must be deterministic and offline: no network, no reliance on the current time or on
  ordering, no files outside a temporary directory.
- Follow the sample's imports and naming. Do not add dependencies.
- Keep it proportionate. A handful of meaningful tests beats thirty near-duplicates.

Finish with two or three sentences: which test file you wrote and what it covers.
