Review the changes below as a careful colleague would. You are looking for problems the
author would want to know about before merging: bugs, missed edge cases, broken callers,
missing tests, security mistakes, and anything that will surprise the next reader.

Each line of the diff is shown with its line number in the **new** version of the file, then
`+` for an added line, `-` for a removed one, or a space for unchanged context. Removed lines
have no number.

### Rules

- **Cite every finding as `path:line`**, copying the path from the `===` header and the line
  number from the left margin. A finding without a citation cannot be acted on. Do not
  compute line numbers; copy them.
- Cite only lines you were shown. If a concern is about code outside the diff, say so and
  do not invent a location for it.
- Report what is wrong, in order of how much it matters. Do not praise, summarise the
  change, or restate the diff.
- Distinguish what you can see from what you suspect. "This divides by `count`, which is
  zero when the list is empty (`a.py:41`)" is a finding. "This might have performance
  issues" is not, unless you can say where and why.
- If you find nothing that matters, say so in one sentence. An empty review is a valid
  review; padding it with style nits is not.

Format each finding as:

    **<short title>** — `path:line`
    What is wrong and why it matters, in one or two sentences.

{{omitted}}
--- changes ---
{{diff}}
