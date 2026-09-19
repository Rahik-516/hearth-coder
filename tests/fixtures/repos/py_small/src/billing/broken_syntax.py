"""Deliberately invalid Python.

DO NOT FIX THIS FILE. The M1 acceptance criteria require that a file with a syntax error
still chunks (via the fallback window path) rather than failing the index run. Its
parse_status is expected to be "failed" or "partial".
"""


def unterminated_call(
    first,
    second
    # missing closing paren, and the body below is unreachable garbage


class Dangling:
    def method(self)
        return "missing colon above"
