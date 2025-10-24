# Repository Agent Instructions

- Prefer running tests with `python -m pytest` from the repository root unless a different command is specified.
- Keep documentation changes concise and up to date with code changes.
- Follow PEP 8 style guidelines for Python code.
- Update or add unit tests alongside code changes when relevant.
- The `mltt` package implements a miniature Martin-Löf type theory with modules for the AST (`ast.py`), de Bruijn index manipulation (`debruijn.py`), evaluation (`eval.py`), typing (`typing.py`), and equality helpers (`eq.py`).
- Tests live under `tests/` and cover the evaluator, typing rules, equality combinators, and de Bruijn utilities—add or update tests there when behaviour changes.
