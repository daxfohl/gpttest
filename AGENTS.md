# Repository Agent Instructions

- Prefer running tests with `python -m pytest` from the repository root unless a different command is specified.
- Keep documentation changes concise and up to date with code changes.
- Follow PEP 8 style guidelines for Python code.
- Update or add unit tests alongside code changes when relevant.
- Run `python -m mypy` when type-safety is affected or after non-trivial Python changes.
- Run `python -m black .` (or `uv run black .`) before sending changes that touch Python files.
- The `mltt` module implements a miniature Martin-Löf type theory with supporting modules for de Bruijn index manipulation (`debruijn.py`), reduction (`beta_reduce.py`), typing (`typecheck.py`), and equality helpers (`eq.py`).
- This codebase is not a public library—feel free to make breaking changes when needed; no backwards compatibility shims are required.
- Tests live under `tests/mltt/` and mirror the source tree—add or update the relevant modules when behaviour changes.
