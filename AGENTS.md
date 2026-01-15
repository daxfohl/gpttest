# Repository Agent Instructions

- Run `source .venv/bin/activate && pytest` before doing anything and note which tests are currently failing. Don't try to fix those tests unless they are directly related to the problem.
- Don't make any changes that the user did not specify.
- Prefer running tests with `source .venv/bin/activate && pytest` from the repository root unless a different command is specified.
- Keep documentation changes up to date with code changes.
- Follow PEP 8 style guidelines for Python code.
- Run `source .venv/bin/activate && mypy` when type-safety is affected or after non-trivial Python changes.
- Run `source .venv/bin/activate && black .` before sending changes that touch Python files.
- Use `source .venv/bin/activate && python` when running repo-local Python scripts.
- Use `scripts/check.sh` to run `black`, `mypy`, and `pytest` together when you intend to run the full suite.
- The `mltt.kernel` module implements a miniature Martin-Löf type theory.
- The `mltt.ind` module implements several common inductives such as `Nat`, `Vec`, `Fin`, `Maybe`, `List`, `Bool`, `Sigma`, `Eq`, `Unit`, `Empty`, and others.
- The `mltt.surface` module implements a parser and elaborator for surface syntax.
- This codebase is not a public library—feel free to make breaking changes when needed; no backwards compatibility shims are required.
- Tests live under `tests/mltt/` and mirror the source tree—add or update the relevant modules when behaviour changes.
- Do not add `__init__.py` files.
- Always use absolute imports. Import classes and top-level functions fully, e.g. `from foo import Bar, bar`.
- Prefer `match` blocks to `isinstance` checks unless there are only 1-2 cases.
