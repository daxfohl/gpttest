# Repository Agent Instructions

- Run `source .venv/bin/activate && pytest` before doing anything and note which tests are currently failing. Don't try to fix those tests unless they are directly related to the problem.
- Don't make any changes that the user did not specify.
- Prefer running tests with `source .venv/bin/activate && pytest` from the repository root unless a different command is specified.
- Keep documentation changes up to date with code changes.
- Follow PEP 8 style guidelines for Python code.
- Run `source .venv/bin/activate && mypy` when type-safety is affected or after non-trivial Python changes.
- Run `source .venv/bin/activate && black .` before sending changes that touch Python files.
- Use `source .venv/bin/activate && python` when running repo-local Python scripts.
- The `mltt` module implements a miniature Martin-Löf type theory with supporting modules for de Bruijn index manipulation (`debruijn.py`), normalization (`normalization.py`), typing (`typecheck.py`), equality helpers (`eq.py`), and structural predicates/utilities (`predicates.py`).
- De Bruijn strategy: binders push indices outward (0 = innermost). Context entries are stored relative to their tails and shifted on lookup; extending a context prepends new binders without rewriting existing entry types. Substitutions decrement indices above the target and shift the inserted term when descending under binders. Inductive parameters are outermost, followed by indices, then constructor arguments; utilities expect substitutions in that order and rely on consistent shifting.
- This codebase is not a public library—feel free to make breaking changes when needed; no backwards compatibility shims are required.
- Tests live under `tests/mltt/` and mirror the source tree—add or update the relevant modules when behaviour changes.
