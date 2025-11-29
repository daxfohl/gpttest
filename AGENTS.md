# Repository Agent Instructions

- Prefer running tests with `pytest` from the repository root unless a different command is specified.
- Keep documentation changes concise and up to date with code changes.
- Follow PEP 8 style guidelines for Python code.
- Update or add unit tests alongside code changes when relevant.
- Run `mypy` when type-safety is affected or after non-trivial Python changes.
- Run `black .` before sending changes that touch Python files.
- The `mltt` module implements a miniature Martin-Löf type theory with supporting modules for de Bruijn index manipulation (`debruijn.py`), normalization (`normalization.py`), typing (`typecheck.py`), equality helpers (`eq.py`), and structural predicates/utilities (`predicates.py`).
- De Bruijn strategy: binders push indices outward (0 = innermost). When extending contexts, shift existing entries so outer references remain stable. Substitutions decrement indices above the target and shift the inserted term when descending under binders. Inductive parameters are outermost, followed by indices, then constructor arguments; utilities like `instantiate_params_indices` expect substitutions in that order and rely on consistent shifting.
- This codebase is not a public library—feel free to make breaking changes when needed; no backwards compatibility shims are required.
- Tests live under `tests/mltt/` and mirror the source tree—add or update the relevant modules when behaviour changes.
