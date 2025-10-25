# gpttest

This repository contains a small Python library that models fragments of Martin-Löf type theory.  The `mltt` package exposes a minimal abstract syntax tree along with utilities for working with de Bruijn indices, evaluating terms, checking types, and constructing proofs of propositional equality.

## Project layout

- `mltt/ast.py` – dataclasses that encode the core language constructs such as lambda abstractions, dependent function and pair types, natural numbers, and identity types.
- `mltt/debruijn.py` – shifting and substitution helpers that operate on the AST using de Bruijn indices.
- `mltt/eval.py` – a small-step evaluator for normalising terms.
- `mltt/typing.py` – type-checking rules for the language and helpers for building typing contexts.
- `mltt/eq.py` – derived rules for propositional equality (congruence, symmetry, and transitivity).
- `tests/` – pytest-based unit tests that exercise the evaluator, typing rules, equality lemmas, and de Bruijn utilities.

## Development

The code targets Python 3.11+ and has no third-party dependencies.  Install an appropriate interpreter and then run the test suite from the repository root:

```bash
python -m pytest
```

Static type checking is configured with [mypy](https://mypy-lang.org/).  Run it against the package to ensure new code type checks cleanly:

```bash
python -m mypy
```

When extending the type theory or adding new utilities, prefer modelling constructs as pure functions on the existing dataclasses and accompany changes with tests that demonstrate the new behaviour.
