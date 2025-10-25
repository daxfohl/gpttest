# gpttest

This repository contains a small Python library that models fragments of Martin-Löf type theory.  The `mltt` package exposes a minimal abstract syntax tree along with utilities for working with de Bruijn indices, evaluating terms, checking types, and constructing proofs of propositional equality.

## Project layout

- `src/mltt/ast.py` – dataclasses that encode the core language constructs such as lambda abstractions, dependent function and pair types, natural numbers, and identity types.
- `src/mltt/debruijn.py` – shifting and substitution helpers that operate on the AST using de Bruijn indices.
- `src/mltt/eval.py` – a small-step evaluator for normalising terms.
- `src/mltt/typing.py` – type-checking rules for the language and helpers for building typing contexts.
- `src/mltt/eq.py` – derived rules for propositional equality (congruence, symmetry, and transitivity).
- `tests/` – pytest-based unit tests that exercise the evaluator, typing rules, equality lemmas, and de Bruijn utilities.

## Development

### macOS setup with uv

1. Install the Xcode Command Line Tools to obtain the developer toolchain and a recent version of `git`:

   ```bash
   xcode-select --install
   ```

2. Install [uv](https://docs.astral.sh/uv/)—which provides Python runtimes, virtual environments, and dependency management—using the official installer:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Follow the post-install instructions printed by the script (for example, adding `$HOME/.local/bin` to your `PATH`) so the `uv` command is available in new shells.

3. Clone this repository and move into the project directory:

   ```bash
   git clone https://github.com/<your-username>/gpttest.git
   cd gpttest
   ```

4. Ask uv to install Python 3.14 and create a project-local virtual environment populated with the development dependencies:

   ```bash
   uv python install 3.14
   uv sync --extra dev
   ```

   The `uv sync` command creates a `.venv` directory in the repository root.  Activate it before running project commands:

   ```bash
   source .venv/bin/activate
   ```

The code targets Python 3.14+ and has no third-party dependencies beyond the development tooling.  Run the test suite from the repository root:

```bash
python -m pytest
```

Static type checking is configured with [mypy](https://mypy-lang.org/).  Run it against the package to ensure new code type checks cleanly:

```bash
python -m mypy
```

When extending the type theory or adding new utilities, prefer modelling constructs as pure functions on the existing dataclasses and accompany changes with tests that demonstrate the new behaviour.
