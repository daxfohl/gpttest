# gpttest

This repository contains a small Python library that models fragments of Martin-Löf type theory.  The `mltt` package exposes a minimal abstract syntax tree along with utilities for working with de Bruijn indices, evaluating terms, checking types, and constructing proofs of propositional equality.

## Project layout

- `mltt/ast.py` – dataclasses that encode the core language constructs such as lambda abstractions, dependent function and pair types, natural numbers, and identity types.
- `mltt/debruijn.py` – shifting and substitution helpers that operate on the AST using de Bruijn indices.
- `mltt/eval.py` – a small-step evaluator for normalising terms.
- `mltt/typing.py` – type-checking rules for the language and helpers for building typing contexts.
- `mltt/eq.py` – derived rules for propositional equality (congruence, symmetry, and transitivity).
- `tests/` – pytest-based unit tests that exercise the evaluator, typing rules, equality lemmas, and de Bruijn utilities.

## Installation on macOS

1. Install the Xcode Command Line Tools to obtain the developer toolchain and a recent version of `git`:

   ```bash
   xcode-select --install
   ```

2. Install [Homebrew](https://brew.sh), the macOS package manager, by running the script recommended on the Homebrew homepage:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Follow the post-install instructions printed by the script to add Homebrew to your shell profile (for example by appending `eval "$(/opt/homebrew/bin/brew shellenv)"` to `~/.zprofile`).

3. Use Homebrew to install Python 3.11 or newer:

   ```bash
   brew install python@3.11
   ```

   Ensure the newly installed interpreter is first on your `PATH` by adding a line such as `export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"` to your shell profile.

4. Clone this repository and create a virtual environment for local development:

   ```bash
   git clone https://github.com/<your-username>/gpttest.git
   cd gpttest
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

5. Upgrade `pip` and install the project in editable mode along with its development extras:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

## Development

The code targets Python 3.11+ and has no third-party dependencies once the interpreter is available.  Run the test suite from the repository root:

```bash
python -m pytest
```

Static type checking is configured with [mypy](https://mypy-lang.org/).  Run it against the package to ensure new code type checks cleanly:

```bash
python -m mypy
```

When extending the type theory or adding new utilities, prefer modelling constructs as pure functions on the existing dataclasses and accompany changes with tests that demonstrate the new behaviour.
