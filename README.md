# gpttest

This repository contains a small Python library that models fragments of Martin-Löf type theory. The `mltt` package exposes a kernel AST and helpers for working with de Bruijn indices, normalization/WHNF, type checking, inductive definitions, and proof terms.

## Project layout

- `src/mltt/kernel/` – core kernel: AST, environments, normalization/WHNF, typing, and inductive eliminators.
- `src/mltt/inductive/` – library of inductive definitions (Nat, Bool, List, Vec, Fin, Sigma, etc.).
- `src/mltt/surface/` – surface syntax, parser, and elaboration into kernel terms (includes a prelude).
- `src/mltt/proofs/` – example proof terms (e.g., `add_comm`).
- `tests/mltt/` – pytest-based unit tests mirroring the source tree; they exercise the kernel, inductives, surface elaboration, and proofs.

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

4. Ask uv to install Python 3.14 and create a project-local virtual environment populated with the runtime and development dependencies (including `ply` for the surface parser):

   ```bash
   uv python install 3.14
   uv sync --extra dev
   ```

   The `uv sync` command creates a `.venv` directory in the repository root.  Activate it before running project commands:

   ```bash
   source .venv/bin/activate
   ```

The code targets Python 3.14+ and depends on `ply` for the surface parser, in addition to the development tooling. Run the test suite from the repository root:

```bash
pytest
```

Static type checking is configured with [mypy](https://mypy-lang.org/).  Run it against the package to ensure new code type checks cleanly:

```bash
mypy
```

When extending the type theory or adding new utilities, prefer modelling constructs as pure functions on the existing dataclasses and accompany changes with tests that demonstrate the new behaviour.
