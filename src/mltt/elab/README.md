# mltt.elab

This layer elaborates surface AST nodes into kernel terms.

Capabilities:
- Elaborate surface terms to kernel terms with implicit arguments and metavariables.
- Solve term constraints and universe-level constraints; generalize implicit universes.
- Elaborate named arguments and enforce positional-before-named ordering.
- Elaborate partial application (`partial`) by producing lambdas for remaining arguments.
- Elaborate surface `match` into kernel eliminators with correct motive checking.
- Elaborate surface inductive definitions and constructors into kernel metadata.
- Track elaboration environments, implicit binder flags, and local/global types.
- Report elaboration errors as `ElabError` with source spans.

Non-goals:
- Parsing or desugaring surface syntax.
- Kernel normalization or definitional equality rules beyond what elaboration needs.

Entry points:
- Surface AST nodes in `sast.py` (elab methods).
- Elaborator AST in `east.py`.
- `ElabState` for metavariables and constraints.
- `ElabEnv` / `ElabType` for elaboration environment metadata.
