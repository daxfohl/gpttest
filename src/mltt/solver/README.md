# Solver Layer

This package houses elaboration-time constraint solving. It operates on kernel
terms and does **not** depend on surface syntax.

## Responsibilities

- Store metas and constraints (`state.py`).
- Solve term equality constraints and assign metas (`solver.py`).
- Solve universe level constraints (`levels.py`).
- Zonk terms and levels (substitute solved metas).

## Interfaces used by elaboration

- `ElabState.fresh_meta` / `fresh_level_meta`
- `ElabState.add_constraint` / `add_level_constraint`
- `ElabState.solve` / `zonk` / `ensure_solved`

Elaboration builds terms, generates constraints, and delegates solving here.
