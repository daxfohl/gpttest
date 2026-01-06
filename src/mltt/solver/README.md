# Solver Layer

This package houses elaboration-time constraint solving. It operates on kernel
terms and does **not** depend on surface syntax.

## Responsibilities

- Store metas and constraints (`solver.py`).
- Solve term equality constraints and assign metas (`solver.py`).
- Solve universe level constraints (`levels.py`).
- Zonk terms and levels (substitute solved metas).

## Interfaces used by elaboration

- `Solver.fresh_meta` / `fresh_level_meta`
- `Solver.add_constraint` / `add_level_constraint`
- `Solver.solve` / `zonk` / `ensure_solved`

Elaboration builds terms, generates constraints, and delegates solving here.
