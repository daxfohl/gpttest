# mltt.surface

This layer is responsible for surface syntax and sugar only.

Capabilities:
- Parse surface syntax into a surface AST.
- Parse `let`, `fun`, `match`, `partial`, implicit binders (`impl`), named arguments, and universe applications.
- Support type-argument sugar via `<A, B>` and universe application via `@{...}`.
- Desugar equation-style recursion into match-with-IH form where applicable.
- Desugar multi-scrutinee and tuple-pattern matches into nested matches.

Non-goals:
- Type checking, constraint solving, or elaboration.
- Kernel normalization or definitional equality.

Entry points:
- `parse_term` / `parse_term_raw` for parsing.
- `desugar` for surface-level rewrites.
