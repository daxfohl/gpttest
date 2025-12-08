"""Pretty-printing utilities for MLTT terms."""

from __future__ import annotations

from .ast import (
    App,
    Id,
    IdElim,
    Ctor,
    Elim,
    I,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
)

ATOM_PREC = 3
APP_PREC = 2
PI_PREC = 1
LAM_PREC = 0


def _fresh_name(ctx: list[str], base: str = "x") -> str:
    """Return a name not already present in ``ctx``."""

    candidate = base
    suffix = 0
    while candidate in ctx:
        suffix += 1
        candidate = f"{base}{suffix}"
    return candidate


def _uses_var(term: Term, target: int, depth: int = 0) -> bool:
    """Return ``True`` if ``Var(target)`` appears free in ``term``."""

    match term:
        case Var(k):
            return k == target + depth
        case Lam(ty, body) | Pi(ty, body):
            return _uses_var(ty, target, depth) or _uses_var(body, target, depth + 1)
        case App(a, f):
            return _uses_var(f, target, depth) or _uses_var(a, target, depth)
        case Elim(inductive, motive, cases, scrutinee):
            return (
                _uses_var(inductive, target, depth)
                or _uses_var(motive, target, depth)
                or any(_uses_var(case, target, depth) for case in cases)
                or _uses_var(scrutinee, target, depth)
            )
        case Id(ty, lhs, rhs):
            return (
                _uses_var(ty, target, depth)
                or _uses_var(lhs, target, depth)
                or _uses_var(rhs, target, depth)
            )
        case Refl(ty, t):
            return _uses_var(ty, target, depth) or _uses_var(t, target, depth)
        case IdElim(A, x, P, d, y, p):
            return any(_uses_var(part, target, depth) for part in (A, x, P, d, y, p))
        case _:
            return False


def _maybe_paren(
    text: str, child_prec: int, parent_prec: int, *, allow_equal: bool
) -> str:
    if child_prec < parent_prec or (child_prec == parent_prec and not allow_equal):
        return f"({text})"
    return text


def _inductive_label(inductive: I) -> str:
    return inductive.name or "Inductive"


def _ctor_label(ctor: Ctor) -> str:
    if ctor.name:
        return ctor.name

    inductive = ctor.inductive
    for idx, candidate in enumerate(inductive.constructors):
        if candidate is ctor:
            return f"{_inductive_label(inductive)}.ctor{idx}"
    return _inductive_label(inductive)


def _render_app_like(head: str, args: list[tuple[str, int]]) -> tuple[str, int]:
    parts = [head]
    for arg_text, arg_prec in args:
        parts.append(_maybe_paren(arg_text, arg_prec, APP_PREC, allow_equal=False))
    return " ".join(parts), APP_PREC


def pretty(term: Term) -> str:
    """Return a human-friendly string for ``term``."""

    def fmt(t: Term, ctx: list[str]) -> tuple[str, int]:
        match t:
            case Var(k):
                name = ctx[k] if k < len(ctx) else f"#{k}"
                return name, ATOM_PREC

            case Univ(level):
                return ("Type" if level == 0 else f"Type{level}"), ATOM_PREC

            case I() as inductive:
                return _inductive_label(inductive), ATOM_PREC

            case Ctor() as ctor:
                return _ctor_label(ctor), ATOM_PREC

            case App(a, f):
                func_text, func_prec = fmt(f, ctx)
                arg_text, arg_prec = fmt(a, ctx)
                func_disp = _maybe_paren(
                    func_text, func_prec, APP_PREC, allow_equal=True
                )
                arg_disp = _maybe_paren(arg_text, arg_prec, APP_PREC, allow_equal=False)
                return f"{func_disp} {arg_disp}", APP_PREC

            case Lam(arg_ty, body):
                binder = _fresh_name(ctx)
                arg_text, arg_prec = fmt(arg_ty, ctx)
                body_text, _ = fmt(body, [binder, *ctx])
                arg_disp = _maybe_paren(arg_text, arg_prec, PI_PREC, allow_equal=False)
                return f"\\{binder} : {arg_disp}. {body_text}", LAM_PREC

            case Pi(arg_ty, body):
                dependent = _uses_var(body, target=0)
                binder = _fresh_name(ctx, base="_" if not dependent else "x")
                arg_text, arg_prec = fmt(arg_ty, ctx)
                body_text, body_prec = fmt(body, [binder, *ctx])
                arg_disp = _maybe_paren(arg_text, arg_prec, PI_PREC, allow_equal=False)
                if not dependent:
                    body_disp = _maybe_paren(
                        body_text, body_prec, PI_PREC, allow_equal=True
                    )
                    return f"{arg_disp} -> {body_disp}", PI_PREC
                body_disp = _maybe_paren(
                    body_text, body_prec, PI_PREC, allow_equal=True
                )
                return f"Pi {binder} : {arg_disp}. {body_disp}", PI_PREC

            case Elim(inductive, motive, cases, scrutinee):
                motive_text, motive_prec = fmt(motive, ctx)
                scrutinee_text, scrutinee_prec = fmt(scrutinee, ctx)
                cases_text = ", ".join(fmt(case, ctx)[0] for case in cases)
                parts = [
                    f"elim {_inductive_label(inductive)}",
                    _maybe_paren(motive_text, motive_prec, APP_PREC, allow_equal=False),
                    f"[{cases_text}]",
                    _maybe_paren(
                        scrutinee_text, scrutinee_prec, APP_PREC, allow_equal=False
                    ),
                ]
                return " ".join(parts), APP_PREC

            case Id(ty, lhs, rhs):
                return _render_app_like(
                    "Id",
                    [
                        fmt(ty, ctx),
                        fmt(lhs, ctx),
                        fmt(rhs, ctx),
                    ],
                )

            case Refl(ty, t0):
                return _render_app_like(
                    "refl",
                    [
                        fmt(ty, ctx),
                        fmt(t0, ctx),
                    ],
                )

            case IdElim(A, x, P, d, y, p):
                return _render_app_like(
                    "J",
                    [
                        fmt(A, ctx),
                        fmt(x, ctx),
                        fmt(P, ctx),
                        fmt(d, ctx),
                        fmt(y, ctx),
                        fmt(p, ctx),
                    ],
                )

        raise TypeError(f"Cannot pretty-print unknown term: {t!r}")

    return fmt(term, [])[0]


__all__ = ["pretty"]
