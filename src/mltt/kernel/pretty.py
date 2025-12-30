"""Pretty-printing utilities for MLTT terms."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var
from mltt.kernel.environment import Const
from mltt.kernel.ind import Elim, Ctor, Ind

ATOM_PREC = 3
APP_PREC = 2
PI_PREC = 1
LAM_PREC = 0


def _fresh_name(env: list[str], base: str = "x") -> str:
    """Return a name not already present in ``env``."""

    candidate = base
    suffix = 0
    while candidate in env:
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
        case Let(arg_ty, value, body):
            return (
                _uses_var(arg_ty, target, depth)
                or _uses_var(value, target, depth)
                or _uses_var(body, target, depth + 1)
            )
        case App(f, a):
            return _uses_var(f, target, depth) or _uses_var(a, target, depth)
        case Elim(inductive, motive, cases, scrutinee):
            return (
                _uses_var(inductive, target, depth)
                or _uses_var(motive, target, depth)
                or any(_uses_var(case, target, depth) for case in cases)
                or _uses_var(scrutinee, target, depth)
            )
        case _:
            return False


def _maybe_paren(
    text: str, child_prec: int, parent_prec: int, *, allow_equal: bool
) -> str:
    if child_prec < parent_prec or (child_prec == parent_prec and not allow_equal):
        return f"({text})"
    return text


def _inductive_label(inductive: Ind) -> str:
    return inductive.name or "Inductive"


def _ctor_label(ctor: Ctor) -> str:
    if ctor.name:
        return ctor.name

    inductive = ctor.inductive
    for idx, candidate in enumerate(inductive.constructors):
        if candidate == ctor:
            return f"{_inductive_label(inductive)}.ctor{idx}"
    return _inductive_label(inductive)


def _render_app_like(head: str, args: list[tuple[str, int]]) -> tuple[str, int]:
    parts = [head]
    for arg_text, arg_prec in args:
        parts.append(_maybe_paren(arg_text, arg_prec, APP_PREC, allow_equal=False))
    return " ".join(parts), APP_PREC


def pretty(term: Term) -> str:
    """Return a human-friendly string for ``term``."""

    def fmt(t: Term, env: list[str]) -> tuple[str, int]:
        match t:
            case Var(k):
                name = env[k] if k < len(env) else f"#{k}"
                return name, ATOM_PREC

            case Univ(level):
                return ("Type" if level == 0 else f"Type{level}"), ATOM_PREC

            case Ind() as inductive:
                return _inductive_label(inductive), ATOM_PREC

            case Ctor() as ctor:
                return _ctor_label(ctor), ATOM_PREC

            case Const(name):
                return name, ATOM_PREC

            case App(f, a):
                func_text, func_prec = fmt(f, env)
                arg_text, arg_prec = fmt(a, env)
                func_disp = _maybe_paren(
                    func_text, func_prec, APP_PREC, allow_equal=True
                )
                arg_disp = _maybe_paren(arg_text, arg_prec, APP_PREC, allow_equal=False)
                return f"{func_disp} {arg_disp}", APP_PREC

            case Lam(arg_ty, body):
                binder = _fresh_name(env)
                arg_text, arg_prec = fmt(arg_ty, env)
                body_text, _ = fmt(body, [binder, *env])
                arg_disp = _maybe_paren(arg_text, arg_prec, PI_PREC, allow_equal=False)
                return f"\\{binder} : {arg_disp}. {body_text}", LAM_PREC

            case Pi(arg_ty, body):
                dependent = _uses_var(body, target=0)
                binder = _fresh_name(env, base="_" if not dependent else "x")
                arg_text, arg_prec = fmt(arg_ty, env)
                body_text, body_prec = fmt(body, [binder, *env])
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

            case Let(arg_ty, value, body):
                binder = _fresh_name(env)
                arg_text, arg_prec = fmt(arg_ty, env)
                value_text, value_prec = fmt(value, env)
                body_text, _ = fmt(body, [binder, *env])
                arg_disp = _maybe_paren(arg_text, arg_prec, PI_PREC, allow_equal=False)
                value_disp = _maybe_paren(
                    value_text, value_prec, APP_PREC, allow_equal=False
                )
                return (
                    f"let {binder} : {arg_disp} := {value_disp}; {body_text}",
                    LAM_PREC,
                )

            case Elim(inductive, motive, cases, scrutinee):
                motive_text, motive_prec = fmt(motive, env)
                scrutinee_text, scrutinee_prec = fmt(scrutinee, env)
                cases_text = ", ".join(fmt(case, env)[0] for case in cases)
                parts = [
                    f"elim {_inductive_label(inductive)}",
                    _maybe_paren(motive_text, motive_prec, APP_PREC, allow_equal=False),
                    f"[{cases_text}]",
                    _maybe_paren(
                        scrutinee_text, scrutinee_prec, APP_PREC, allow_equal=False
                    ),
                ]
                return " ".join(parts), APP_PREC

        raise TypeError(f"Cannot pretty-print unknown term: {t!r}")

    return fmt(term, [])[0]
