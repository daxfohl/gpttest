"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    Ctor,
    Elim,
    Ind,
    Lam,
    Pi,
    Term,
    Univ,
    Var,
)
from .debruijn import Ctx, subst, shift
from .reduce.normalize import normalize
from .reduce.whnf import whnf


def expect_universe(term: Term, ctx: Ctx) -> int:
    """Return the universe level of ``term`` or raise if it is not a type.

    Infers ``term`` and reduces it to weak head normal form so universe
    annotations reflect canonical shapes, then enforces that the result is a
    ``Univ``.
    """
    ty = infer_type(term, ctx)
    ty = whnf(ty)
    if not isinstance(ty, Univ):
        raise TypeError(
            "Expected a universe:\n" f"  term = {term}\n" f"  inferred = {ty}"
        )
    return ty.level


def type_equal(t1: Term, t2: Term, ctx: Ctx | None = None) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` are convertible via head reduction."""

    ctx = ctx or Ctx()
    t1_whnf = whnf(t1)
    t2_whnf = whnf(t2)

    if t1_whnf == t2_whnf:
        return True

    match t1_whnf, t2_whnf:
        case (Pi(arg1, body1), Pi(arg2, body2)):
            return type_equal(arg1, arg2, ctx) and type_equal(
                body1, body2, ctx.prepend_each(arg1)
            )
        case (Lam(arg_ty1, body1), Lam(arg_ty2, body2)):
            return type_equal(arg_ty1, arg_ty2, ctx) and type_equal(
                body1, body2, ctx.prepend_each(arg_ty1)
            )
        case (App(f1, a1), App(f2, a2)):
            return type_equal(f1, f2, ctx) and type_equal(a1, a2, ctx)
        case (
            Elim(ind1, motive1, cases1, scrutinee1),
            Elim(ind2, motive2, cases2, scrutinee2),
        ) if (
            ind1 is ind2
        ):
            if len(cases1) != len(cases2):
                return False
            return (
                type_equal(motive1, motive2, ctx)
                and all(
                    type_equal(case1, case2, ctx)
                    for case1, case2 in zip(cases1, cases2, strict=True)
                )
                and type_equal(scrutinee1, scrutinee2, ctx)
            )
        case (Ctor() as ctor1, Ctor() as ctor2):
            return ctor1 is ctor2
        case (Ind() as ind1, Ind() as ind2):
            return ind1 is ind2

    return False


def infer_type(term: Term, ctx: Ctx | None = None) -> Term:
    """Infer the type of ``term`` under the optional De Bruijn context ``ctx``.

    Follows the syntax-directed typing rules; raises on ill-formed terms
    instead of returning ``None`` so callers don't silently accept mistakes.
    """

    ctx = ctx or Ctx()
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i < len(ctx):
                return shift(ctx[i].ty, i + 1)
            else:
                raise TypeError(f"Unbound variable {i}")
        case Lam(arg_ty, body):
            # Lambdas infer to Pis: infer the body under an extended context.
            body_ty = infer_type(body, ctx.prepend_each(arg_ty))
            return Pi(arg_ty, body_ty)
        case App(f, a):
            # Application: infer the function, ensure it is a Pi, and that the
            # argument checks against its domain.
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError(
                    "Application of non-function:\n"
                    f"  term = {term}\n"
                    f"  function = {f}\n"
                    f"  inferred f_ty = {f_ty}"
                )
            try:
                type_check(a, f_ty.arg_ty, ctx)
            except TypeError as exc:
                raise TypeError(
                    "Application argument type mismatch:\n"
                    f"  term = {term}\n"
                    f"  argument = {a}\n"
                    f"  expected arg_ty = {f_ty.arg_ty}\n"
                    f"  inferred arg_ty = {infer_type(a, ctx)}\n"
                    f"  inferred f_ty = {f_ty}"
                ) from exc
            return subst(f_ty.return_ty, a)
        case Pi(arg_ty, body):
            # Pi formation: both sides must be types; universe level is max.
            arg_level = expect_universe(arg_ty, ctx)
            body_level = expect_universe(body, ctx.prepend_each(arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case Ind():
            from .inductive_utils import infer_ind_type

            return infer_ind_type(ctx, term)
        case Ctor():
            from .inductive_utils import infer_ctor_type

            return infer_ctor_type(term)
        case Elim():
            from .inductive_utils import infer_elim_type

            return infer_elim_type(term, ctx)

    raise TypeError("Unexpected term in infer_type:\n" f"  term = {term!r}")


def type_check(term: Term, ty: Term, ctx: Ctx | None = None) -> None:
    """Raise ``TypeError`` if ``term`` is not well-typed with type ``ty``."""

    ctx = ctx or Ctx()
    expected_ty = whnf(ty)
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i >= len(ctx):
                raise TypeError(f"Unbound variable {i}")
            found_ty = shift(ctx[i].ty, i + 1)
            if not type_equal(found_ty, expected_ty, ctx):
                raise TypeError(
                    "Variable type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  found = {found_ty}"
                )
            return None
        case Lam(arg_ty, body):
            # Lambdas must check against a Pi; ensure domains align, then check
            # the body under the extended context.
            match expected_ty:
                case Pi(dom, cod):
                    # if arg_ty != NatType():  # DELETE ME!!!
                    #     raise ValueError(f"a={arg_ty}\nb={dom}\nctx={ctx}")

                    if not type_equal(arg_ty, dom, ctx):
                        raise TypeError(
                            "Lambda domain mismatch:\n"
                            f"  term = {term}\n"
                            f"  expected domain = {dom}\n"
                            f"  found domain = {arg_ty}"
                        )
                    ctx1 = ctx.prepend_each(arg_ty)
                    try:
                        type_check(body, cod, ctx1)
                    except TypeError as exc:
                        raise TypeError(
                            "Lambda body has wrong type:\n"
                            f"  term = {term}\n"
                            f"  expected codomain = {cod}\n"
                            f"  inferred body = {infer_type(body, ctx1)}"
                        ) from exc
                    return None
                case _:
                    raise TypeError(
                        "Lambda expected to have Pi type:\n"
                        f"  term = {term}\n"
                        f"  expected = {expected_ty}"
                    )
        case App(f, a):
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError(
                    "Application of non-function:\n"
                    f"  term = {term}\n"
                    f"  function = {f}\n"
                    f"  inferred f_ty = {f_ty}"
                )
            try:
                type_check(a, f_ty.arg_ty, ctx)
            except TypeError as exc:
                raise TypeError(
                    "Application argument type mismatch:\n"
                    f"  term = {term}\n"
                    f"  argument = {a}\n"
                    f"  expected arg_ty = {f_ty.arg_ty}\n"
                    f"  inferred arg_ty = {infer_type(a, ctx)}\n"
                    f"  inferred f_ty = {f_ty}"
                ) from exc
            inferred_ty = subst(f_ty.return_ty, a)
            if not type_equal(expected_ty, inferred_ty):
                raise TypeError(
                    "Application result type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Pi(_, _):
            # Pi formation uses inference for its type; just compare expected.
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Pi type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Ind():
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Inductive type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Ctor():
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Constructor type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Elim():
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Eliminator type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}\n"
                    f"  normalized expected = {normalize(expected_ty)}\n"
                    f"  normalized inferred = {normalize(inferred_ty)}"
                )
            return None
        case Univ(_):
            if not isinstance(expected_ty, Univ):
                raise TypeError(
                    "Universe type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}"
                )
            return None

    raise TypeError("Unexpected term in type_check:\n" f"  term = {term!r}")


__all__ = ["type_equal", "infer_type", "type_check", "expect_universe"]
