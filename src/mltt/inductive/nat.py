"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from .maybe import MaybeType, Nothing, Just
from ..core.ast import App, Ctor, Elim, I, Lam, Term, Var
from ..core.utli import nested_lam
from ..core.util import apply_term

Nat = I(name="Nat", level=0)
ZeroCtor = Ctor(name="Zero", inductive=Nat)
SuccCtor = Ctor(name="Succ", inductive=Nat, arg_types=(Nat,))
object.__setattr__(Nat, "constructors", (ZeroCtor, SuccCtor))


def NatType() -> I:
    return Nat


def Zero() -> Term:
    return ZeroCtor


def Succ(n: Term) -> Term:
    return App(SuccCtor, n)


def NatRec(P: Term, base: Term, step: Term, n: Term) -> Elim:
    """Recursor for Nat expressed via the generalized inductive eliminator."""

    return Elim(
        inductive=Nat,
        motive=P,
        cases=(base, step),
        scrutinee=n,
    )


def numeral(value: int) -> Term:
    """Return the canonical term representing the natural number ``value``."""

    term: Term = Zero()
    for _ in range(value):
        term = Succ(term)
    return term


def add(lhs: Term, rhs: Term) -> Term:
    """
    add : Nat → Nat → Nat
    Addition by recursion on the first argument.

    add a b =
      NatRec (λ_. Nat) b (λ_ r. Succ r) a

    Rules:
      add Zero b = b
      add (Succ a) b = Succ (add a b)
    """

    return NatRec(
        P=Lam(NatType(), NatType()),
        base=rhs,
        step=nested_lam(
            NatType(),
            NatType(),
            body=Succ(Var(0)),
        ),
        n=lhs,
    )


def add_term() -> Term:
    """Build ``add lhs rhs`` as nested applications."""
    return nested_lam(NatType(), NatType(), body=add(Var(1), Var(0)))


def pred_maybe(n: Term) -> Term:
    return NatRec(
        P=Lam(NatType(), MaybeType(NatType())),
        base=Nothing(NatType()),
        step=nested_lam(
            NatType(),
            MaybeType(NatType()),
            body=Just(NatType(), Var(1)),
        ),
        n=n,
    )


def pred_maybe_term() -> Term:
    """Predecessor as ``Maybe Nat``: ``Nothing`` for zero, ``Just k`` for ``Succ k``."""

    return Lam(NatType(), pred_maybe(Var(0)))
