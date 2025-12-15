"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Lam, Term, Var
from ..core.inductive_utils import apply_term, nested_lam

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


def add() -> Term:
    """
    add : Nat → Nat → Nat
    Addition by recursion on the first argument.

    add a b =
      NatRec (λ_. Nat) b (λ_ r. Succ r) a

    Rules:
      add Zero b = b
      add (Succ a) b = Succ (add a b)
    """
    return nested_lam(
        NatType(),
        NatType(),
        body=NatRec(
            P=Lam(NatType(), NatType()),
            base=Var(0),
            step=nested_lam(
                NatType(),
                NatType(),
                body=Succ(Var(0)),
            ),
            n=Var(1),
        ),
    )


def add_terms(lhs: Term, rhs: Term) -> Term:
    """Build ``add lhs rhs`` as nested applications."""

    return apply_term(add(), lhs, rhs)
