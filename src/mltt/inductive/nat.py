"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from mltt.inductive.maybe import MaybeType, Nothing, Just
from mltt.kernel.ast import App, Lam, Term, Var
from mltt.kernel.telescope import mk_lams, Telescope
from mltt.kernel.ind import Elim, Ctor, Ind

Nat = Ind(name="Nat", level=0)
ZeroCtor = Ctor(name="Zero", inductive=Nat)
SuccCtor = Ctor(name="Succ", inductive=Nat, field_schemas=Telescope.of(Nat))
object.__setattr__(Nat, "constructors", (ZeroCtor, SuccCtor))


def NatType() -> Ind:
    return Nat


def Zero() -> Term:
    return ZeroCtor


def Succ(n: Term) -> Term:
    return App(SuccCtor, n)


def NatElim(P: Term, base: Term, step: Term, n: Term) -> Elim:
    return Elim(
        inductive=Nat,
        motive=P,
        cases=(base, step),
        scrutinee=n,
    )


def NatRec(A: Term, base: Term, step: Term, n: Term) -> Term:
    return NatElim(
        P=Lam(NatType(), A),
        base=base,
        step=step,
        n=n,
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
        A=NatType(),
        base=rhs,
        step=mk_lams(
            NatType(),
            NatType(),
            body=Succ(Var(0)),
        ),
        n=lhs,
    )


def add_term() -> Term:
    """Build ``add lhs rhs`` as nested applications."""
    return mk_lams(NatType(), NatType(), body=add(Var(1), Var(0)))


def pred_maybe(n: Term) -> Term:
    return NatRec(
        A=MaybeType(NatType()),
        base=Nothing(NatType()),
        step=mk_lams(
            NatType(),
            MaybeType(NatType()),
            body=Just(NatType(), Var(1)),
        ),
        n=n,
    )


def pred_maybe_term() -> Term:
    """Predecessor as ``Maybe Nat``: ``Nothing`` for zero, ``Just k`` for ``Succ k``."""

    return Lam(NatType(), pred_maybe(Var(0)))
