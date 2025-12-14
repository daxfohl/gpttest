"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from .eq import Id, Refl, ap
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


def add_n_0() -> Term:
    """Proof that n + 0 == n"""
    return Lam(
        NatType(),  # n
        NatRec(
            # motive P(n) = Id Nat (add n 0) n
            P=Lam(
                NatType(),
                Id(NatType(), add_terms(Var(0), Zero()), Var(0)),
            ),
            # base: add 0 0 ≡ 0  ⇒ refl
            base=Refl(ty=NatType(), t=Zero()),
            # step: ih : Id Nat (add k 0) k  ⇒ need Id Nat (add (Succ k) 0) (Succ k)
            # definitional eqn: add (Succ k) 0 ≡ Succ (add k 0)
            # so use ap Succ ih : Id Nat (Succ (add k 0)) (Succ k)
            step=nested_lam(
                NatType(),  # k
                Id(NatType(), add_terms(Var(0), Zero()), Var(0)),  # ih
                body=ap(
                    f=Lam(NatType(), Succ(Var(0))),  # Succ as a function
                    A=NatType(),
                    B0=NatType(),
                    x=add_terms(Var(1), Zero()),  # add k 0
                    y=Var(1),  # k
                    p=Var(0),  # ih
                ),
            ),
            n=Var(0),  # recurse on n
        ),
    )
