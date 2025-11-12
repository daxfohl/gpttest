"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from .ast import App, Id, Lam, NatRec, NatType, Pi, Refl, Succ, Term, Var, Zero
from .eq import cong, sym, trans, ap
from .eval import normalize
from .typing import type_check


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
    return Lam(
        NatType(),
        Lam(
            NatType(),
            NatRec(
                P=Lam(NatType(), NatType()),
                z=Var(0),
                s=Lam(
                    NatType(),
                    Lam(NatType(), Succ(Var(0))),
                ),
                n=Var(1),
            ),
        ),
    )


def add_n_0() -> Term:
    """Proof that n + 0 == n"""
    add_term = add()
    return Lam(
        NatType(),  # n
        NatRec(
            # motive P(n) = Id Nat (add n 0) n
            P=Lam(
                NatType(),
                Id(NatType(), App(App(add_term, Var(0)), Zero()), Var(0)),
            ),
            # base: add 0 0 ≡ 0  ⇒ refl
            z=Refl(ty=NatType(), t=Zero()),
            # step: ih : Id Nat (add k 0) k  ⇒ need Id Nat (add (Succ k) 0) (Succ k)
            # definitional eqn: add (Succ k) 0 ≡ Succ (add k 0)
            # so use ap Succ ih : Id Nat (Succ (add k 0)) (Succ k)
            s=Lam(
                NatType(),  # k
                Lam(
                    Id(NatType(), App(App(add_term, Var(0)), Zero()), Var(0)),  # ih
                    ap(
                        f=Lam(NatType(), Succ(Var(0))),  # Succ as a function
                        A=NatType(),
                        B0=NatType(),
                        x=App(App(add_term, Var(1)), Zero()),  # add k 0
                        y=Var(1),  # k
                        p=Var(0),  # ih
                    ),
                ),
            ),
            n=Var(0),  # recurse on n
        ),
    )
