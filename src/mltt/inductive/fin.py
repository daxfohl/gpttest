"""Finite ordinals ``Fin n`` (natural numbers strictly less than ``n``)."""

from __future__ import annotations

from mltt.inductive.nat import NatType, Succ, Zero, numeral
from mltt.core.ast import (
    App,
    Term,
    Var,
)
from mltt.core.debruijn import mk_app, mk_lams, Telescope, ArgList
from mltt.core.ind import Elim, Ctor, Ind

Fin = Ind(name="Fin", index_types=Telescope.of(NatType()), level=0)
FZCtor = Ctor(
    name="FZ",
    inductive=Fin,
    field_schemas=Telescope.of(NatType()),  # n : Nat
    result_indices=ArgList.of(Succ(Var(0))),
)
FSCtor = Ctor(
    name="FS",
    inductive=Fin,
    field_schemas=Telescope.of(
        NatType(),  # n : Nat
        App(Fin, Var(0)),  # Fin n
    ),
    result_indices=ArgList.of(Succ(Var(1))),
)
object.__setattr__(Fin, "constructors", (FZCtor, FSCtor))


def FinType(n: Term) -> Term:
    return App(Fin, n)


def FZ(n: Term) -> Term:
    return App(FZCtor, n)


def FS(n: Term, k: Term) -> Term:
    return mk_app(FSCtor, n, k)


def FinElim(P: Term, base: Term, step: Term, k: Term) -> Elim:
    return Elim(
        inductive=Fin,
        motive=P,
        cases=(base, step),
        scrutinee=k,
    )


def of_int(i: int, n: int) -> Term:
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 <= i < n:
        raise ValueError("i is out of range")
    if i == 0:
        return FZ(numeral(n - 1))
    return FS(numeral(n - 1), of_int(i - 1, n - 1))


def fin_modulus(n: Term) -> Term:
    return n


def fin_modulus_term() -> Term:
    return mk_lams(
        NatType(),  # n
        FinType(Var(0)),  # x : Fin n
        body=fin_modulus(Var(1)),  # return n
    )


def fin_to_nat_term() -> Term:
    return mk_lams(
        NatType(),
        FinType(Var(0)),
        body=fin_to_nat(Var(0)),  # return n
    )


def fin_to_nat(k: Term) -> Term:
    P = mk_lams(NatType(), FinType(Var(0)), body=NatType())
    base = mk_lams(NatType(), body=Zero())
    step = mk_lams(
        NatType(),
        FinType(Var(0)),
        mk_app(P, Var(1), Var(0)),
        body=Succ(Var(0)),
    )

    return FinElim(P=P, base=base, step=step, k=k)
