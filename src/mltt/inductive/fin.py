"""Finite ordinals ``Fin n`` (natural numbers strictly less than ``n``)."""

from __future__ import annotations

from .nat import NatType, Succ, Zero, numeral
from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Term,
    Var,
)
from ..core.inductive_utils import apply_term, nested_lam

Fin = I(name="Fin", index_types=(NatType(),), level=0)
FZCtor = Ctor(
    name="FZ",
    inductive=Fin,
    arg_types=(NatType(),),  # n : Nat
    result_indices=(Succ(Var(0)),),
)
FSCtor = Ctor(
    name="FS",
    inductive=Fin,
    arg_types=(
        NatType(),  # n : Nat
        App(Fin, Var(0)),  # Fin n
    ),
    result_indices=(Succ(Var(1)),),
)
object.__setattr__(Fin, "constructors", (FZCtor, FSCtor))


def FinType(n: Term) -> Term:
    return App(Fin, n)


def FZ(n: Term) -> Term:
    return App(FZCtor, n)


def FS(n: Term, k: Term) -> Term:
    return apply_term(FSCtor, n, k)


def FinRec(P: Term, base: Term, step: Term, k: Term) -> Elim:
    """Recursor for ``Fin`` expressed via the generalized eliminator."""

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


def fin_modulus_term() -> Term:
    return nested_lam(
        NatType(),  # n
        FinType(Var(0)),  # x : Fin n
        body=Var(1),  # return n
    )


def fin_modulus(n: Term, k: Term) -> Term:
    return apply_term(fin_modulus_term(), n, k)


def fin_to_nat_term() -> Term:
    P = nested_lam(NatType(), FinType(Var(0)), body=NatType())
    base = nested_lam(NatType(), body=Zero())
    step = nested_lam(
        NatType(),
        FinType(Var(0)),
        apply_term(P, Var(1), Var(0)),
        body=Succ(Var(0)),
    )

    return nested_lam(
        NatType(),
        FinType(Var(0)),
        body=FinRec(P=P, base=base, step=step, k=Var(0)),
    )


def fin_to_nat(n: Term, k: Term) -> Term:
    return apply_term(fin_to_nat_term(), n, k)
