"""Helpers for dependent length-indexed vectors."""

from __future__ import annotations

from mltt.inductive.fin import FinType, FZ, FS
from mltt.inductive.nat import NatType, Succ, Zero
from mltt.kernel.ast import Term, Univ, Var, UApp
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.tel import mk_app, mk_uapp, mk_lams, Telescope, ArgList


def _vec() -> tuple[Ind, Ctor, Ctor]:
    u = LVar(0)
    vec_ind = Ind(
        name="Vec",
        uarity=1,
        param_types=Telescope.of(Univ(u)),
        index_types=Telescope.of(NatType()),
        level=u,
    )
    nil_ctor = Ctor(
        name="Nil",
        inductive=vec_ind,
        result_indices=ArgList.of(Zero()),
        uarity=1,
    )
    cons_ctor = Ctor(
        name="Cons",
        inductive=vec_ind,
        field_schemas=Telescope.of(
            NatType(),  # n : Nat
            Var(1),  # head : A
            mk_uapp(vec_ind, (u,), Var(2), Var(1)),  # tail : Vec A n
        ),
        result_indices=ArgList.of(Succ(Var(2))),  # result index = Succ n
        uarity=1,
    )
    object.__setattr__(vec_ind, "constructors", (nil_ctor, cons_ctor))
    return vec_ind, nil_ctor, cons_ctor


Vec_U, Nil_U, Cons_U = _vec()


def VecAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Vec_U, level)


def NilCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Nil_U, level)


def ConsCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Cons_U, level)


Vec = VecAt()
NilCtor = NilCtorAt()
ConsCtor = ConsCtorAt()


def VecType(elem_ty: Term, length: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(VecAt(level), elem_ty, length)


def Nil(elem_ty: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(NilCtorAt(level), elem_ty)


def Cons(
    elem_ty: Term, n: Term, head: Term, tail: Term, *, level: LevelExpr | int = 0
) -> Term:
    return mk_app(ConsCtorAt(level), elem_ty, n, head, tail)


def VecElim(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    return Elim(
        inductive=Vec_U,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )


def vec_to_fin_term() -> Term:
    """
    Π A. Π n. Vec A n -> Fin (Succ n)

    Converts a length-indexed vector into an inhabitant of ``Fin (Succ n)`` by
    recursion on the vector, incrementing the induction hypothesis in the
    ``Cons`` branch.
    """

    motive = mk_lams(
        NatType(),  # n
        VecType(Var(3), Var(0)),  # xs : Vec A n (A is Var(3) in Γ,n,xs)
        body=FinType(Succ(Var(1))),  # Fin (Succ n)
    )
    step = mk_lams(
        NatType(),  # n
        Var(3),  # x : A
        VecType(Var(4), Var(1)),  # xs : Vec A n
        mk_app(motive.shift(2), Var(2), Var(0)),  # ih : P n xs
        body=FS(Succ(Var(3)), Var(0)),  # Fin (Succ (Succ n))
    )

    return mk_lams(
        Univ(0),  # A
        NatType(),  # n
        VecType(Var(1), Var(0)),  # xs : Vec A n
        body=VecElim(motive, FZ(Zero()), step, Var(0)),
    )


def to_fin(elem_ty: Term, length: Term, xs: Term) -> Term:
    """Apply ``vec_to_fin_term`` to concrete arguments."""

    return mk_app(vec_to_fin_term(), elem_ty, length, xs)
