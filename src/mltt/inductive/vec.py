"""Helpers for dependent length-indexed vectors."""

from __future__ import annotations

from functools import cache

from .fin import FinType, FZ, FS
from .nat import NatType, Succ, Zero
from ..core.ast import ConstLevel, LevelExpr, LevelLike, LevelVar, Term, Univ, Var
from ..core.debruijn import decompose_app, mk_app, mk_lams, Telescope, ArgList
from ..core.ind import Elim, Ctor, Ind


@cache
def _vec_family(level: LevelExpr) -> tuple[Ind, Ctor, Ctor]:
    vec_ind = Ind(
        name="Vec",
        param_types=Telescope.of(Univ(level)),
        index_types=Telescope.of(NatType()),
        level=level,
    )
    nil_ctor = Ctor(
        name="Nil",
        inductive=vec_ind,
        result_indices=ArgList.of(Zero()),
    )
    cons_ctor = Ctor(
        name="Cons",
        inductive=vec_ind,
        field_schemas=Telescope.of(
            NatType(),  # n : Nat
            Var(1),  # head : A
            mk_app(vec_ind, Var(2), Var(1)),  # tail : Vec A n
        ),
        result_indices=ArgList.of(Succ(Var(2))),  # result index = Succ n
    )
    object.__setattr__(vec_ind, "constructors", (nil_ctor, cons_ctor))
    return vec_ind, nil_ctor, cons_ctor


def _normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, LevelExpr):
        return level
    return ConstLevel(level)


Vec, NilCtor, ConsCtor = _vec_family(LevelVar(0))


def VecAt(level: LevelLike) -> Ind:
    return _vec_family(_normalize_level(level))[0]


def NilCtorAt(level: LevelLike) -> Ctor:
    return _vec_family(_normalize_level(level))[1]


def ConsCtorAt(level: LevelLike) -> Ctor:
    return _vec_family(_normalize_level(level))[2]


def VecType(elem_ty: Term, length: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = elem_ty.expect_universe()
    return mk_app(VecAt(level), elem_ty, length)


def Nil(elem_ty: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = elem_ty.expect_universe()
    return mk_app(NilCtorAt(level), elem_ty)


def Cons(
    elem_ty: Term, n: Term, head: Term, tail: Term, *, level: LevelLike | None = None
) -> Term:
    if level is None:
        level = elem_ty.expect_universe()
    return mk_app(ConsCtorAt(level), elem_ty, n, head, tail)


def VecElim(
    P: Term, base: Term, step: Term, xs: Term, *, level: LevelLike | None = None
) -> Elim:
    """Recursor for vectors."""

    if level is None:
        scrut_ty = xs.infer_type().whnf()
        head, _ = decompose_app(scrut_ty)
        if not isinstance(head, Ind):
            raise TypeError(f"VecElim scrutinee is not a Vec: {scrut_ty}")
        inductive = head
    else:
        inductive = VecAt(level)
    return Elim(
        inductive=inductive,
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
        VecType(Var(3), Var(0), level=0),  # xs : Vec A n (A is Var(3) in Γ,n,xs)
        body=FinType(Succ(Var(1))),  # Fin (Succ n)
    )
    step = mk_lams(
        NatType(),  # n
        Var(3),  # x : A
        VecType(Var(4), Var(1), level=0),  # xs : Vec A n
        mk_app(motive.shift(2), Var(2), Var(0)),  # ih : P n xs
        body=FS(Succ(Var(3)), Var(0)),  # Fin (Succ (Succ n))
    )

    return mk_lams(
        Univ(0),  # A
        NatType(),  # n
        VecType(Var(1), Var(0), level=0),  # xs : Vec A n
        body=VecElim(motive, FZ(Zero()), step, Var(0), level=0),
    )


def to_fin(elem_ty: Term, length: Term, xs: Term) -> Term:
    """Apply ``vec_to_fin_term`` to concrete arguments."""

    return mk_app(vec_to_fin_term(), elem_ty, length, xs)
