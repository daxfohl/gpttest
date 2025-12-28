"""All-vec predicate indexed by the actual vector (Option 2)."""

from __future__ import annotations

from functools import cache

from mltt.inductive.nat import NatType, Succ, Zero
from mltt.inductive.vec import ConsCtorAt, NilCtorAt, VecType
from mltt.core.ast import App, Pi, Term, Univ, Var
from mltt.core.debruijn import mk_app, Telescope, ArgList
from mltt.core.ind import Ctor, Elim, Ind


@cache
def _allvec_family(level: int) -> tuple[Ind, Ctor, Ctor]:
    nil_ctor = NilCtorAt(level)
    cons_ctor = ConsCtorAt(level)

    allvec_ind = Ind(
        name="AllVec",
        param_types=Telescope.of(
            Univ(level), Pi(Var(0), Univ(level))
        ),  # A : Type, P : A -> Type
        index_types=Telescope.of(
            NatType(), VecType(Var(2), Var(0), level=level)
        ),  # n : Nat, xs : Vec A n
        level=level,
    )

    all_nil_ctor = Ctor(
        name="AllNil",
        inductive=allvec_ind,
        result_indices=ArgList.of(
            Zero(),  # n = 0
            App(nil_ctor, Var(1)),  # xs = Nil A (A = Var(1) in (params)(fields))
        ),
    )

    all_cons_ctor = Ctor(
        name="AllCons",
        inductive=allvec_ind,
        field_schemas=Telescope.of(
            NatType(),  # n : Nat
            Var(2),  # x : A (context: (A,P,n))
            VecType(Var(3), Var(1), level=level),  # xs : Vec A n (context: (A,P,n,x))
            App(Var(3), Var(1)),  # px : P x (context: (A,P,n,x,xs))
            mk_app(allvec_ind, Var(5), Var(4), Var(3), Var(1)),  # ih : AllVec A P n xs
        ),
        result_indices=ArgList.of(
            Succ(Var(4)),  # S n
            mk_app(cons_ctor, Var(6), Var(4), Var(3), Var(2)),  # Cons A n x xs
        ),
    )

    object.__setattr__(allvec_ind, "constructors", (all_nil_ctor, all_cons_ctor))
    return allvec_ind, all_nil_ctor, all_cons_ctor


AllVec, AllNilCtor, AllConsCtor = _allvec_family(0)


def AllVecAt(level: int = 0) -> Ind:
    return _allvec_family(level)[0]


def AllNilCtorAt(level: int = 0) -> Ctor:
    return _allvec_family(level)[1]


def AllConsCtorAt(level: int = 0) -> Ctor:
    return _allvec_family(level)[2]


def AllVecType(A: Term, P: Term, n: Term, xs: Term, *, level: int = 0) -> Term:
    return mk_app(AllVecAt(level), A, P, n, xs)


def AllNil(A: Term, P: Term, *, level: int = 0) -> Term:
    return mk_app(AllNilCtorAt(level), A, P)


def AllCons(
    A: Term,
    P: Term,
    n: Term,
    x: Term,
    xs: Term,
    px: Term,
    ih: Term,
    *,
    level: int = 0,
) -> Term:
    return mk_app(AllConsCtorAt(level), A, P, n, x, xs, px, ih)


def AllVecElim(
    motive: Term, all_nil: Term, all_cons: Term, scrutinee: Term, *, level: int = 0
) -> Elim:
    return Elim(
        inductive=AllVecAt(level),
        motive=motive,
        cases=(all_nil, all_cons),
        scrutinee=scrutinee,
    )
