"""All-vec predicate indexed by the actual vector (Option 2)."""

from __future__ import annotations

from functools import cache

from .nat import NatType, Succ, Zero
from .vec import Cons, Nil, VecAt, VecType
from ..core.ast import (
    App,
    ConstLevel,
    LevelExpr,
    LevelLike,
    LevelVar,
    Pi,
    Term,
    Univ,
    Var,
)
from ..core.debruijn import ArgList, Telescope, decompose_app, mk_app
from ..core.ind import Ctor, Elim, Ind


@cache
def _allvec_family(level: LevelExpr) -> tuple[Ind, Ctor, Ctor]:
    vec_ind = VecAt(level)
    all_vec = Ind(
        name="AllVec",
        param_types=Telescope.of(
            Univ(level), Pi(Var(0), Univ(level))
        ),  # A : Type, P : A -> Type
        index_types=Telescope.of(
            NatType(), VecType(Var(2), Var(0), level=level)
        ),  # n : Nat, xs : Vec A n
        level=level,
    )
    all_nil = Ctor(
        name="AllNil",
        inductive=all_vec,
        result_indices=ArgList.of(
            Zero(),  # n = 0
            Nil(Var(1), level=level),  # xs = Nil A   (A = Var(1) in (params)(fields))
        ),
    )
    all_cons = Ctor(
        name="AllCons",
        inductive=all_vec,
        field_schemas=Telescope.of(
            NatType(),  # n : Nat
            Var(2),  # x : A (context: (A,P,n))
            VecType(Var(3), Var(1), level=level),  # xs : Vec A n
            App(Var(3), Var(1)),  # px : P x (context: (A,P,n,x,xs))
            mk_app(all_vec, Var(5), Var(4), Var(3), Var(1)),  # ih : AllVec A P n xs
        ),
        result_indices=ArgList.of(
            Succ(Var(4)),  # S n
            Cons(Var(6), Var(4), Var(3), Var(2), level=level),  # Cons A n x xs
        ),
    )
    object.__setattr__(all_vec, "constructors", (all_nil, all_cons))
    return all_vec, all_nil, all_cons


def _normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, LevelExpr):
        return level
    return ConstLevel(level)


AllVec, AllNilCtor, AllConsCtor = _allvec_family(LevelVar(0))


def AllVecAt(level: LevelLike) -> Ind:
    return _allvec_family(_normalize_level(level))[0]


def AllNilCtorAt(level: LevelLike) -> Ctor:
    return _allvec_family(_normalize_level(level))[1]


def AllConsCtorAt(level: LevelLike) -> Ctor:
    return _allvec_family(_normalize_level(level))[2]


def AllVecType(
    A: Term, P: Term, n: Term, xs: Term, *, level: LevelLike | None = None
) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(AllVecAt(level), A, P, n, xs)


def AllNil(A: Term, P: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = A.expect_universe()
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
    level: LevelLike | None = None,
) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(AllConsCtorAt(level), A, P, n, x, xs, px, ih)


def AllVecElim(
    motive: Term,
    all_nil: Term,
    all_cons: Term,
    scrutinee: Term,
    *,
    level: LevelLike | None = None,
) -> Elim:
    if level is None:
        scrut_ty = scrutinee.infer_type().whnf()
        head, _ = decompose_app(scrut_ty)
        if not isinstance(head, Ind):
            raise TypeError(f"AllVecElim scrutinee is not an AllVec: {scrut_ty}")
        inductive = head
    else:
        inductive = AllVecAt(level)
    return Elim(
        inductive=inductive,
        motive=motive,
        cases=(all_nil, all_cons),
        scrutinee=scrutinee,
    )
