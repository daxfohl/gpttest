"""All-vec predicate indexed by the actual vector (Option 2)."""

from __future__ import annotations

from mltt.inductive.nat import NatType, Succ, Zero
from mltt.inductive.vec import ConsCtorAt, NilCtorAt, VecAt
from mltt.kernel.ast import App, Pi, Term, Univ, Var, UApp
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.telescope import mk_app, Telescope, ArgList


def _allvec() -> tuple[Ind, Ctor, Ctor]:
    u = LVar(0)
    nil_ctor = NilCtorAt(u)
    cons_ctor = ConsCtorAt(u)
    vec_head = VecAt(u)
    allvec_ind = Ind(
        name="AllVec",
        uarity=1,
        param_types=Telescope.of(
            Univ(u), Pi(Var(0), Univ(u))
        ),  # A : Type, P : A -> Type
        index_types=Telescope.of(
            NatType(),
            mk_app(vec_head, Var(2), Var(0)),
        ),  # n : Nat, xs : Vec A n
        level=u,
    )

    all_nil_ctor = Ctor(
        name="AllNil",
        inductive=allvec_ind,
        result_indices=ArgList.of(
            Zero(),  # n = 0
            App(nil_ctor, Var(1)),  # xs = Nil A (A = Var(1) in (params)(fields))
        ),
        uarity=1,
    )

    all_cons_ctor = Ctor(
        name="AllCons",
        inductive=allvec_ind,
        field_schemas=Telescope.of(
            NatType(),  # n : Nat
            Var(2),  # x : A (context: (A,P,n))
            mk_app(vec_head, Var(3), Var(1)),  # xs : Vec A n (context: (A,P,n,x))
            App(Var(3), Var(1)),  # px : P x (context: (A,P,n,x,xs))
            mk_app(
                UApp(allvec_ind, u), Var(5), Var(4), Var(3), Var(1)
            ),  # ih : AllVec A P n xs
        ),
        result_indices=ArgList.of(
            Succ(Var(4)),  # S n
            mk_app(cons_ctor, Var(6), Var(4), Var(3), Var(2)),  # Cons A n x xs
        ),
        uarity=1,
    )

    object.__setattr__(allvec_ind, "constructors", (all_nil_ctor, all_cons_ctor))
    return allvec_ind, all_nil_ctor, all_cons_ctor


AllVec_U, AllNil_U, AllCons_U = _allvec()


def AllVecAt(level: LevelExpr | int = 0) -> Term:
    return UApp(AllVec_U, level)


def AllNilCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(AllNil_U, level)


def AllConsCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(AllCons_U, level)


AllVec = AllVecAt()
AllNilCtor = AllNilCtorAt()
AllConsCtor = AllConsCtorAt()


def AllVecType(
    A: Term, P: Term, n: Term, xs: Term, *, level: LevelExpr | int = 0
) -> Term:
    return mk_app(AllVecAt(level), A, P, n, xs)


def AllNil(A: Term, P: Term, *, level: LevelExpr | int = 0) -> Term:
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
    level: LevelExpr | int = 0,
) -> Term:
    return mk_app(AllConsCtorAt(level), A, P, n, x, xs, px, ih)


def AllVecElim(motive: Term, all_nil: Term, all_cons: Term, scrutinee: Term) -> Elim:
    return Elim(
        inductive=AllVec_U,
        motive=motive,
        cases=(all_nil, all_cons),
        scrutinee=scrutinee,
    )
