"""All-vec predicate indexed by the actual vector (Option 2)."""

from __future__ import annotations

from .nat import NatType, Succ, Zero
from .vec import Cons, Nil, VecType
from ..core.ast import App, Pi, Term, Univ, Var
from ..core.debruijn import mk_app, Telescope, ArgList
from ..core.ind import Ctor, Elim, Ind

AllVec = Ind(
    name="AllVec",
    param_types=Telescope.of(Univ(0), Pi(Var(0), Univ(0))),  # A : Type, P : A -> Type
    index_types=Telescope.of(
        NatType(), VecType(Var(2), Var(0))
    ),  # n : Nat, xs : Vec A n
)

AllNilCtor = Ctor(
    name="AllNil",
    inductive=AllVec,
    result_indices=ArgList.of(
        Zero(),  # n = 0
        Nil(Var(1)),  # xs = Nil A   (A = Var(1) in (params)(fields))
    ),
)

AllConsCtor = Ctor(
    name="AllCons",
    inductive=AllVec,
    field_schemas=Telescope.of(
        NatType(),  # n : Nat
        Var(2),  # x : A (context: (A,P,n))
        VecType(Var(3), Var(1)),  # xs : Vec A n (context: (A,P,n,x))
        App(Var(3), Var(1)),  # px : P x (context: (A,P,n,x,xs))
        mk_app(AllVec, Var(5), Var(4), Var(3), Var(1)),  # ih : AllVec A P n xs
    ),
    result_indices=ArgList.of(
        Succ(Var(4)),  # S n
        Cons(Var(6), Var(4), Var(3), Var(2)),  # Cons A n x xs
    ),
)

object.__setattr__(AllVec, "constructors", (AllNilCtor, AllConsCtor))


def AllVecType(A: Term, P: Term, n: Term, xs: Term) -> Term:
    return mk_app(AllVec, A, P, n, xs)


def AllNil(A: Term, P: Term) -> Term:
    return mk_app(AllNilCtor, A, P)


def AllCons(A: Term, P: Term, n: Term, x: Term, xs: Term, px: Term, ih: Term) -> Term:
    return mk_app(AllConsCtor, A, P, n, x, xs, px, ih)


def AllVecElim(motive: Term, all_nil: Term, all_cons: Term, scrutinee: Term) -> Elim:
    return Elim(
        inductive=AllVec, motive=motive, cases=(all_nil, all_cons), scrutinee=scrutinee
    )
