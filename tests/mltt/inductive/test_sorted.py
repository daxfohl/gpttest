from mltt.core.ast import Lam, Pi, Univ, Var, Term
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.core.util import apply_term, nested_pi
from mltt.inductive.eq import Id, Refl
from mltt.inductive.list import Cons, List, Nil
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.sorted import (
    SortedCons,
    SortedConsCtor,
    SortedNil,
    SortedNilCtor,
    SortedOne,
    SortedOneCtor,
    SortedType,
)


def reflexive_relation(A: Term) -> Lam:
    return Lam(A, Lam(A, Id(A, Var(1), Var(0))))


def test_sorted_nil_and_one_typecheck() -> None:
    A = NatType()
    R = reflexive_relation(A)
    nil_proof = SortedNil(A, R)
    one_proof = SortedOne(A, R, Zero())
    type_check(nil_proof, SortedType(A, R, Nil(A)))
    type_check(one_proof, SortedType(A, R, Cons(A, Zero(), Nil(A))))


def test_sorted_cons_typechecks() -> None:
    A = NatType()
    R = reflexive_relation(A)
    xs = Nil(A)
    rel = Refl(A, Zero())
    ih = SortedOne(A, R, Zero())
    proof = SortedCons(A, R, xs, Zero(), Zero(), rel, ih)
    expected = SortedType(A, R, Cons(A, Zero(), Cons(A, Zero(), xs)))
    type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)


def test_sorted_ctor_types() -> None:
    expected_nil = nested_pi(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        return_ty=SortedType(Var(1), Var(0), Nil(Var(1))),
    )
    assert type_equal(infer_type(SortedNilCtor), expected_nil)

    expected_one = nested_pi(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        Var(1),
        return_ty=SortedType(Var(2), Var(1), Cons(Var(2), Var(0), Nil(Var(2)))),
    )
    assert type_equal(infer_type(SortedOneCtor), expected_one)

    expected_cons = nested_pi(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        apply_term(List, Var(1)),  # xs
        Var(2),  # x
        Var(3),  # y
        apply_term(Var(3), Var(1), Var(0)),  # R x y
        SortedType(
            Var(5), Var(4), Cons(Var(5), Var(1), Var(3))
        ),  # ih: Sorted (y :: xs)
        return_ty=SortedType(
            Var(6),
            Var(5),
            Cons(Var(6), Var(3), Cons(Var(6), Var(2), Var(4))),
        ),
    )
    assert type_equal(infer_type(SortedConsCtor), expected_cons)
