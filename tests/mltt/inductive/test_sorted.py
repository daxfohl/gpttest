from mltt.kernel.ast import Lam, Pi, Univ, Var, Term
from mltt.kernel.telescope import mk_app, mk_pis
from mltt.inductive.eq import Id, Refl
from mltt.inductive.list import Cons, List, ListAt, Nil
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.sorted import (
    SortedCons,
    SortedConsCtor,
    SortedAt,
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
    nil_proof.type_check(SortedType(A, R, Nil(A)))
    one_proof.type_check(SortedType(A, R, Cons(A, Zero(), Nil(A))))


def test_sorted_cons_typechecks() -> None:
    A = NatType()
    R = reflexive_relation(A)
    xs = Nil(A)
    rel = Refl(A, Zero())
    ih = SortedOne(A, R, Zero())
    proof = SortedCons(A, R, xs, Zero(), Zero(), rel, ih)
    expected = SortedType(A, R, Cons(A, Zero(), Cons(A, Zero(), xs)))
    proof.type_check(expected)
    assert proof.infer_type().type_equal(expected)


def test_sorted_ctor_types() -> None:
    expected_nil = mk_pis(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        return_ty=SortedType(Var(1), Var(0), Nil(Var(1))),
    )
    assert SortedNilCtor.infer_type().type_equal(expected_nil)

    expected_one = mk_pis(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        Var(1),
        return_ty=SortedType(Var(2), Var(1), Cons(Var(2), Var(0), Nil(Var(2)))),
    )
    assert SortedOneCtor.infer_type().type_equal(expected_one)

    expected_cons = mk_pis(
        Univ(0),
        Pi(Var(0), Pi(Var(1), Univ(0))),
        mk_app(List, Var(1)),  # xs
        Var(2),  # x
        Var(3),  # y
        mk_app(Var(3), Var(1), Var(0)),  # R x y
        SortedType(
            Var(5), Var(4), Cons(Var(5), Var(1), Var(3))
        ),  # ih: Sorted (y :: xs)
        return_ty=SortedType(
            Var(6),
            Var(5),
            Cons(Var(6), Var(3), Cons(Var(6), Var(2), Var(4))),
        ),
    )
    assert SortedConsCtor.infer_type().type_equal(expected_cons)


def test_infer_sorted_type_constructor_at_level() -> None:
    expected = mk_pis(
        Univ(1),
        Pi(Var(0), Pi(Var(1), Univ(1))),
        mk_app(ListAt(1), Var(1)),
        return_ty=Univ(1),
    )
    assert SortedAt(1).infer_type().type_equal(expected)
