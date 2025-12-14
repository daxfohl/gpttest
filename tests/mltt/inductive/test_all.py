from mltt.core.ast import Lam, Var
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.all import AllCons, AllNil, AllType
from mltt.inductive.list import Cons, Nil
from mltt.inductive.nat import NatType, Zero


def trivial_predicate() -> Lam:
    return Lam(NatType(), NatType())


def test_all_nil_typechecks() -> None:
    A = NatType()
    P = trivial_predicate()
    proof = AllNil(A, P)
    expected = AllType(A, P, Nil(A))
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)


def test_all_cons_typechecks() -> None:
    A = NatType()
    P = trivial_predicate()
    xs = Nil(A)
    x = Zero()
    px = Zero()
    ih = AllNil(A, P)
    proof = AllCons(A, P, xs, x, px, ih)
    expected_list = Cons(A, x, xs)
    expected = AllType(A, P, expected_list)
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)
