import mltt.inductive.list as listm
from mltt.core.ast import App, Lam, Pi, Univ, Var
from mltt.inductive.nat import NatType, Succ, Zero
from mltt.core.normalization import normalize
from mltt.core.typing import infer_type, type_check


def test_infer_list_type_constructor() -> None:
    assert infer_type(listm.List) == Pi(Univ(0), Univ(0))


def test_list_nil_and_cons_type_check() -> None:
    elem_ty = NatType()
    nil_nat = listm.Nil(elem_ty)
    assert type_check(nil_nat, listm.ListType(elem_ty))

    cons_nat = listm.Cons(elem_ty, Zero(), nil_nat)
    assert type_check(cons_nat, listm.ListType(elem_ty))


def test_listrec_length_of_singleton() -> None:
    elem_ty = NatType()
    list_ty = listm.ListType(elem_ty)
    xs = listm.Cons(elem_ty, Zero(), listm.Nil(elem_ty))
    P = Lam(list_ty, NatType())
    base = Zero()
    step = Lam(
        elem_ty,
        Lam(
            list_ty,
            Lam(
                App(P, Var(0)),
                Succ(Var(0)),
            ),
        ),
    )

    length_term = listm.ListRec(elem_ty, P, base, step, xs)

    assert type_check(length_term, NatType())
    assert normalize(length_term) == Succ(Zero())
