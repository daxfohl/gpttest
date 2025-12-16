import pytest

from mltt.core.ast import Lam, Pi, Univ, Var, Term
from mltt.core.inductive_utils import nested_pi, apply_term
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.maybe import (
    Just,
    JustCtor,
    Maybe,
    MaybeRec,
    MaybeType,
    Nothing,
    NothingCtor,
)
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_maybe_type_constructor() -> None:
    assert infer_type(Maybe) == Pi(Univ(0), Univ(0))


def test_ctor_types() -> None:
    assert infer_type(NothingCtor) == Pi(Univ(0), MaybeType(Var(0)))
    assert infer_type(JustCtor) == nested_pi(
        Univ(0), Var(0), return_ty=MaybeType(Var(1))
    )


def test_maybe_rec_eliminates() -> None:
    elem_ty = NatType()
    maybe_nat = MaybeType(elem_ty)
    just_zero = Just(elem_ty, Zero())

    motive = Lam(maybe_nat, NatType())
    nothing_case = Zero()
    just_case = Lam(elem_ty, Succ(Var(0)))

    term_just = MaybeRec(motive, nothing_case, just_case, just_zero)
    term_nothing = MaybeRec(motive, nothing_case, just_case, Nothing(elem_ty))

    assert normalize(term_just) == Succ(Zero())
    assert normalize(term_nothing) == Zero()
    assert type_check(term_just, NatType())
    assert type_check(term_nothing, NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), NatType(), MaybeType(NatType()), Univ(0))
)
def test_infer_type(elem: Term) -> None:
    elem_ty = infer_type(elem)
    assert infer_type(Nothing(elem_ty)) == MaybeType(elem_ty)
    assert infer_type(Just(elem_ty, elem)) == MaybeType(elem_ty)
