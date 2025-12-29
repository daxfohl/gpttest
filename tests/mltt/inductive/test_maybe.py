import pytest

from mltt.kernel.ast import Lam, Pi, Univ, Var, Term
from mltt.kernel.telescope import mk_pis
from mltt.inductive.maybe import (
    Just,
    JustCtor,
    Maybe,
    MaybeAt,
    MaybeElim,
    MaybeType,
    Nothing,
    NothingCtor,
)
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_maybe_type_constructor() -> None:
    assert Maybe.infer_type() == Pi(Univ(0), Univ(0))


def test_infer_maybe_type_constructor_at_level() -> None:
    assert MaybeAt(1).infer_type() == Pi(Univ(1), Univ(1))


def test_ctor_types() -> None:
    assert NothingCtor.infer_type() == Pi(Univ(0), MaybeType(Var(0)))
    assert JustCtor.infer_type() == mk_pis(Univ(0), Var(0), return_ty=MaybeType(Var(1)))


def test_maybe_rec_eliminates() -> None:
    elem_ty = NatType()
    maybe_nat = MaybeType(elem_ty)
    just_zero = Just(elem_ty, Zero())

    motive = Lam(maybe_nat, NatType())
    nothing_case = Zero()
    just_case = Lam(elem_ty, Succ(Var(0)))

    term_just = MaybeElim(motive, nothing_case, just_case, just_zero)
    term_nothing = MaybeElim(motive, nothing_case, just_case, Nothing(elem_ty))

    assert term_just.normalize() == Succ(Zero())
    assert term_nothing.normalize() == Zero()

    term_just.type_check(NatType())
    term_nothing.type_check(NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), NatType(), MaybeType(NatType()), Univ(0))
)
def test_infer_type(elem: Term) -> None:
    elem_ty = elem.infer_type()
    level = elem_ty.expect_universe()
    assert Nothing(elem_ty, level=level).infer_type() == MaybeType(elem_ty, level=level)
    assert Just(elem_ty, elem, level=level).infer_type() == MaybeType(
        elem_ty, level=level
    )
