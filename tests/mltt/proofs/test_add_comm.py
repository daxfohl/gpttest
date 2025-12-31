from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Succ, Zero, add, numeral
from mltt.kernel.ast import Var
from mltt.kernel.tel import mk_app, mk_pis
from mltt.proofs.add_comm import (
    add_zero_left,
    add_zero_right,
    succ_add,
    add_succ_right,
    add_comm,
)


def test_add_zero_right_typechecks() -> None:
    lemma = add_zero_right()
    expected_ty = mk_pis(
        NatType(),
        return_ty=Id(NatType(), add(Var(0), Zero()), Var(0)),
    )
    assert lemma.infer_type().type_equal(expected_ty)


def test_add_zero_right_normalizes() -> None:
    lemma = add_zero_right()
    applied = mk_app(lemma, numeral(5))
    assert applied.normalize() == Refl(NatType(), numeral(5))


def test_add_zero_right_normalizes_multiple_inputs() -> None:
    lemma = add_zero_right()

    for value in range(6):
        applied = mk_app(lemma, numeral(value))
        assert applied.normalize() == Refl(NatType(), numeral(value))


def test_add_zero_right_applied_term_typechecks() -> None:
    lemma = add_zero_right()
    three = numeral(3)
    applied = mk_app(lemma, three)
    expected = Id(
        NatType(),
        add(three, Zero()),
        three,
    )

    applied.type_check(expected)


def test_add_zero_left_typechecks() -> None:
    lemma = add_zero_left()
    expected_ty = mk_pis(
        NatType(),
        return_ty=Id(NatType(), add(Zero(), Var(0)), Var(0)),
    )
    assert lemma.infer_type().type_equal(expected_ty)


def test_succ_add_typechecks() -> None:
    lemma = succ_add()
    expected_ty = mk_pis(
        NatType(),
        NatType(),
        return_ty=Id(
            NatType(),
            add(Succ(Var(1)), Var(0)),
            Succ(add(Var(1), Var(0))),
        ),
    )
    assert lemma.infer_type().type_equal(expected_ty)


def test_add_succ_right_typechecks() -> None:
    lemma = add_succ_right()
    expected_ty = mk_pis(
        NatType(),  # n
        NatType(),  # m
        return_ty=Id(
            NatType(),
            # this is add m (Succ n) in your definitional encoding:
            add(Var(0), Succ(Var(1))),  # note the swap
            Succ(add(Var(0), Var(1))),
        ),
    )
    assert lemma.infer_type().type_equal(expected_ty)


def test_add_comm_typechecks_and_examples() -> None:
    lemma = add_comm()
    expected_ty = mk_pis(
        NatType(),
        NatType(),
        return_ty=Id(NatType(), add(Var(1), Var(0)), add(Var(0), Var(1))),
    )
    assert lemma.infer_type().type_equal(expected_ty)

    m = numeral(2)
    n = numeral(3)
    proof = mk_app(lemma, m, n)
    expected_proof_ty = Id(NatType(), add(m, n), add(n, m))

    proof.type_check(expected_proof_ty)
