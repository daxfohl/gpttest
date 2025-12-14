from mltt.core.ast import Lam, Pi, Univ, Var
from mltt.core.debruijn import shift
from mltt.core.inductive_utils import apply_term, nested_pi
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.expr import Const, ConstCtor, ExprType, Pair, PairCtor
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.sigma import Sigma


def test_const_typechecks() -> None:
    Ty = Univ(0)
    tau = NatType()
    value = Zero()
    term = Const(Ty, tau, value)
    expected = ExprType(Ty, tau)
    assert type_check(term, expected)
    assert type_equal(infer_type(term), expected)


def test_pair_typechecks() -> None:
    Ty = Univ(0)
    A = NatType()
    B = NatType()
    lhs = Const(Ty, A, Zero())
    rhs = Const(Ty, B, Zero())
    term = Pair(Ty, A, B, lhs, rhs)
    expected_idx = apply_term(Sigma, A, Lam(A, B))
    expected = ExprType(Ty, expected_idx)
    assert type_check(term, expected)
    assert type_equal(infer_type(term), expected)


def test_ctor_types() -> None:
    expected_const = nested_pi(
        Univ(0),
        Var(0),  # τ : Ty
        Var(0),  # value : τ
        return_ty=ExprType(Var(2), Var(1)),
    )
    assert type_equal(infer_type(ConstCtor), expected_const)

    expected_pair = nested_pi(
        Univ(0),
        Var(0),  # index τ = A × B
        Var(1),  # A : Ty
        Var(2),  # B : Ty
        ExprType(Var(3), Var(1)),
        ExprType(Var(4), Var(1)),
        return_ty=ExprType(
            Var(5),  # Ty
            apply_term(Sigma, Var(3), Lam(Var(3), shift(Var(2), 1))),
        ),
    )
    assert type_equal(infer_type(PairCtor), expected_pair)
