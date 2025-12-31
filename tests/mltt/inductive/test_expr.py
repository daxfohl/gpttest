from mltt.inductive.expr import Const, ConstCtor, ExprType, Pair, PairCtor
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.sigma import Sigma
from mltt.kernel.ast import Lam, Univ, Var
from mltt.kernel.tel import mk_app, mk_pis


def test_const_typechecks() -> None:
    Ty = Univ(0)
    tau = NatType()
    value = Zero()
    term = Const(Ty, tau, value)
    expected = ExprType(Ty, tau)
    term.type_check(expected)
    assert term.infer_type().type_equal(expected)


def test_pair_typechecks() -> None:
    Ty = Univ(0)
    A = NatType()
    B = NatType()
    lhs = Const(Ty, A, Zero())
    rhs = Const(Ty, B, Zero())
    term = Pair(Ty, A, B, lhs, rhs)
    expected_idx = mk_app(Sigma, A, Lam(A, B))
    expected = ExprType(Ty, expected_idx)
    term.type_check(expected)
    assert term.infer_type().type_equal(expected)


def test_ctor_types() -> None:
    expected_const = mk_pis(
        Univ(1),
        Var(0),  # τ : Ty
        Var(0),  # value : τ
        return_ty=ExprType(Var(2), Var(1)),
    )
    assert ConstCtor.infer_type().type_equal(expected_const)

    expected_pair = mk_pis(
        Univ(1),
        Var(0),  # A : Ty
        Var(1),  # B : Ty
        ExprType(Var(2), Var(1)),
        ExprType(Var(3), Var(1)),
        return_ty=ExprType(
            Var(4),  # Ty
            mk_app(Sigma, Var(3), Lam(Var(3), Var(2).shift(1))),
        ),
    )
    assert PairCtor.infer_type().type_equal(expected_pair)
