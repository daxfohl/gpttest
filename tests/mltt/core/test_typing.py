import pytest

from mltt.kernel.ast import App, Lam, Pi, Term, Univ, Var
from mltt.kernel.telescope import mk_app, mk_lams
from mltt.kernel.environment import Env
from mltt.inductive.eq import Id, IdElim, Refl
from mltt.inductive.nat import NatRec, NatType, Zero, add, numeral


def test_infer_var() -> None:
    with pytest.raises(IndexError, match="Unbound variable"):
        assert Var(0).infer_type()
    t = NatType()
    assert Var(0).infer_type(Env.of(t)) == t


def test_infer_lam() -> None:
    assert Lam(Var(0), Var(0)).infer_type() == Pi(Var(0), Var(1))
    assert Lam(Var(10), Var(0)).infer_type() == Pi(Var(10), Var(11))
    assert Lam(Univ(0), Var(0)).infer_type() == Pi(Univ(0), Univ(0))
    assert Lam(Univ(10), Var(0)).infer_type() == Pi(Univ(10), Univ(10))
    with pytest.raises(IndexError, match="Unbound variable"):
        Lam(Var(0), Var(1)).infer_type()
    with pytest.raises(IndexError, match="Unbound variable"):
        Lam(Var(10), Var(1)).infer_type()
    with pytest.raises(IndexError, match="Unbound variable"):
        Lam(Univ(0), Var(1)).infer_type()
    with pytest.raises(IndexError, match="Unbound variable"):
        Lam(Univ(10), Var(1)).infer_type()
    assert Lam(Var(0), Univ(0)).infer_type() == Pi(Var(0), Univ(1))
    assert Lam(Var(10), Univ(0)).infer_type() == Pi(Var(10), Univ(1))
    assert Lam(Univ(0), Univ(0)).infer_type() == Pi(Univ(0), Univ(1))
    assert Lam(Univ(10), Univ(0)).infer_type() == Pi(Univ(10), Univ(1))
    assert Lam(Var(0), Univ(10)).infer_type() == Pi(Var(0), Univ(11))
    assert Lam(Var(10), Univ(10)).infer_type() == Pi(Var(10), Univ(11))
    assert Lam(Univ(0), Univ(10)).infer_type() == Pi(Univ(0), Univ(11))
    assert Lam(Univ(10), Univ(10)).infer_type() == Pi(Univ(10), Univ(11))


def test_infer_lam_ctx() -> None:
    def infer(t: Term) -> Term:
        return t.infer_type(Env.of(Univ(100)))

    assert infer(Lam(NatType(), Var(0))) == Pi(NatType(), NatType())
    assert infer(Lam(NatType(), Zero())) == Pi(NatType(), NatType())
    assert infer(Lam(NatType(), NatType())) == Pi(NatType(), Univ(0))
    assert infer(Lam(NatType(), Var(1))) == Pi(NatType(), Univ(100))
    assert infer(Lam(Var(0), Var(0))) == Pi(Var(0), Var(1))
    assert infer(Lam(Var(10), Var(0))) == Pi(Var(10), Var(11))
    assert infer(Lam(Univ(0), Var(0))) == Pi(Univ(0), Univ(0))
    assert infer(Lam(Univ(10), Var(0))) == Pi(Univ(10), Univ(10))
    assert infer(Lam(Var(0), Var(1))) == Pi(Var(0), Univ(100))
    assert infer(Lam(Var(10), Var(1))) == Pi(Var(10), Univ(100))
    assert infer(Lam(Univ(0), Var(1))) == Pi(Univ(0), Univ(100))
    assert infer(Lam(Univ(10), Var(1))) == Pi(Univ(10), Univ(100))
    assert infer(Lam(Var(0), Univ(0))) == Pi(Var(0), Univ(1))
    assert infer(Lam(Var(10), Univ(0))) == Pi(Var(10), Univ(1))
    assert infer(Lam(Univ(0), Univ(0))) == Pi(Univ(0), Univ(1))
    assert infer(Lam(Univ(10), Univ(0))) == Pi(Univ(10), Univ(1))
    assert infer(Lam(Var(0), Univ(10))) == Pi(Var(0), Univ(11))
    assert infer(Lam(Var(10), Univ(10))) == Pi(Var(10), Univ(11))
    assert infer(Lam(Univ(0), Univ(10))) == Pi(Univ(0), Univ(11))
    assert infer(Lam(Univ(10), Univ(10))) == Pi(Univ(10), Univ(11))


@pytest.mark.parametrize("i", [0, 2])
def test_infer_lam_4_level(i: int) -> None:
    # i==0: let f x y = y
    # i==2: let f x y = x
    fxy = Lam(
        arg_ty=Univ(10),
        body=Lam(
            arg_ty=Var(0),
            body=Lam(
                arg_ty=Univ(20),
                body=Lam(
                    arg_ty=Var(0),
                    body=Var(i),
                ),
            ),
        ),
    )
    t = App(App(App(App(fxy, Univ(9)), Univ(8)), Univ(19)), Univ(18))
    assert t.normalize() == Univ(18 if i == 0 else 8)
    assert t.infer_type() == Univ(19 if i == 0 else 9)


@pytest.mark.parametrize("i", [0, 1])
def test_infer_lam_3_level(i: int) -> None:
    # i==0: let f (x:A) (y:A) = y
    # i==1: let f (x:A) (y:A) = x
    fxy = Lam(
        arg_ty=Univ(10),
        body=Lam(
            arg_ty=Var(0),
            body=Lam(
                arg_ty=Var(1),
                body=Var(i),
            ),
        ),
    )
    t = App(App(App(fxy, Univ(9)), Univ(8)), Univ(8))
    assert t.normalize() == Univ(8)
    assert t.infer_type() == Univ(9)


def test_two_level_lambda_type_refers_to_previous_binder() -> None:
    """
    λ (A : Type₀). λ (x : A). x

    In de Bruijn:
      Lam(ty = Univ(0),
          body = Lam(ty = Var(0),   # A
                     body = Var(0))) # x

    This only type-checks if, when we prepend the context with A and then x:A,
    the internal context extension + shifting are correct.
    """

    term = Lam(
        arg_ty=Univ(level=0),  # Γ ⊢ Type₀ : Type₁ (ignored here)
        body=Lam(
            arg_ty=Var(k=0),  # x : A   (Var(0) refers to A)
            body=Var(k=0),  # body = x
        ),
    )

    # Empty context
    env = Env()

    ty = term.infer_type(env)

    # Expected type: Π (A : Type₀). Π (x : A). A
    # De Bruijn: Pi(Univ(0), Pi(Var(0), Var(1)))
    assert isinstance(ty, Pi)
    assert ty.arg_ty == Univ(level=0)  # domain of outer Pi is Type₀

    inner = ty.return_ty
    assert isinstance(inner, Pi)
    # domain of inner Pi is A (which is Var(0) in the outer Pi's scope)
    assert inner.arg_ty == Var(k=0)
    # codomain is also A
    assert inner.return_ty == Var(k=1)


def test_type_equal_normalizes_beta_equivalent_terms() -> None:
    beta_equiv = App(Lam(Univ(), Var(0)), Univ())

    assert beta_equiv.type_equal(Univ())
    assert not beta_equiv.type_equal(NatType())


def test_type_universe_levels_are_indexed() -> None:
    assert Univ().infer_type() == Univ(1)
    assert Univ(2).infer_type() == Univ(3)


def test_type_check_universe_levels_are_cumulative() -> None:
    Univ(0).type_check(Univ(1))
    Univ(0).type_check(Univ(2))
    Univ(1).type_check(Univ(3))

    with pytest.raises(TypeError, match="Universe level mismatch"):
        Univ(0).type_check(Univ(0))
    with pytest.raises(TypeError, match="Universe level mismatch"):
        Univ(1).type_check(Univ(1))


def test_infer_type_of_lambda_returns_pi_type() -> None:
    term = Lam(NatType(), Var(0))

    assert term.infer_type() == Pi(NatType(), NatType())


def test_infer_type_of_pi_uses_maximum_universe_level() -> None:
    assert Pi(NatType(), NatType()).infer_type() == Univ(0)
    higher = Pi(Univ(), NatType())
    assert higher.infer_type() == Univ(1)
    cod_dominates = Pi(NatType(), Univ(1))
    assert cod_dominates.infer_type() == Univ(2)


def test_infer_type_application_requires_function() -> None:
    with pytest.raises(TypeError, match="Application of non-function"):
        App(Zero(), Zero()).infer_type()


def test_type_check_natrec_rejects_invalid_base_case() -> None:
    A = NatType()
    z = Univ()
    s = Zero()
    n = Zero()
    term = NatRec(A, z, s, n)

    with pytest.raises(TypeError, match="Universe type mismatch"):
        term.type_check(A)


def test_type_check_accepts_add_application() -> None:
    term = add(numeral(2), numeral(3))

    term.type_check(NatType())


def test_type_check_lambda_with_wrong_domain() -> None:
    term = Lam(NatType(), Var(0))
    expected = Pi(Univ(), NatType())
    with pytest.raises(TypeError, match="Lambda domain mismatch"):
        term.type_check(expected)


def test_type_check_application_argument_mismatch() -> None:
    f = Lam(NatType(), Var(0))
    term = App(f, Univ())
    with pytest.raises(TypeError, match="Universe type mismatch"):
        term.type_check(NatType())


def test_infer_type_idelim() -> None:
    P = mk_lams(NatType(), Id(NatType(), Zero(), Var(0)), body=Univ())
    term = IdElim(
        P=P,
        d=NatType(),
        p=Refl(NatType(), Zero()),
    )
    inferred = term.infer_type()
    assert inferred == mk_app(P, Zero(), Refl(NatType(), Zero()))
