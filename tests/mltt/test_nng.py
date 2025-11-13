from mltt.ast import App, Id, Lam, NatType, Pi, Refl, Succ, Term, Var
from mltt.eq import ap
from mltt.nat import add, numeral
from mltt.typing import infer_type, type_check, type_equal


def test_refl_proves_succ_self_equality() -> None:
    witness = Succ(Var(0))
    ty = NatType()
    proof = Refl(ty, witness)
    ctx: list[Term] = [ty]
    assert type_check(proof, Id(ty, witness, witness), ctx)


def test_double_preserves_y_equals_x_plus_seven() -> None:
    add_term = add()
    two = numeral(2)
    seven = numeral(7)
    double = Lam(NatType(), App(App(add_term, Var(0)), Var(0)))

    lemma = Lam(
        NatType(),
        Lam(
            NatType(),
            Lam(
                Id(NatType(), Var(1), App(App(add_term, Var(2)), seven)),
                ap(
                    f=double,
                    A=NatType(),
                    B0=NatType(),
                    x=Var(1),
                    y=App(App(add_term, Var(2)), seven),
                    p=Var(0),
                ),
            ),
        ),
    )

    expected_type = Pi(
        NatType(),
        Pi(
            NatType(),
            Pi(
                Id(NatType(), Var(1), App(App(add_term, Var(2)), seven)),
                Id(
                    NatType(),
                    App(App(add_term, Var(1)), Var(1)),
                    App(App(add_term, App(App(add_term, Var(2)), seven)), App(App(add_term, Var(2)), seven)),
                ),
            ),
        ),
    )

    assert type_equal(infer_type(lemma), expected_type)
