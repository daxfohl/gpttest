from mltt.core.ast import Lam, Var
from mltt.core.debruijn import Ctx
from mltt.core.util import nested_pi, nested_lam
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.eq import Id, Refl, ap
from mltt.inductive.nat import NatType, Succ, add, numeral


def test_refl_proves_succ_self_equality() -> None:
    witness = Succ(Var(0))
    ty = NatType()
    proof = Refl(ty, witness)
    ctx = Ctx.as_ctx(ty)

    type_check(proof, Id(ty, witness, witness), ctx)


def test_double_preserves_y_equals_x_plus_seven() -> None:
    seven = numeral(7)
    # double = λz. z + z, expressed via the primitive add operator.
    # We use add rather than Nat multiplication because only addition primitives exist.
    double = Lam(NatType(), add(Var(0), Var(0)))

    # Build a lemma with the following structure:
    #   λx. λy. λp : Id Nat y (x+7). ap double p
    # so that any proof of y = x + 7 can be turned into a proof that double y = double (x+7).
    # Because `ap` already packages the standard non-dependent congruence rule,
    # the actual body is just an invocation of `ap` with the appropriate substitutions.
    lemma = nested_lam(
        NatType(),  # 1st λ: x : Nat
        NatType(),  # 2nd λ: y : Nat
        Id(
            NatType(),
            Var(0),  # y
            add(Var(1), seven),  # x + 7
        ),  # 3rd λ: p : Id(Nat, y, x+7)
        body=ap(
            f=double,
            A=NatType(),
            B0=NatType(),
            x=Var(1),  # y
            y=add(Var(2), seven),  # x + 7
            p=Var(0),  # p : y = x+7
        ),
    )

    # The inferred type of `lemma` should be the iterated Pi corresponding to the
    # english statement: for all x, y, and proofs that y = x + 7, the doubled values are equal.
    # We write out the Pi tower explicitly so the test checks that infer_type produces it.
    expected_type = nested_pi(
        NatType(),
        NatType(),
        Id(NatType(), Var(0), add(Var(1), seven)),
        return_ty=Id(
            NatType(),
            add(Var(1), Var(1)),
            add(
                add(Var(2), seven),
                add(Var(2), seven),
            ),
        ),
    )

    assert type_equal(infer_type(lemma), expected_type)
