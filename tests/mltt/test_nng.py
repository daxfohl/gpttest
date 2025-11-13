from mltt.ast import Id, NatType, Refl, Succ, Term, Univ, Var
from mltt.typing import type_check


def test_refl_proves_succ_self_equality() -> None:
    witness = Succ(Var(0))
    ty = NatType()
    proof = Refl(ty, witness)
    ctx: list[Term] = [ty]
    assert type_check(proof, Id(ty, witness, witness), ctx)
