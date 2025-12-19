import mltt.inductive.vec as vec
from mltt.core.util import apply_term
from mltt.core.ast import Var
from mltt.core.inductive_utils import _build_ih_types, _instantiate_ctor_arg_types
from mltt.inductive.fin import FinType
from mltt.inductive.nat import Succ


def test_instantiate_ctor_arg_types_shifts_params_by_fields() -> None:
    params = (Var(0),)  # pretend A is at depth 0 in Γ

    inst = _instantiate_ctor_arg_types(vec.ConsCtor.arg_types, params)
    # n : Nat
    assert inst[0] == vec.NatType()
    # head : A at depth 1 (under previous field n)
    assert inst[1] == Var(1)
    # tail : Vec A n with A at depth 2 (under n, head)
    assert inst[2] == apply_term(vec.Vec, Var(2), Var(1))


def test_ih_types_shift_indices_under_remaining_fields() -> None:
    params = (Var(0),)  # A in Γ
    inst = _instantiate_ctor_arg_types(vec.ConsCtor.arg_types, params)

    motive = vec.nested_lam(
        vec.NatType(), vec.VecType(Var(1), Var(0)), body=FinType(Succ(Var(1)))
    )
    params_in_fields_ctx = tuple(param.shift(len(inst)) for param in params)

    ih_types = _build_ih_types(
        ind=vec.Vec,
        inst_arg_types=inst,
        params_actual=params,
        params_in_fields_ctx=params_in_fields_ctx,
        motive=motive,
    )

    # In context Γ,(n, head, tail) the IH should refer to n at Var(2).
    assert ih_types == [FinType(Succ(Var(2)))]
