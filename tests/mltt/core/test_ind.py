import mltt.inductive.vec as vec
from mltt.core.ast import Univ, Var
from mltt.core.debruijn import mk_app, mk_pis, mk_lams, discharge_binders, Telescope
from mltt.core.ind import Ctor, Elim, Ind
from mltt.inductive.nat import NatType, Succ, Zero


def test_instantiate_ctor_arg_types_shifts_params_by_fields() -> None:
    params = (Var(0),)  # pretend A is at depth 0 in Γ

    inst = tuple(
        discharge_binders(schema, params, depth_above=i)
        for i, schema in enumerate(vec.ConsCtor.field_schemas)
    )
    # n : Nat
    assert inst[0] == vec.NatType()
    # head : A at depth 1 (under previous field n)
    assert inst[1] == Var(1)
    # tail : Vec A n with A at depth 2 (under n, head)
    assert inst[2] == mk_app(vec.Vec, Var(2), Var(1))


# def test_ih_types_shift_indices_under_remaining_fields() -> None:
#     params = (Var(0),)  # A in Γ
#     inst = _instantiate_ctor_arg_types(vec.ConsCtor.arg_types, params)
#
#     motive = vec.nested_lam(
#         vec.NatType(), vec.VecType(Var(1), Var(0)), body=FinType(Succ(Var(1)))
#     )
#     params_in_fields_ctx = tuple(param.shift(len(inst)) for param in params)
#
#     ih_types = _build_ih_types(
#         ind=vec.Vec,
#         inst_arg_types=inst,
#         params_actual=params,
#         params_in_fields_ctx=params_in_fields_ctx,
#         motive=motive,
#     )
#
#     # In context Γ,(n, head, tail) the IH should refer to n at Var(2).
#     assert ih_types == [FinType(Succ(Var(2)))]


def test_iota_reduce_detects_whnf_recursive_fields() -> None:
    wrap = Ind(name="Wrap", level=0)
    base_ctor = Ctor(name="Base", inductive=wrap)
    wrap_ctor = Ctor(
        name="Wrap",
        inductive=wrap,
        field_schemas=Telescope.of(mk_app(mk_lams(Univ(0), body=Var(0)), wrap)),
    )
    object.__setattr__(wrap, "constructors", (base_ctor, wrap_ctor))

    motive = mk_lams(wrap, body=NatType())
    base_case = Zero()
    wrap_case = mk_lams(
        mk_app(mk_lams(Univ(0), body=Var(0)), wrap),
        mk_app(motive.shift(1), Var(0)),
        body=Succ(Var(0)),
    )

    scrutinee = mk_app(wrap_ctor, mk_app(base_ctor))
    elim = Elim(
        inductive=wrap, motive=motive, cases=(base_case, wrap_case), scrutinee=scrutinee
    )

    assert elim.normalize() == Succ(Zero())
    elim.type_check(NatType())


def test_iota_reduce_shares_instantiation_with_typing() -> None:
    wrap = Ind(name="WrapRec", level=0)
    z_ctor = Ctor(name="Z", inductive=wrap)
    s_ctor = Ctor(
        name="S",
        inductive=wrap,
        field_schemas=Telescope.of(mk_app(mk_lams(Univ(0), body=Var(0)), wrap)),
    )
    object.__setattr__(wrap, "constructors", (z_ctor, s_ctor))

    motive = mk_lams(wrap, body=NatType())
    z_case = Zero()
    s_case = mk_lams(
        mk_app(mk_lams(Univ(0), body=Var(0)), wrap),
        mk_app(motive.shift(1), Var(0)),
        body=Succ(Var(0)),
    )

    height = mk_lams(
        wrap,
        body=Elim(
            inductive=wrap, motive=motive, cases=(z_case, s_case), scrutinee=Var(0)
        ),
    )

    expected_ty = mk_pis(wrap, return_ty=NatType())
    height.type_check(expected_ty)

    scrutinee = mk_app(s_ctor, mk_app(z_ctor))
    assert mk_app(height, scrutinee).normalize() == Succ(Zero())
