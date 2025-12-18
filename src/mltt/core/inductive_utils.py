"""Shared helpers for inductive types."""

from __future__ import annotations

from .ast import Ctor, Term, Var, Elim, Pi, Univ, App, Ind
from .debruijn import subst, shift, Ctx
from .reduce.normalize import normalize
from .reduce.whnf import whnf
from .util import nested_pi, apply_term, decompose_app


def _instantiate_ctor_arg_types(
    ctor_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
) -> tuple[Term, ...]:
    schemas: list[Term] = []
    p = len(params_actual)
    for i, schema in enumerate(ctor_arg_types):
        t = schema
        # eliminate param binders outermost → innermost at their indexed depth
        for s, param in enumerate(params_actual):
            index = i + p - s - 1
            t = subst(t, shift(param, i + (p - s - 1)), index)
        schemas.append(t)
    return tuple(schemas)


def _instantiate_ctor_result_indices_under_fields(
    result_indices: tuple[Term, ...],
    params_actual_shifted: tuple[Term, ...],  # already shifted by m at callsite
    m: int,  # number of ctor fields in scope
) -> tuple[Term, ...]:
    """
    result_indices schemas are written in context (params)(fields).
    We are currently in context Γ,(fields) (params already in Γ but shifted by m).
    Discharge params binders only; keep field vars (0..m-1) intact.
    """
    p = len(params_actual_shifted)
    out: list[Term] = []
    for schema in result_indices:
        t = schema
        # eliminate param binders outermost → innermost so inner indices stay stable
        for s, param in enumerate(params_actual_shifted):
            index = m + p - s - 1
            t = subst(t, shift(param, p - s - 1), index)
        out.append(t)
    return tuple(out)


def infer_ind_type(ctx: Ctx, ind: Ind) -> Term:
    """Compute the dependent function type of an inductive.

    The resulting Pi-tower has parameters outermost, then indices,
    finishing with the universe level.
    """
    from .typing import expect_universe

    for b in ind.all_binders:
        expect_universe(b, ctx)
        ctx = ctx.prepend_each(b)
    return nested_pi(*ind.all_binders, return_ty=Univ(ind.level))


def infer_ctor_type(ctor: Ctor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then constructor
    arguments, finishing with the inductive head applied to the result
    indices.
    """
    ind = ctor.inductive
    if len(ctor.result_indices) != len(ind.index_types):
        raise TypeError(
            "Constructor result indices must match inductive index arity:\n"
            f"  ctor = {ctor}\n"
            f"  expected arity = {len(ind.index_types)}\n"
            f"  found arity = {len(ctor.result_indices)}"
        )
    # Parameters bind outermost, then constructor arguments.
    #   [params][args] from outermost to innermost.
    offset = len(ctor.arg_types)
    param_vars = [
        Var(i) for i in reversed(range(offset, offset + len(ind.param_types)))
    ]
    return nested_pi(
        *ind.param_types,
        *ctor.arg_types,
        return_ty=apply_term(ctor.inductive, *param_vars, *ctor.result_indices),
    )


def infer_elim_type(elim: Elim, ctx: Ctx) -> Term:
    """Infer the type of an ``InductiveElim`` while checking its well-formedness."""
    # 1. Infer type of scrutinee and extract params/indices.
    from .typing import infer_type, type_equal, type_check, expect_universe

    scrut = elim.scrutinee
    ind = elim.inductive
    scrut_ty = whnf(infer_type(scrut, ctx))
    scrut_ty_head, scrut_ty_bindings = decompose_app(scrut_ty)
    if scrut_ty_head is not ind:
        raise TypeError(
            "Eliminator scrutinee not of the right inductive type:\n"
            f"  scrutinee = {scrut}\n"
            f"  expected head = {ind}\n"
            f"  found head = {scrut_ty_head}"
        )

    # 2.1 Partially apply motive to the actual indices
    motive = elim.motive
    p = len(ind.param_types)
    q = len(ind.index_types)
    params_actual = scrut_ty_bindings[:p]
    indices_actual = scrut_ty_bindings[p:]
    motive_applied = apply_term(motive, *indices_actual)

    # 2.2 Infer the type of this partially applied motive
    motive_applied_ty = whnf(infer_type(motive_applied, ctx))
    if not isinstance(motive_applied_ty, Pi):
        raise TypeError(
            "InductiveElim motive must take scrutinee after indices:\n"
            f"  motive = {motive}\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty = {motive_applied_ty}"
        )

    # 2.3 The scrutinee binder domain must match the scrutinee type
    scrut_dom = motive_applied_ty.arg_ty
    if not type_equal(scrut_dom, scrut_ty):
        raise TypeError(
            "InductiveElim motive scrutinee domain mismatch:\n"
            f"  expected scrut_ty = {scrut_ty}\n"
            f"  found scrut_dom = {scrut_dom}"
        )

    # 2.4 The motive codomain must be a universe
    body_ty = whnf(motive_applied_ty.return_ty)
    if not isinstance(body_ty, Univ):
        raise TypeError(
            "InductiveElim motive codomain must be a universe:\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty.return_ty = {motive_applied_ty.return_ty}\n"
            f"  normalized = {body_ty}"
        )

    # 3. For each constructor, compute the expected branch type and check
    for ctor, case in zip(ind.constructors, elim.cases, strict=True):
        # Context legend:
        #   Γ          = current context passed to this eliminator
        #   fields     = constructor arguments (outermost → innermost)
        #   ihs        = induction hypotheses for recursive fields
        #
        # Most ctor schemas are written in context (params)(fields). At this
        # point params are already fixed by the scrutinee in Γ, so we
        # instantiate only those param binders and keep fields as de Bruijn
        # variables.

        # 3.1 Instantiate ctor field types with the actual parameters.
        inst_arg_types = _instantiate_ctor_arg_types(ctor.arg_types, params_actual)
        m = len(inst_arg_types)

        # 3.2 Work in context Γ,fields: shift Γ-level params and motive by m.
        params_in_fields_ctx = tuple(shift(p, m) for p in params_actual)
        motive_in_fields_ctx = shift(motive, m)

        # Build a scrutinee-shaped term: C params field_vars.
        # field_vars are Var(m-1) .. Var(0) in the Γ,fields context.
        field_vars = tuple(Var(j) for j in reversed(range(m)))
        scrut_like = apply_term(ctor, *params_in_fields_ctx, *field_vars)

        # 3.3 Instantiate ctor result indices under fields (params only).
        result_indices_inst = _instantiate_ctor_result_indices_under_fields(
            ctor.result_indices, params_in_fields_ctx, m
        )

        # 3.4 Build IH types for recursive fields.
        # Each IH is in context Γ,fields,ihs_so_far, so shift by m + ri.
        recursive_field_positions = [
            j
            for j, field_ty in enumerate(inst_arg_types)
            if decompose_app(field_ty)[0] is ind
        ]
        r = len(recursive_field_positions)
        ih_types: list[Term] = []
        for ri, field_index in enumerate(recursive_field_positions):
            field_ty = inst_arg_types[field_index]
            _, field_args = decompose_app(field_ty)
            params_field = field_args[:p]
            indices_field = field_args[p : p + q]
            assert params_field == params_actual

            # field_index is outermost → innermost. Translate to de Bruijn index.
            field_var_index = m - 1 - field_index
            ih_type = apply_term(
                shift(motive, m + ri),
                *(shift(i, field_var_index + ri) for i in indices_field),
                Var(field_var_index + ri),
            )
            ih_types.append(ih_type)

        # 3.5 Branch codomain in Γ,fields; then shift for inserted IH binders.
        codomain_in_fields_ctx = apply_term(
            motive_in_fields_ctx, *result_indices_inst, scrut_like
        )
        codomain = shift(codomain_in_fields_ctx, r)

        # 3.6 Expected branch type: Π fields. Π ihs. codomain.
        telescope = (*inst_arg_types, *ih_types)
        branch_ty = nested_pi(*telescope, return_ty=codomain)
        try:
            type_check(case, branch_ty, ctx)
        except TypeError as exc:
            raise TypeError(
                "Case for constructor has wrong type:\n"
                f"  ctor = {ctor}\n"
                f"  case = {normalize(case)}\n"
                f"  expected = {normalize(branch_ty)}"
            ) from exc

    u = expect_universe(motive_applied_ty.return_ty, ctx)  # cod should be Univ(u)

    # target type is P i⃗_actual scrut
    target_ty = App(motive_applied, scrut)

    # sanity check target_ty really is a type in Type u (or ≤ u with cumulativity)
    _ = expect_universe(infer_type(target_ty, ctx), ctx)

    if u < ind.level:
        raise TypeError(
            "Eliminator motive returns too small a universe:\n"
            f"  motive level = {u}\n"
            f"  inductive level = {ind.level}"
        )

    return target_ty


__all__ = [
    "infer_elim_type",
    "infer_ctor_type",
    "infer_ind_type",
]
