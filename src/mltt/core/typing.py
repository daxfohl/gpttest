"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    Ctor,
    Elim,
    I,
    Lam,
    Pi,
    Term,
    Univ,
    Var,
)
from .debruijn import Ctx, subst, shift
from .inductive_utils import (
    apply_term,
    decompose_app,
    nested_pi,
    instantiate_ctor_arg_types,
    instantiate_ctor_result_indices_under_fields,
)
from .reduce import normalize
from .reduce.whnf import whnf


def _ctor_type(ctor: Ctor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then constructor
    arguments, finishing with the inductive head applied to the result
    indices.
    """
    ind = ctor.inductive
    if len(ctor.result_indices) != len(ind.index_types):
        raise TypeError("Constructor result indices must match inductive index arity")
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


def _infer_inductive_elim(elim: Elim, ctx: Ctx) -> Term:
    """Infer the type of an ``InductiveElim`` while checking its well-formedness."""
    # 1. Infer type of scrutinee and extract params/indices.
    scrut = elim.scrutinee
    ind = elim.inductive
    scrut_ty = whnf(infer_type(scrut, ctx))
    scrut_ty_head, scrut_ty_bindings = decompose_app(scrut_ty)
    if scrut_ty_head is not ind:
        raise TypeError(
            f"Eliminator scrutinee not of the right inductive type\n{scrut}\n{scrut_ty_head}"
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
            f"  motive          = {motive}\n"
            f"  motive_applied  = {motive_applied}\n"
            f"  motive_applied_ty = {motive_applied_ty}"
        )

    # 2.3 The scrutinee binder domain must match the scrutinee type
    scrut_dom = motive_applied_ty.arg_ty
    if not type_equal(scrut_dom, scrut_ty):
        raise TypeError(
            "InductiveElim motive scrutinee domain mismatch:\n"
            f"  expected scrut_ty = {scrut_ty}\n"
            f"  found    scrut_dom    = {scrut_dom}"
        )

    # 2.4 The motive codomain must be a universe
    body_ty = whnf(motive_applied_ty.return_ty)
    if not isinstance(body_ty, Univ):
        raise TypeError(
            "InductiveElim motive codomain must be a universe:\n"
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
        inst_arg_types = instantiate_ctor_arg_types(ctor.arg_types, params_actual)
        m = len(inst_arg_types)

        # 3.2 Work in context Γ,fields: shift Γ-level params and motive by m.
        params_in_fields_ctx = tuple(shift(p, m) for p in params_actual)
        motive_in_fields_ctx = shift(motive, m)

        # Build a scrutinee-shaped term: C params field_vars.
        # field_vars are Var(m-1) .. Var(0) in the Γ,fields context.
        field_vars = tuple(Var(j) for j in reversed(range(m)))
        scrut_like = apply_term(ctor, *params_in_fields_ctx, *field_vars)

        # 3.3 Instantiate ctor result indices under fields (params only).
        result_indices_inst = instantiate_ctor_result_indices_under_fields(
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

    u = _expect_universe(motive_applied_ty.return_ty, ctx)  # cod should be Univ(u)

    # target type is P i⃗_actual scrut
    target_ty = App(motive_applied, scrut)

    # sanity check target_ty really is a type in Type u (or ≤ u with cumulativity)
    _ = _expect_universe(infer_type(target_ty, ctx), ctx)

    if u < ind.level:
        raise TypeError("Eliminator motive returns too small a universe")

    return target_ty


def type_equal(t1: Term, t2: Term, ctx: Ctx | None = None) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` are convertible via head reduction."""

    ctx = ctx or Ctx()
    t1_whnf = whnf(t1)
    t2_whnf = whnf(t2)

    if t1_whnf == t2_whnf:
        return True

    match t1_whnf, t2_whnf:
        case (Pi(arg1, body1), Pi(arg2, body2)):
            return type_equal(arg1, arg2, ctx) and type_equal(
                body1, body2, ctx.prepend_each(arg1)
            )
        case (Lam(arg_ty1, body1), Lam(arg_ty2, body2)):
            return type_equal(arg_ty1, arg_ty2, ctx) and type_equal(
                body1, body2, ctx.prepend_each(arg_ty1)
            )
        case (App(f1, a1), App(f2, a2)):
            return type_equal(f1, f2, ctx) and type_equal(a1, a2, ctx)
        case (
            Elim(ind1, motive1, cases1, scrutinee1),
            Elim(ind2, motive2, cases2, scrutinee2),
        ) if (
            ind1 is ind2
        ):
            if len(cases1) != len(cases2):
                return False
            return (
                type_equal(motive1, motive2, ctx)
                and all(
                    type_equal(case1, case2, ctx)
                    for case1, case2 in zip(cases1, cases2, strict=True)
                )
                and type_equal(scrutinee1, scrutinee2, ctx)
            )
        case (Ctor() as ctor1, Ctor() as ctor2):
            return ctor1 is ctor2
        case (I() as ind1, I() as ind2):
            return ind1 is ind2

    return False


def _expect_universe(term: Term, ctx: Ctx) -> int:
    """Return the universe level of ``term`` or raise if it is not a type.

    Infers ``term`` and reduces it to weak head normal form so universe
    annotations reflect canonical shapes, then enforces that the result is a
    ``Univ``.
    """
    ty = infer_type(term, ctx)
    ty = whnf(ty)
    if not isinstance(ty, Univ):
        raise TypeError(f"Expected a universe, got {ty!r}")
    return ty.level


def infer_type(term: Term, ctx: Ctx | None = None) -> Term:
    """Infer the type of ``term`` under the optional De Bruijn context ``ctx``.

    Follows the syntax-directed typing rules; raises on ill-formed terms
    instead of returning ``None`` so callers don't silently accept mistakes.
    """

    ctx = ctx or Ctx()
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i < len(ctx):
                return shift(ctx[i].ty, i + 1)
            else:
                raise TypeError(f"Unbound variable {i}")
        case Lam(arg_ty, body):
            # Lambdas infer to Pis: infer the body under an extended context.
            body_ty = infer_type(body, ctx.prepend_each(arg_ty))
            return Pi(arg_ty, body_ty)
        case App(f, a):
            # Application: infer the function, ensure it is a Pi, and that the
            # argument checks against its domain.
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError(
                    f"Application of non-function\narg: {a},\narg_ty: {infer_type(a, ctx)}\nf: {f}\nf_ty: {f_ty}\nctx: {ctx}"
                )
            try:
                type_check(a, f_ty.arg_ty, ctx)
            except TypeError as exc:
                raise TypeError(
                    "Application argument type mismatch:\n"
                    f"  term = {term}\n"
                    f"  argument = {a}\n"
                    f"  expected arg_ty = {f_ty.arg_ty}\n"
                    f"  inferred arg_ty = {infer_type(a, ctx)}\n"
                    f"  inferred f_ty = {f_ty}"
                ) from exc
            return subst(f_ty.return_ty, a)
        case Pi(arg_ty, body):
            # Pi formation: both sides must be types; universe level is max.
            arg_level = _expect_universe(arg_ty, ctx)
            body_level = _expect_universe(body, ctx.prepend_each(arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case I():
            # Inductive type: check parameter and index kinds, build its
            # telescope (params then indices) ending in the inductive's level.
            for b in term.all_binders:
                _expect_universe(b, ctx)
                ctx = ctx.prepend_each(b)
            return nested_pi(*term.all_binders, return_ty=Univ(term.level))
        case Ctor():
            return _ctor_type(term)
        case Elim():
            return _infer_inductive_elim(term, ctx)

    raise TypeError(f"Unexpected term in infer_type: {term!r}")


def type_check(term: Term, ty: Term, ctx: Ctx | None = None) -> None:
    """Raise ``TypeError`` if ``term`` is not well-typed with type ``ty``."""

    ctx = ctx or Ctx()
    expected_ty = whnf(ty)
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i >= len(ctx):
                raise TypeError(f"Unbound variable {i}")
            found_ty = shift(ctx[i].ty, i + 1)
            if not type_equal(found_ty, expected_ty, ctx):
                raise TypeError(
                    "Variable type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  found = {found_ty}"
                )
            return None
        case Lam(arg_ty, body):
            # Lambdas must check against a Pi; ensure domains align, then check
            # the body under the extended context.
            match expected_ty:
                case Pi(dom, cod):
                    # if arg_ty != NatType():  # DELETE ME!!!
                    #     raise ValueError(f"a={arg_ty}\nb={dom}\nctx={ctx}")

                    if not type_equal(arg_ty, dom, ctx):
                        raise TypeError(
                            "Lambda domain mismatch:\n"
                            f"  term = {term}\n"
                            f"  expected domain = {dom}\n"
                            f"  found domain = {arg_ty}"
                        )
                    ctx1 = ctx.prepend_each(arg_ty)
                    try:
                        type_check(body, cod, ctx1)
                    except TypeError as exc:
                        raise TypeError(
                            "Lambda body has wrong type:\n"
                            f"  term = {term}\n"
                            f"  expected codomain = {cod}\n"
                            f"  inferred body = {infer_type(body, ctx1)}"
                        ) from exc
                    return None
                case _:
                    raise TypeError(
                        "Lambda expected to have Pi type:\n"
                        f"  term = {term}\n"
                        f"  expected = {expected_ty}"
                    )
        case App(f, a):
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError(
                    "Application of non-function:\n"
                    f"  term = {term}\n"
                    f"  function = {f}\n"
                    f"  inferred f_ty = {f_ty}"
                )
            try:
                type_check(a, f_ty.arg_ty, ctx)
            except TypeError as exc:
                raise TypeError(
                    "Application argument type mismatch:\n"
                    f"  term = {term}\n"
                    f"  argument = {a}\n"
                    f"  expected arg_ty = {f_ty.arg_ty}\n"
                    f"  inferred arg_ty = {infer_type(a, ctx)}\n"
                    f"  inferred f_ty = {f_ty}"
                ) from exc
            inferred_ty = subst(f_ty.return_ty, a)
            if not type_equal(expected_ty, inferred_ty):
                raise TypeError(
                    "Application result type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Pi(_, _):
            # Pi formation uses inference for its type; just compare expected.
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Pi type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case I():
            inferred_ty = infer_type(term, ctx)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Inductive type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Ctor():
            inferred_ty = _ctor_type(term)
            if not type_equal(expected_ty, inferred_ty, ctx):
                raise TypeError(
                    "Constructor type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred_ty}"
                )
            return None
        case Elim():
            inferred = _infer_inductive_elim(term, ctx)
            if not type_equal(expected_ty, inferred, ctx):
                raise TypeError(
                    "Eliminator type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}\n"
                    f"  inferred = {inferred}\n"
                    f"  normalized expected = {normalize(expected_ty)}\n"
                    f"  normalized inferred = {normalize(inferred)}"
                )
            return None
        case Univ(_):
            if not isinstance(expected_ty, Univ):
                raise TypeError(
                    "Universe type mismatch:\n"
                    f"  term = {term}\n"
                    f"  expected = {expected_ty}"
                )
            return None

    raise TypeError(f"Unexpected term in type_check: {term!r}")


__all__ = ["type_equal", "infer_type", "type_check"]
