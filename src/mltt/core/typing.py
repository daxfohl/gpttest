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
from .debruijn import Ctx, subst
from .inductive_utils import (
    apply_term,
    decompose_app,
    decompose_lam,
    instantiate_for_inductive,
    nested_pi,
    match_inductive_application,
    nested_lam,
)
from .reduce.whnf import whnf


def _ctor_type(ctor: Ctor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then indices, then
    constructor arguments, finishing with the inductive head applied to
    the instantiated result indices.
    """
    ind = ctor.inductive
    index_count = len(ind.index_types)
    if len(ctor.result_indices) != index_count:
        raise TypeError("Constructor result indices must match inductive index arity")
    # Parameters bind outermost, then indices, then constructor arguments.
    # The locals are introduced in the same order the inductive signature expects:
    #   [params][indices][args] from outermost to innermost.
    offset = index_count + len(ctor.arg_types)
    param_vars = [
        Var(i) for i in reversed(range(offset, offset + len(ind.param_types)))
    ]
    return nested_pi(
        *ind.param_types,
        *ind.index_types,
        *ctor.arg_types,
        return_ty=apply_term(ctor.inductive, *param_vars, *ctor.result_indices),
    )


def _expected_case_type(
    inductive: I,
    param_args: tuple[Term, ...],
    index_args: tuple[Term, ...],
    motive: Term,
    ctor: Ctor,
) -> Term:
    """Return the required case type for ``ctor`` under given params/indices.

    The case receives one binder per constructor argument, plus an induction
    hypothesis for each recursive argument that matches the current params.
    The case ultimately returns ``motive indices (ctor params indices args)``.
    """
    # Build the Pi type the case must inhabit for this constructor.
    # We interleave constructor arguments with any recursive occurrences
    # (marking those as needing an IH). The motive is applied to the
    # fully-applied constructor to produce the case result type.
    binder_roles: list[tuple[str, int, Term | None]] = []
    arg_positions: list[int] = []
    instantiated_arg_types = instantiate_for_inductive(
        inductive, param_args, index_args, ctor.arg_types
    )

    for idx, arg_ty in enumerate(instantiated_arg_types):
        arg_positions.append(len(binder_roles))
        binder_roles.append(("arg", idx, arg_ty))
        match match_inductive_application(arg_ty, inductive):
            case (ctor_params, _):
                if len(ctor_params) == len(param_args) and all(
                    type_equal(p, a) for p, a in zip(ctor_params, param_args)
                ):
                    binder_roles.append(("ih", idx, None))
            case _:
                pass

    total_binders = len(binder_roles)
    ctor_args = tuple(Var(total_binders - 1 - arg_pos) for arg_pos in arg_positions)
    target = apply_term(
        motive, *index_args, apply_term(ctor, *param_args, *index_args, *ctor_args)
    )

    binder_types: list[Term] = []
    for pos, (role, arg_idx, maybe_arg_ty) in enumerate(binder_roles):
        if role == "arg":
            assert maybe_arg_ty is not None
            binder_types.append(maybe_arg_ty)
        else:
            index = pos - 1 - arg_idx
            binder_types.append(apply_term(motive, *index_args, Var(index)))

    return nested_pi(*binder_types, return_ty=target)


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

        # 3.1 instantiate arg types with actual params/indices
        inst_arg_types = instantiate_for_inductive(
            ind, params_actual, indices_actual, ctor.arg_types
        )

        # 3.2 identify recursive ctor args and their indices
        recursive_positions = []
        for j, inst_ty in enumerate(inst_arg_types):
            head_j, args_j = decompose_app(inst_ty)
            if head_j is ind:
                # args_j = params_for_field ++ indices_for_field
                params_field = args_j[:p]
                indices_field = args_j[p : p + q]
                assert params_field == params_actual
                recursive_positions.append((j, indices_field))

        # 3.3 compute result indices for this ctor
        m = len(inst_arg_types)
        r = len(recursive_positions)
        arg_vars = [Var(r + m - j - 1) for j in range(m)]
        result_indices_inst = instantiate_for_inductive(
            ind,
            params_actual,
            indices_actual,
            ctor.result_indices,
            args=tuple(arg_vars),
        )

        # 3.4 scrutinee-like value for this branch:
        #     C params_actual result_indices args
        scrut_like = apply_term(ctor, *params_actual, *result_indices_inst, *arg_vars)

        # 3.5 branch codomain: motive result_indices scrut_like
        codomain = apply_term(motive, *result_indices_inst, scrut_like)

        # 3.6 Build IH types
        # ih_j : motive indices_j arg_j
        ih_types = [
            apply_term(motive, *indices_j, Var(ri + m - j - 1))
            for ri, (j, indices_j) in enumerate(recursive_positions)
        ]

        # 3.7 Add binders, right-to-left
        # The codomain has all the arg_vars, and this Pi construction allows them to
        # reference the arg types without needing an actual value for them.

        # # This is a dupe of the below test.
        ctx2 = ctx
        # Add binders (right-to-left as in de Bruijn)
        for ty in (*inst_arg_types, *ih_types):
            ctx2 = ctx2.extend(ty)
        num_args = len(inst_arg_types) + len(ih_types)
        args = tuple(
            Var(num_args - 1 - k) for k in range(num_args)
        )  # a1..an, ih1..ihm in order
        applied = apply_term(case, *args)  # (((case a1) a2) ...)
        if not type_check(whnf(applied), codomain, ctx2):
            raise TypeError("Case for constructor has wrong type!")

        body = nested_pi(*inst_arg_types, *ih_types, return_ty=codomain)
        case_head, case_bindings = decompose_lam(case)
        inst_case_bindings = instantiate_for_inductive(
            ind, params_actual, indices_actual, case_bindings
        )
        case = nested_lam(*inst_case_bindings, body=case_head)
        if not type_check(case, body, ctx):
            raise TypeError(
                f"Case for constructor has wrong type\n{ctor}\n{case}\n{body}\n{ctx}"
            )

    target_ty = App(motive_applied, scrut)
    target_level = _expect_universe(target_ty, ctx)
    body = nested_pi(
        *(infer_type(b, ctx) for b in indices_actual), return_ty=motive_applied_ty
    )
    motive_level_source = body.return_ty if isinstance(body, Pi) else body
    motive_level = _expect_universe(motive_level_source, ctx)
    if target_level > motive_level:
        raise TypeError("InductiveElim motive returns too small a universe")
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
                body1, body2, ctx.extend(arg1)
            )
        case (Lam(arg_ty1, body1), Lam(arg_ty2, body2)):
            return type_equal(arg_ty1, arg_ty2, ctx) and type_equal(
                body1, body2, ctx.extend(arg_ty1)
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
                return ctx[i].ty
            else:
                raise TypeError(f"Unbound variable {i}")
        case Lam(arg_ty, body):
            # Lambdas infer to Pis: infer the body under an extended context.
            body_ty = infer_type(body, ctx.extend(arg_ty))
            return Pi(arg_ty, body_ty)
        case App(f, a):
            # Application: infer the function, ensure it is a Pi, and that the
            # argument checks against its domain.
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.arg_ty, ctx):
                raise TypeError(
                    f"Application argument type mismatch\narg: {a},\narg_ty: {infer_type(a, ctx)}\nf: {f}\nf_ty: {f_ty}\nf_arg_ty: {f_ty.arg_ty}\nctx: {ctx}"
                )
            return subst(f_ty.return_ty, a)
        case Pi(arg_ty, body):
            # Pi formation: both sides must be types; universe level is max.
            arg_level = _expect_universe(arg_ty, ctx)
            body_level = _expect_universe(body, ctx.extend(arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case I():
            # Inductive type: check parameter and index kinds, build its
            # telescope (params then indices) ending in the inductive's level.
            for b in term.all_binders:
                _expect_universe(b, ctx)
                ctx = ctx.extend(b)
            return nested_pi(*term.all_binders, return_ty=Univ(term.level))
        case Ctor():
            return _ctor_type(term)
        case Elim():
            return _infer_inductive_elim(term, ctx)

    raise TypeError(f"Unexpected term in infer_type: {term!r}")


def type_check(term: Term, ty: Term, ctx: Ctx | None = None) -> bool:
    """Check that ``term`` has type ``ty`` under ``ctx``, raising on mismatches."""

    ctx = ctx or Ctx()
    expected_ty = whnf(ty)
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i >= len(ctx):
                raise TypeError(f"Unbound variable {i}")
            return type_equal(ctx[i].ty, expected_ty, ctx)
        case Lam(arg_ty, body):
            # Lambdas must check against a Pi; ensure domains align, then check
            # the body under the extended context.
            match expected_ty:
                case Pi(dom, cod):
                    # if arg_ty != NatType():  # DELETE ME!!!
                    #     raise ValueError(f"a={arg_ty}\nb={dom}\nctx={ctx}")

                    if not type_equal(arg_ty, dom, ctx):
                        raise TypeError(
                            f"Lambda domain mismatch\n"
                            f"arg_ty:{arg_ty}\n"
                            f"dom: {dom}\n"
                        )
                    return type_check(body, cod, ctx.extend(arg_ty))
                case _:
                    raise TypeError("Lambda expected to have Pi type")
        case App(f, a):
            f_ty = whnf(infer_type(f, ctx))
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.arg_ty, ctx):
                raise TypeError(
                    f"Application argument type mismatch\n{a},\n{f_ty},\n{ctx}"
                )
            return type_equal(expected_ty, subst(f_ty.return_ty, a))
        case Pi(_, _):
            # Pi formation uses inference for its type; just compare expected.
            return type_equal(expected_ty, infer_type(term, ctx))
        case I():
            return type_equal(expected_ty, infer_type(term, ctx))
        case Ctor():
            return type_equal(expected_ty, _ctor_type(term))
        case Elim():
            inferred = _infer_inductive_elim(term, ctx)
            return type_equal(expected_ty, inferred)
        case Univ(_):
            return isinstance(expected_ty, Univ)

    raise TypeError(f"Unexpected term in type_check: {term!r}")


__all__ = ["type_equal", "infer_type", "type_check"]
