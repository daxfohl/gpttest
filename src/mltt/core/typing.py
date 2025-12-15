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
    instantiate_ctor_result_indices,
)
from .reduce import normalize
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
        inst_arg_types = instantiate_ctor_arg_types(
            ctor.arg_types, params_actual, indices_actual
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
        arg_vars = tuple(Var(j) for j in reversed(range(m)))
        result_indices_inst = instantiate_ctor_result_indices(
            ctor.result_indices, params_actual, indices_actual, m
        )

        # 3.4 scrutinee-like value for this branch:
        #     C params_actual result_indices args
        scrut_like = apply_term(ctor, *params_actual, *indices_actual, *arg_vars)

        # # 3.5 branch codomain: motive result_indices scrut_like
        codomain_base = apply_term(motive, *result_indices_inst, scrut_like)
        codomain = shift(
            codomain_base, r, cutoff=0
        )  # because r IH binders are inserted inside

        # 3.6 Build IH types
        # ih_j : motive indices_j arg_j
        # in Γ, args (only)
        ih_base = [
            apply_term(motive, *indices_field, Var(m - 1 - j))
            for (j, indices_field) in recursive_positions
        ]
        # IH types in Γ, args, ihs (shift each by number of later IH binders)
        ih_types = [shift(ih_base[ri], r - 1 - ri, cutoff=0) for ri in range(r)]

        # Add binders left-to-right (outermost → innermost).
        telescope = (*inst_arg_types, *ih_types)
        case_ctx = ctx.prepend_each(*telescope)
        tel_len = len(telescope)
        case_args = tuple(Var(tel_len - 1 - k) for k in range(tel_len))  # a0...ih0...
        applied = apply_term(case, *case_args)  # (((case a0) a1) ...)
        if not type_check(whnf(applied), codomain, case_ctx):
            raise TypeError(
                f"Case for constructor has wrong type1\n{ctor}\n{normalize(case)}\n{normalize(applied)}\n{normalize(codomain)}\n{[normalize(e.ty) for e in case_ctx]}"
            )

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
                return ctx[i].ty
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
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.arg_ty, ctx):
                raise TypeError(
                    f"Application argument type mismatch\narg: {a},\narg_ty: {infer_type(a, ctx)}\nf: {f}\nf_ty: {f_ty}\nf_arg_ty: {f_ty.arg_ty}\nctx: {ctx}"
                )
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
                    return type_check(body, cod, ctx.prepend_each(arg_ty))
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
