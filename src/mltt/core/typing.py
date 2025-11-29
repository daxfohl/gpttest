"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    Id,
    IdElim,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
)
from .debruijn import Ctx, subst
from .inductive_utils import (
    apply_term,
    decompose_ctor_app,
    instantiate_params_indices,
    match_inductive_application,
)
from .reduce.normalize import normalize


def _ctor_type(ctor: InductiveConstructor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then indices, then
    constructor arguments, finishing with the inductive head applied to
    the instantiated result indices.
    """
    ind = ctor.inductive
    param_count = len(ind.param_types)
    index_count = len(ind.index_types)
    if len(ctor.result_indices) != index_count:
        raise TypeError("Constructor result indices must match inductive index arity")
    arg_count = len(ctor.arg_types)
    # Parameters bind outermost, then indices, then constructor arguments.
    # The locals are introduced in the same order the inductive signature expects:
    #   [params][indices][args] from outermost to innermost.
    param_vars = tuple(
        Var(arg_count + index_count + param_count - 1 - idx)
        for idx in range(param_count)
    )
    index_vars = tuple(
        Var(arg_count + index_count - 1 - idx) for idx in range(index_count)
    )
    # Result indices may mention params/indices; instantiate them in that order.
    result_indices = tuple(
        instantiate_params_indices(idx_term, param_vars, index_vars, offset=arg_count)
        for idx_term in ctor.result_indices
    )
    assert ctor.result_indices == result_indices
    result: Term = apply_term(ctor.inductive, (*param_vars, *result_indices))

    for arg_ty in reversed(ctor.arg_types):
        result = Pi(arg_ty, result)
    for index_ty in reversed(ctor.inductive.index_types):
        result = Pi(index_ty, result)
    for param_ty in reversed(ctor.inductive.param_types):
        result = Pi(param_ty, result)
    return result


def _expected_case_type(
    inductive: InductiveType,
    param_args: tuple[Term, ...],
    index_args: tuple[Term, ...],
    motive: Term,
    ctor: InductiveConstructor,
) -> Term:
    """Return the required branch type for ``ctor`` under given params/indices.

    The branch receives one binder per constructor argument, plus an induction
    hypothesis for each recursive argument that matches the current params.
    The branch ultimately returns ``motive (ctor params indices args)``.
    """
    # Build the Pi type the branch must inhabit for this constructor.
    # We interleave constructor arguments with any recursive occurrences
    # (marking those as needing an IH). The motive is applied to the
    # fully-applied constructor to produce the branch result type.
    binder_roles: list[tuple[str, int, Term | None]] = []
    arg_positions: list[int] = []
    instantiated_arg_types = [
        instantiate_params_indices(arg_ty, param_args, index_args, offset=idx)
        for idx, arg_ty in enumerate(ctor.arg_types)
    ]

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
    args = (*param_args, *index_args, *ctor_args)
    target: Term = App(motive, apply_term(ctor, args))

    binder_types: list[Term] = []
    for pos, (role, arg_idx, maybe_arg_ty) in enumerate(binder_roles):
        if role == "arg":
            assert maybe_arg_ty is not None
            binder_types.append(maybe_arg_ty)
        else:
            index = pos - 1 - arg_idx
            binder_types.append(App(motive, Var(index)))

    result = target
    for binder_ty in reversed(binder_types):
        result = Pi(binder_ty, result)
    return result


def _type_check_inductive_elim(
    inductive: InductiveType,
    motive: Term,
    cases: list[Term],
    scrutinee: Term,
    expected_ty: Term,
    ctx: Ctx,
) -> bool:
    """Type-check an ``InductiveElim`` against ``expected_ty``.

    The structure closely follows the informal typing rule:
      • The scrutinee must be an application of the inductive with the right
        parameter/index arguments.
      • The motive must quantify over that instantiated inductive.
      • Each branch must have the eliminator-specific case type for its ctor.
      • The resulting motive application must live in a universe no larger than
        the motive's codomain.
    """

    scrutinee_ty = normalize(infer_type(scrutinee, ctx))
    # Recover param/index arguments from the scrutinee type.
    application = match_inductive_application(scrutinee_ty, inductive)
    if application is None:
        raise TypeError("InductiveElim scrutinee has wrong type")
    param_args, index_args = application
    inductive_applied = apply_term(inductive, (*param_args, *index_args))
    if not type_equal(scrutinee_ty, inductive_applied):
        raise TypeError("InductiveElim scrutinee has wrong type")

    motive_ty = infer_type(motive, ctx)
    if not isinstance(motive_ty, Pi):
        raise TypeError("InductiveElim motive not a function")
    if not type_equal(motive_ty.ty, inductive_applied):
        raise TypeError("InductiveElim motive domain mismatch")
    motive_level = _expect_universe(motive_ty.body, ctx.extend(inductive_applied))

    if len(cases) != len(inductive.constructors):
        raise TypeError("InductiveElim cases do not match constructors")

    # Prefer ctor arguments from the scrutinee if it is itself a constructor app,
    # which can be more specific than the fully instantiated scrutinee type.
    param_args_for_cases = param_args
    index_args_for_cases = index_args
    decomposition = decompose_ctor_app(scrutinee)
    if decomposition:
        ctor_head, ctor_args = decomposition
        param_count = len(inductive.param_types)
        index_count = len(inductive.index_types)
        expected_args = param_count + index_count + len(ctor_head.arg_types)
        if ctor_head.inductive is inductive and len(ctor_args) == expected_args:
            param_args_for_cases = ctor_args[:param_count]
            index_args_for_cases = ctor_args[param_count : param_count + index_count]

    def _pi_arity(term: Term) -> int:
        count = 0
        t = term
        while isinstance(t, Pi):
            count += 1
            t = t.body
        return count

    # We try typing branches with either the decomposed ctor args (if available)
    # or the scrutinee's instantiated param/index args.
    candidate_args = [
        (param_args_for_cases, index_args_for_cases),
    ]
    if (param_args, index_args) not in candidate_args:
        candidate_args.append((param_args, index_args))

    for ctor, branch in zip(inductive.constructors, cases):
        success = False
        last_error: TypeError | None = None
        for cand_param_args, cand_index_args in candidate_args:
            # Derive the expected branch type for this ctor under the chosen args.
            branch_ty = _expected_case_type(
                inductive, cand_param_args, cand_index_args, motive, ctor
            )
            # Index arguments may be left implicit in branches; we opportunistically
            # feed them when the branch is a lambda expecting exactly those types.
            index_arg_types = [
                instantiate_params_indices(index_ty, cand_param_args, (), offset=0)
                for index_ty in inductive.index_types
            ]
            branch_term = branch
            lam_count = 0
            branch_scan = branch_term
            while isinstance(branch_scan, Lam):
                lam_count += 1
                branch_scan = branch_scan.body
            extra_needed = max(0, lam_count - _pi_arity(branch_ty))

            for idx_arg, idx_ty in zip(cand_index_args, index_arg_types):
                if extra_needed <= 0:
                    break
                if isinstance(branch_term, Lam) and type_equal(branch_term.ty, idx_ty):
                    branch_term = App(branch_term, idx_arg)
                    extra_needed -= 1
                else:
                    break
            try:
                if type_check(branch_term, branch_ty, ctx):
                    success = True
                    break
            except TypeError as exc:
                last_error = exc
        if not success:
            if last_error is not None:
                raise last_error
            raise TypeError("Case for constructor has wrong type")

    target_ty = App(motive, scrutinee)
    target_level = _expect_universe(target_ty, ctx)
    if target_level > motive_level:
        raise TypeError("InductiveElim motive returns too small a universe")
    return type_equal(expected_ty, target_ty)


def type_equal(t1: Term, t2: Term) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` normalize to the same term."""

    return normalize(t1) == normalize(t2)


def _expect_universe(term: Term, ctx: Ctx) -> int:
    """Return the universe level of ``term`` or raise if it is not a type.

    Normalizes and infers ``term`` so universe annotations reflect canonical
    shapes, then enforces that the result is a ``Univ``.
    """

    ty = normalize(infer_type(term, ctx))
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
            f_ty = infer_type(f, ctx)
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Function argument type mismatch")
            return subst(f_ty.body, a)
        case Pi(arg_ty, body):
            # Pi formation: both sides must be types; universe level is max.
            arg_level = _expect_universe(arg_ty, ctx)
            body_level = _expect_universe(body, ctx.extend(arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case InductiveType(
            name=_,
            param_types=param_types,
            index_types=index_types,
            constructors=_,
            level=level,
        ):
            # Inductive type: check parameter and index kinds, build its
            # telescope (params then indices) ending in the inductive's level.
            ctx1 = ctx
            for param_ty in param_types:
                _expect_universe(param_ty, ctx1)
                ctx1 = ctx1.extend(param_ty)
            for index_ty in index_types:
                _expect_universe(index_ty, ctx1)
                ctx1 = ctx1.extend(index_ty)
            result: Term = Univ(level)
            for index_ty in reversed(index_types):
                result = Pi(index_ty, result)
            for param_ty in reversed(param_types):
                result = Pi(param_ty, result)
            return result
        case InductiveConstructor():
            return _ctor_type(term)
        case InductiveElim(_, motive, _, scrutinee):
            return App(motive, scrutinee)
        case Id(ty, lhs, rhs):
            # Identity type is a type when both endpoints check against ``ty``.
            if not type_check(lhs, ty, ctx) or not type_check(rhs, ty, ctx):
                raise TypeError("Id sides must have given type")
            return Univ(_expect_universe(ty, ctx))
        case Refl(ty, t):
            # Refl inhabits the corresponding identity type when ``t`` checks.
            if not type_check(t, ty, ctx):
                raise TypeError("Refl term not of stated type")
            return Id(ty, t, t)
        case IdElim(A, x, P, d, y, p):
            # Eliminator returns the motive applied to the target endpoints/proof.
            return App(App(P, y), p)

    raise TypeError(f"Unexpected term in infer_type: {term!r}")


def type_check(term: Term, ty: Term, ctx: Ctx | None = None) -> bool:
    """Check that ``term`` has type ``ty`` under ``ctx``, raising on mismatches."""

    ctx = ctx or Ctx()
    expected_ty = normalize(ty)
    match term:
        case Var(i):
            # A variable is well-typed only if a binder exists at that index.
            if i >= len(ctx):
                raise TypeError(f"Unbound variable {i}")
            return type_equal(ctx[i].ty, expected_ty)
        case Lam(arg_ty, body):
            # Lambdas must check against a Pi; ensure domains align, then check
            # the body under the extended context.
            match expected_ty:
                case Pi(dom, cod):
                    if not type_equal(arg_ty, dom):
                        print(arg_ty)
                        print(dom)
                        raise TypeError("Lambda domain mismatch")
                    return type_check(body, cod, ctx.extend(arg_ty))
                case _:
                    raise TypeError("Lambda expected to have Pi type")
        case App(f, a):
            f_ty = infer_type(f, ctx)
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Application argument type mismatch")
            return type_equal(expected_ty, subst(f_ty.body, a))
        case Pi(_, _):
            # Pi formation uses inference for its type; just compare expected.
            return type_equal(expected_ty, infer_type(term, ctx))
        case InductiveType():
            return type_equal(expected_ty, infer_type(term, ctx))
        case InductiveConstructor():
            return type_equal(expected_ty, _ctor_type(term))
        case InductiveElim(inductive, motive, cases, scrutinee):
            return _type_check_inductive_elim(
                inductive, motive, cases, scrutinee, expected_ty, ctx
            )
        case Id(id_ty, l, r):
            # Identity type formation: both sides must check against the given
            # ambient type; the result is a type.
            if not type_check(l, id_ty, ctx) or not type_check(r, id_ty, ctx):
                raise TypeError("Id sides not of given type")
            return isinstance(expected_ty, Univ)
        case Refl(rty, t):
            # Refl inhabits ``Id ty t t`` provided ``t`` checks against ``ty``.
            if not type_check(t, rty, ctx):
                raise TypeError("Refl term not of stated type")
            return type_equal(expected_ty, Id(rty, t, t))
        case IdElim(A, x, P, d, y, p):
            # Identity elimination (J): verify endpoints/proof, base case ``d``,
            # then ensure the expected type matches the motive application.
            if not type_check(x, A, ctx):
                raise TypeError("IdElim: x : A fails")
            if not type_check(y, A, ctx):
                raise TypeError("IdElim: y : A fails")
            if not type_check(p, Id(A, x, y), ctx):
                raise TypeError("IdElim: p : Id(A,x,y) fails")
            if not type_check(d, App(App(P, x), Refl(A, x)), ctx):
                raise TypeError("IdElim: d : P x (Refl x) fails")
            return type_equal(expected_ty, App(App(P, y), p))
        case Univ(_):
            return isinstance(expected_ty, Univ)

    raise TypeError(f"Unexpected term in type_check: {term!r}")


__all__ = ["type_equal", "infer_type", "type_check"]
