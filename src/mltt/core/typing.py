"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    Id,
    IdElim,
    Ctor,
    Elim,
    I,
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
    match_inductive_application,
    decompose_app,
    instantiate_into,
    instantiate_params_indices,
    instantiate_forward,
)
from .reduce.normalize import normalize


def _ctor_type(ctor: Ctor) -> Term:
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
    assert ctor.result_indices == result_indices  # So why do we do this?
    result: Term = apply_term(ctor.inductive, (*param_vars, *result_indices))

    for arg_ty in reversed(ctor.arg_types):
        result = Pi(arg_ty, result)
    for index_ty in reversed(ctor.inductive.index_types):
        result = Pi(index_ty, result)
    for param_ty in reversed(ctor.inductive.param_types):
        result = Pi(param_ty, result)
    return result


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
    The case ultimately returns ``motive (ctor params indices args)``.
    """
    # Build the Pi type the case must inhabit for this constructor.
    # We interleave constructor arguments with any recursive occurrences
    # (marking those as needing an IH). The motive is applied to the
    # fully-applied constructor to produce the case result type.
    binder_roles: list[tuple[str, int, Term | None]] = []
    arg_positions: list[int] = []
    instantiated_arg_types = instantiate_into(
        (*param_args, *index_args), ctor.arg_types
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
    elim: Elim,
    expected_ty: Term,
    ctx: Ctx,
) -> bool:
    """Type-check an ``InductiveElim`` against ``expected_ty``.

    The structure closely follows the informal typing rule:
      • The scrutinee must be an application of the inductive with the right
        parameter/index arguments.
      • The motive must quantify over that instantiated inductive.
      • Each case must have the eliminator-specific case type for its ctor.
      • The resulting motive application must live in a universe no larger than
        the motive's codomain.
    """
    scrutinee = elim.scrutinee
    motive = elim.motive
    inductive = elim.inductive
    # 1. Infer type of scrutinee and extract params/indices.
    scrutinee_ty = normalize(infer_type(scrutinee, ctx))
    scrut_head, scrut_args = decompose_app(scrutinee_ty)
    if scrut_head is not inductive:
        raise TypeError(
            f"Eliminator scrutinee not of the right inductive type\n{scrutinee}\n{scrut_head}"
        )

    # # 2. Check the motive’s type
    motive_ty = infer_type(motive, ctx)
    if not isinstance(motive_ty, Pi):
        raise TypeError("InductiveElim motive not a function")

    # 2.1 Check param binders
    inst_inductive_param_tys = instantiate_forward(
        inductive.param_types + inductive.index_types, scrut_args
    )
    ty: Pi = motive_ty
    print()
    print(inductive)
    print(inductive.param_types)
    print(inductive.index_types)
    print(scrut_args)
    print(inst_inductive_param_tys)
    for k, param_ty in enumerate(inductive.param_types + inductive.index_types):
        print(k)
        print(param_ty)
        print(ty)
        print(ty.arg_ty)
        if not isinstance(ty, Pi):
            raise TypeError("Motive missing param binder")
        if not type_equal(ty.arg_ty, param_ty):
            raise TypeError(
                f"Motive param binder type mismatch\n{ty.arg_ty}\n{param_ty}"
            )
        ty = ty.return_ty  # move under Π

    # # 2.3 Check scrutinee binder
    # if not isinstance(ty, Pi):
    #     raise TypeError("Motive missing scrutinee binder")
    # scrut_dom = ty.ty
    # # expected scrut_dom is I applied to the bound params/indices:
    # expected_scrut_dom = apply_inductive_head(I, param_vars, index_vars)
    # if not convertible(scrut_dom, expected_scrut_dom):
    #     raise TypeError("Motive scrutinee domain mismatch")
    # ty = ty.body  # body after all binders
    #
    # # 2.4 Final body must be a universe
    # if not is_universe(ty):
    #     raise TypeError("Motive codomain not a universe")

    # 3. For each constructor, compute the expected branch type and check
    for ctor, case in zip(inductive.constructors, elim.cases, strict=True):
        # 3.1 instantiate arg types with the actual params/indices
        head, scrut_args = decompose_app(scrutinee_ty)
        arg_tys = instantiate_into(
            inductive.param_types + inductive.index_types, ctor.arg_types
        )

        # 3.2 create recursive references (like in Step 2 of iota, but on types)
        recursive_refs: list[Var] = []
        args_count = len(arg_tys)
        for j, arg_ty in enumerate(arg_tys):
            head, _ = decompose_app(arg_ty)
            if head is inductive:
                # make a reference from the IH back to the arg. It has to refer back past the earlier recursive_refs
                # (through `len(recursive_refs)` binders), then to arg j, (but the scrut_args binders are reversed, so it must
                # refer back `args_count-j-1` levels.
                recursive_refs.append(Var(len(recursive_refs) + args_count - j - 1))

        # 3.3 build the expected branch type telescope
        m = len(arg_tys)
        r = len(recursive_refs)
        hypotheticals = tuple(Var(r + m - 1 - j) for j in range(m))
        scrut_like = ctor
        for arg in (
            scrut_args + hypotheticals
        ):  # scrut_args from decompose_app(scrutinee_ty)
            scrut_like = App(func=scrut_like, arg=arg)

        # 3.4 Add binders, right-to-left
        body = motive
        for arg in scrut_args:  # scrut_args from decompose_app(scrutinee_ty)
            body = App(body, arg)
        body = App(body, scrut_like)
        for ref_var in reversed(recursive_refs):
            # IH for arg_j : motive arg_j
            ih_ty = motive
            for arg in scrut_args:
                ih_ty = App(ih_ty, arg)
            ih_ty = App(ih_ty, ref_var)
            body = Pi(arg_ty=ih_ty, return_ty=body)
        for arg_ty in reversed(arg_tys):
            body = Pi(arg_ty=arg_ty, return_ty=body)

        expected_branch_ty = body
        if not type_check(case, expected_branch_ty, ctx):
            raise TypeError(
                f"Case for constructor has wrong type\n{ctor}\n{case}\n{body}\n{ctx}"
            )

    body = motive
    for arg in scrut_args:
        body = App(body, arg)
    target_ty = App(body, scrutinee)
    target_level = _expect_universe(target_ty, ctx)
    body = motive_ty
    for arg in scrut_args:
        body = Pi(body, arg)
    motive_level = _expect_universe(body.return_ty, ctx.extend(scrutinee_ty))
    if target_level > motive_level:
        raise TypeError("InductiveElim motive returns too small a universe")
    return type_equal(expected_ty, target_ty)


def _type_check_inductive_elim1(
    inductive: I,
    motive: Term,
    cases: tuple[Term, ...],
    scrutinee: Term,
    expected_ty: Term,
    ctx: Ctx,
) -> bool:
    """Type-check an ``InductiveElim`` against ``expected_ty``.

    The structure closely follows the informal typing rule:
      • The scrutinee must be an application of the inductive with the right
        parameter/index arguments.
      • The motive must quantify over that instantiated inductive.
      • Each case must have the eliminator-specific case type for its ctor.
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
    if not type_equal(motive_ty.arg_ty, inductive_applied):
        raise TypeError("InductiveElim motive domain mismatch")
    motive_level = _expect_universe(motive_ty.return_ty, ctx.extend(inductive_applied))

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
            t = t.return_ty
        return count

    # We try typing casees with either the decomposed ctor args (if available)
    # or the scrutinee's instantiated param/index args.
    candidate_args = [
        (param_args_for_cases, index_args_for_cases),
    ]
    if (param_args, index_args) not in candidate_args:
        candidate_args.append((param_args, index_args))

    for ctor, case in zip(inductive.constructors, cases):
        success = False
        last_error: TypeError | None = None
        for cand_param_args, cand_index_args in candidate_args:
            # Derive the expected case type for this ctor under the chosen args.
            case_ty = _expected_case_type(
                inductive, cand_param_args, cand_index_args, motive, ctor
            )
            # Index arguments may be left implicit in casees; we opportunistically
            # feed them when the case is a lambda expecting exactly those types.

            index_arg_types = instantiate_into(cand_param_args, inductive.index_types)
            case_term = case
            lam_count = 0
            case_scan = case_term
            while isinstance(case_scan, Lam):
                lam_count += 1
                case_scan = case_scan.body
            extra_needed = max(0, lam_count - _pi_arity(case_ty))

            for idx_arg, idx_ty in zip(cand_index_args, index_arg_types):
                if extra_needed <= 0:
                    break
                if isinstance(case_term, Lam) and type_equal(case_term.arg_ty, idx_ty):
                    case_term = App(case_term, idx_arg)
                    extra_needed -= 1
                else:
                    break
            try:
                if type_check(case_term, case_ty, ctx):
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
            if not type_check(a, f_ty.arg_ty, ctx):
                raise TypeError(
                    f"Application argument type mismatch\narg: {a},\narg_ty: {infer_type(a)}\nf: {f}\nf_ty: {f_ty},\n{ctx}"
                )
            return subst(f_ty.return_ty, a)
        case Pi(arg_ty, body):
            # Pi formation: both sides must be types; universe level is max.
            arg_level = _expect_universe(arg_ty, ctx)
            body_level = _expect_universe(body, ctx.extend(arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case I(name, param_types, index_types, constructors, level):
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
        case Ctor():
            return _ctor_type(term)
        case Elim(inductive, motive, cases, scrutinee):
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
                        raise TypeError("Lambda domain mismatch")
                    return type_check(body, cod, ctx.extend(arg_ty))
                case _:
                    raise TypeError("Lambda expected to have Pi type")
        case App(f, a):
            f_ty = infer_type(f, ctx)
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
            return _type_check_inductive_elim(term, expected_ty, ctx)
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
            a1 = App(P, x)
            b = Refl(A, x)
            if not type_check(d, App(a1, b), ctx):
                raise TypeError("IdElim: d : P x (Refl x) fails")
            a2 = App(P, y)
            return type_equal(expected_ty, App(a2, p))
        case Univ(_):
            return isinstance(expected_ty, Univ)

    raise TypeError(f"Unexpected term in type_check: {term!r}")


__all__ = ["type_equal", "infer_type", "type_check"]
