"""Elaboration from surface core to kernel terms."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from mltt.common.span import Span
from mltt.elab.apply_matcher import ArgMatcher
from mltt.elab.ast import (
    EAnn,
    EApp,
    EArg,
    EBranch,
    ECtor,
    EHole,
    EInd,
    EInductiveDef,
    ELet,
    ELevel,
    ELam,
    EMatch,
    ENamedArg,
    EPartial,
    EPi,
    EUApp,
    EUniv,
    EVar,
    ETerm,
    EPatCtor,
    EPatVar,
    EPatWild,
)
from mltt.elab.errors import ElabError
from mltt.elab.types import (
    BinderSpec,
    ElabEnv,
    ElabType,
    apply_binder_specs,
    attach_binder_types,
    normalize_binder_name,
)
from mltt.kernel.ast import App, Const, Lam, Let, Pi, Term, UApp, Univ, Var
from mltt.kernel.env import Env, GlobalDecl
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LConst, LevelExpr, LVar
from mltt.kernel.tel import Spine, Telescope, decompose_uapp, mk_app, mk_lams, mk_pis, mk_uapp
from mltt.solver.solver import Solver


def elab_infer(term: ETerm, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    match term:
        case EVar():
            return _infer_name(term.name, env, solver, term.span, allow_uapp=True)
        case EInd():
            return _infer_name(term.name, env, solver, term.span, allow_uapp=True, ind=True)
        case ECtor():
            return _infer_name(
                term.name, env, solver, term.span, allow_uapp=True, ctor=True
            )
        case EUniv():
            level = _elab_level(term.level, solver, term.span)
            kernel = Univ(level)
            return kernel, ElabType(kernel.infer_type(env.kenv))
        case EAnn():
            ty_term = _elab_type(term.ty, env, solver)
            term_k = elab_check(term.term, ElabType(ty_term), env, solver)
            return term_k, ElabType(ty_term)
        case EHole():
            raise ElabError("Hole needs expected type", term.span)
        case ELam():
            return _elab_lam_infer(term, env, solver)
        case EPi():
            return _elab_pi(term, env, solver)
        case EApp():
            return _elab_app(term, env, solver, allow_partial=False)
        case EUApp():
            return _elab_uapp(term, env, solver)
        case EPartial():
            return _elab_partial(term, env, solver)
        case ELet():
            return _elab_let(term, env, solver)
        case EMatch():
            return _elab_match(term, env, solver, expected=None)
        case EInductiveDef():
            return _elab_inductive_def(term, env, solver)
        case _:
            raise ElabError("Unsupported term for elaboration", term.span)


def elab_check(
    term: ETerm, expected: ElabType, env: ElabEnv, solver: Solver
) -> Term:
    match term:
        case EHole():
            return solver.fresh_meta(env.kenv, expected.term, term.span, kind="hole")
        case ELam():
            return _elab_lam_check(term, expected, env, solver)
        case EMatch():
            term_k, _ty = _elab_match(term, env, solver, expected=expected.term)
            return term_k
        case _:
            term_k, inferred = elab_infer(term, env, solver)
            solver.add_constraint(env.kenv, inferred.term, expected.term, term.span)
            return term_k


def _infer_name(
    name: str,
    env: ElabEnv,
    solver: Solver,
    span: Span,
    *,
    allow_uapp: bool,
    ind: bool = False,
    ctor: bool = False,
) -> tuple[Term, ElabType]:
    local = env.lookup_local(name)
    if local is not None:
        term: Term = Var(local)
        ty = env.local_type(local)
        uarity = env.kenv.binders[local].uarity
        if allow_uapp and uarity:
            term, ty = _apply_uparams(term, ty, uarity, solver, span)
        return term, ty
    info = env.global_info(name)
    if info is None:
        raise ElabError(f"Unknown name {name}", span)
    decl, ty = info
    if ind and not isinstance(decl.value, Ind):
        raise ElabError(f"{name} is not an inductive", span)
    if ctor and not isinstance(decl.value, Ctor):
        raise ElabError(f"{name} is not a constructor", span)
    if isinstance(decl.value, (Ind, Ctor)):
        term = decl.value
    else:
        term = Const(name)
    if allow_uapp and decl.uarity:
        term, ty = _apply_uparams(term, ty, decl.uarity, solver, span)
    return term, ty


def _apply_uparams(
    term: Term, ty: ElabType, uarity: int, solver: Solver, span: Span
) -> tuple[Term, ElabType]:
    levels = tuple(solver.fresh_level_meta("implicit", span) for _ in range(uarity))
    return UApp(term, levels), ty.inst_levels(levels)


def _elab_level(level: ELevel | None, solver: Solver, span: Span) -> LevelExpr:
    if level is None:
        return solver.fresh_level_meta("type", span)
    if level.kind == "const":
        return LConst(level.value)
    return LVar(level.value)


def _elab_type(term: ETerm, env: ElabEnv, solver: Solver) -> Term:
    ty_term, ty_ty = elab_infer(term, env, solver)
    ty_ty_whnf = ty_ty.term.whnf(env.kenv)
    if not isinstance(ty_ty_whnf, Univ):
        raise ElabError("Expected a type", term.span)
    return ty_term


def _elab_lam_infer(term: ELam, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    binders = term.binders
    current_env = env
    lam_body_env = env
    arg_types: list[Term] = []
    binder_specs: list[BinderSpec] = []
    for binder in binders:
        arg_ty = _elab_type(binder.ty, current_env, solver)
        arg_types.append(arg_ty)
        spec = BinderSpec(
            name=normalize_binder_name(binder.name),
            implicit=binder.implicit,
            ty=arg_ty,
        )
        binder_specs.append(spec)
        lam_body_env = lam_body_env.push_binder(
            ElabType(arg_ty), name=spec.name
        )
        current_env = current_env.push_binder(ElabType(arg_ty), name=spec.name)
    body_term, body_ty = elab_infer(term.body, lam_body_env, solver)
    lam_term = mk_lams(*arg_types, body=body_term)
    lam_ty = mk_pis(*arg_types, return_ty=body_ty.term)
    return lam_term, ElabType(lam_ty, tuple(binder_specs))


def _elab_lam_check(
    term: ELam, expected: ElabType, env: ElabEnv, solver: Solver
) -> Term:
    current_env = env
    current_ty = expected.term
    lam_body_env = env
    arg_types: list[Term] = []
    for binder in term.binders:
        current_ty = current_ty.whnf(env.kenv)
        if not isinstance(current_ty, Pi):
            raise ElabError("Lambda expected to have Pi type", term.span)
        arg_ty = _elab_type(binder.ty, current_env, solver)
        solver.add_constraint(env.kenv, arg_ty, current_ty.arg_ty, term.span)
        arg_types.append(arg_ty)
        name = normalize_binder_name(binder.name)
        lam_body_env = lam_body_env.push_binder(ElabType(arg_ty), name=name)
        current_env = current_env.push_binder(ElabType(arg_ty), name=name)
        current_ty = current_ty.return_ty
    body_term = elab_check(term.body, ElabType(current_ty), lam_body_env, solver)
    return mk_lams(*arg_types, body=body_term)


def _elab_pi(term: EPi, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    current_env = env
    arg_types: list[Term] = []
    for binder in term.binders:
        arg_ty = _elab_type(binder.ty, current_env, solver)
        arg_types.append(arg_ty)
        name = normalize_binder_name(binder.name)
        current_env = current_env.push_binder(ElabType(arg_ty), name=name)
    body_ty = _elab_type(term.body, current_env, solver)
    pi_term = mk_pis(*arg_types, return_ty=body_ty)
    return pi_term, ElabType(pi_term.infer_type(env.kenv))


def _elab_uapp(term: EUApp, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    head_term, head_ty = _infer_name(
        term.head.name if isinstance(term.head, EVar) else "",
        env,
        solver,
        term.span,
        allow_uapp=False,
    ) if isinstance(term.head, EVar) else elab_infer(term.head, env, solver)
    levels = tuple(_elab_level(level, solver, term.span) for level in term.levels)
    return UApp(head_term, levels), head_ty.inst_levels(levels)


def _elab_app(
    term: EApp, env: ElabEnv, solver: Solver, *, allow_partial: bool
) -> tuple[Term, ElabType]:
    fn_term, fn_ty = elab_infer(term.fn, env, solver)
    if allow_partial:
        return _apply_partial(fn_term, fn_ty, term.args, term.named_args, env, solver, term.span)
    return _apply_args(fn_term, fn_ty, term.args, term.named_args, env, solver, term.span)


def _elab_partial(
    term: EPartial, env: ElabEnv, solver: Solver
) -> tuple[Term, ElabType]:
    if isinstance(term.term, EApp):
        return _elab_app(term.term, env, solver, allow_partial=True)
    return elab_infer(term.term, env, solver)


def _apply_args(
    fn_term: Term,
    fn_ty: ElabType,
    args: tuple[EArg, ...],
    named_args: tuple[ENamedArg, ...],
    env: ElabEnv,
    solver: Solver,
    span: Span,
) -> tuple[Term, ElabType]:
    binders = attach_binder_types(fn_ty.term, fn_ty.binders, env.kenv)
    matcher = ArgMatcher(binders, args, named_args, span)
    term = fn_term
    current_ty = fn_ty.term
    remaining = list(binders)
    named_map = {arg.name: arg for arg in named_args}
    while remaining:
        binder = remaining[0]
        decision = matcher.match_for_binder(binder, allow_partial=False)
        if decision.kind == "missing":
            _raise_missing_explicit(binder, remaining, named_map, span)
        if decision.kind == "stop":
            raise ElabError("Missing explicit argument", span)
        current_ty = current_ty.whnf(env.kenv)
        if not isinstance(current_ty, Pi):
            raise ElabError("Application of non-function", span)
        arg_ty = binder.ty or current_ty.arg_ty
        if decision.kind == "implicit":
            arg_term = solver.fresh_meta(env.kenv, arg_ty, span, kind="implicit")
        else:
            assert decision.arg is not None
            arg_term = _elab_arg(decision.arg, ElabType(arg_ty), env, solver)
        term = App(term, arg_term)
        current_ty = current_ty.return_ty.subst(arg_term)
        remaining = list(apply_binder_specs(tuple(remaining[1:]), arg_term))
    if matcher.has_positional() or matcher.unknown_named() is not None:
        raise ElabError("Application of non-function", span)
    return term, ElabType(current_ty, ())


def _apply_partial(
    fn_term: Term,
    fn_ty: ElabType,
    args: tuple[EArg, ...],
    named_args: tuple[ENamedArg, ...],
    env: ElabEnv,
    solver: Solver,
    span: Span,
) -> tuple[Term, ElabType]:
    binders = attach_binder_types(fn_ty.term, fn_ty.binders, env.kenv)
    matcher = ArgMatcher(binders, args, named_args, span)
    term = fn_term
    current_ty = fn_ty.term
    remaining = list(binders)
    lam_binders: list[BinderSpec] = []
    current_env = env
    while remaining:
        binder = remaining[0]
        decision = matcher.match_for_binder(binder, allow_partial=True)
        if decision.kind in ("missing", "stop"):
            for rem in remaining:
                current_ty_whnf = current_ty.whnf(current_env.kenv)
                if not isinstance(current_ty_whnf, Pi):
                    raise ElabError("Application of non-function", span)
                arg_ty = current_ty_whnf.arg_ty
                name = normalize_binder_name(rem.name)
                lam_binders.append(
                    BinderSpec(name=name, implicit=rem.implicit, ty=arg_ty)
                )
                current_env = current_env.push_binder(ElabType(arg_ty), name=name)
                term = term.shift(1)
                current_ty = current_ty.shift(1)
                term = App(term, Var(0))
                current_ty = current_ty_whnf.return_ty.shift(1).subst(Var(0))
            remaining = []
            break
        current_ty = current_ty.whnf(current_env.kenv)
        if not isinstance(current_ty, Pi):
            raise ElabError("Application of non-function", span)
        arg_ty = current_ty.arg_ty
        if decision.kind == "implicit":
            arg_term = solver.fresh_meta(current_env.kenv, arg_ty, span, kind="implicit")
        else:
            assert decision.arg is not None
            arg_term = _elab_arg(decision.arg, ElabType(arg_ty), current_env, solver)
        term = App(term, arg_term)
        current_ty = current_ty.return_ty.subst(arg_term)
        remaining = remaining[1:]
    if matcher.unknown_named() is not None:
        name = matcher.unknown_named()
        raise ElabError(f"Unknown named argument {name}", span)
    result_ty = current_ty
    for binder in reversed(lam_binders):
        assert binder.ty is not None
        term = Lam(binder.ty, term)
        result_ty = Pi(binder.ty, result_ty)
    return term, ElabType(result_ty, tuple(lam_binders))


def _elab_arg(
    arg: EArg | ENamedArg, expected: ElabType, env: ElabEnv, solver: Solver
) -> Term:
    term = arg.term
    return elab_check(term, expected, env, solver)


def _raise_missing_explicit(
    binder: BinderSpec,
    remaining: list[BinderSpec],
    named_map: dict[str, ENamedArg],
    span: Span,
) -> None:
    missing_name = binder.name or "_"
    dependent: list[str] = []
    for j, other in enumerate(remaining[1:], start=1):
        if other.name is None or other.name not in named_map:
            continue
        if other.ty is None:
            continue
        idx = j - 1
        if _depends_on_var(other.ty, idx, 0):
            dependent.append(other.name)
    if dependent:
        joined = ", ".join(dependent)
        raise ElabError(
            f"Missing explicit argument {missing_name}; later arguments depend on it: {joined}",
            span,
        )
    raise ElabError("Missing explicit argument", span)


def _depends_on_var(term: Term, index: int, depth: int) -> bool:
    match term:
        case Var(k):
            return k == index + depth
        case Lam(arg_ty=arg_ty, body=body):
            return _depends_on_var(arg_ty, index, depth) or _depends_on_var(
                body, index, depth + 1
            )
        case Pi(arg_ty=arg_ty, return_ty=body):
            return _depends_on_var(arg_ty, index, depth) or _depends_on_var(
                body, index, depth + 1
            )
        case Let(arg_ty=arg_ty, value=value, body=body):
            return (
                _depends_on_var(arg_ty, index, depth)
                or _depends_on_var(value, index, depth)
                or _depends_on_var(body, index, depth + 1)
            )
        case App(func=func, arg=arg):
            return _depends_on_var(func, index, depth) or _depends_on_var(
                arg, index, depth
            )
        case UApp(head=head, levels=_levels):
            return _depends_on_var(head, index, depth)
        case _:
            return False


def _elab_match(
    term: EMatch,
    env: ElabEnv,
    solver: Solver,
    expected: Term | None,
) -> tuple[Term, ElabType]:
    scrut_term, scrut_ty = elab_infer(term.scrutinee, env, solver)
    scrut_ty_whnf = scrut_ty.term.whnf(env.kenv)
    head, levels, args = decompose_uapp(scrut_ty_whnf)
    if isinstance(head, Const):
        decl = env.lookup_global(head.name)
        if decl is not None and isinstance(decl.value, Ind):
            head = decl.value
    if not isinstance(head, Ind):
        raise ElabError("Match scrutinee must be inductive", term.span)
    ind = head
    p = len(ind.param_types)
    params_actual = args[:p]
    indices_actual = args[p:]
    if term.motive is None:
        if expected is None:
            raise ElabError("Cannot infer match result type", term.span)
        index_types = ind.index_types.inst_levels(levels).instantiate(params_actual)
        expected_body = expected.shift(len(index_types) + 1)
        motive = mk_lams(*index_types, scrut_ty.term, body=expected_body)
    else:
        motive, _motive_ty = elab_infer(term.motive, env, solver)
    cases = _elab_match_cases(
        term.branches,
        ind,
        levels,
        params_actual,
        indices_actual,
        motive,
        env,
        solver,
        term.span,
    )
    elim = Elim(inductive=ind, motive=motive, cases=cases, scrutinee=scrut_term)
    return elim, ElabType(elim.infer_type(env.kenv))


def _elab_match_cases(
    branches: tuple[EBranch, ...],
    ind: Ind,
    levels: tuple[LevelExpr, ...],
    params_actual: Spine,
    indices_actual: Spine,
    motive: Term,
    env: ElabEnv,
    solver: Solver,
    span: Span,
) -> tuple[Term, ...]:
    ctor_map = {ctor.name: ctor for ctor in ind.constructors}
    ctor_branches: dict[str, EBranch] = {}
    default_branch: EBranch | None = None
    for branch in branches:
        match branch.pat:
            case EPatCtor(ctor=name):
                ctor_branches[name] = branch
            case EPatWild():
                default_branch = branch
            case EPatVar():
                default_branch = branch
            case _:
                raise ElabError("Unsupported match pattern", branch.span)
    cases: list[Term] = []
    for ctor in ind.constructors:
        branch = ctor_branches.get(ctor.name)
        if branch is None:
            branch = ctor_branches.get(f"{ind.name}.{ctor.name}")
        if branch is None:
            branch = default_branch
        if branch is None:
            raise ElabError("Missing match branch", span)
        case_term = _elab_match_branch(
            branch, ctor, ind, levels, params_actual, motive, env, solver
        )
        cases.append(case_term)
    return tuple(cases)


def _elab_match_branch(
    branch: EBranch,
    ctor: Ctor,
    ind: Ind,
    levels: tuple[LevelExpr, ...],
    params_actual: Spine,
    motive: Term,
    env: ElabEnv,
    solver: Solver,
) -> Term:
    field_types, ih_types, scrut_like, result_indices = _elim_info(
        ctor, ind, levels, params_actual, motive
    )
    field_count = len(field_types)
    ih_count = len(ih_types)
    pat_args: tuple = ()
    if isinstance(branch.pat, EPatCtor):
        pat_args = branch.pat.args
    elif isinstance(branch.pat, (EPatWild, EPatVar)):
        pat_args = ()
    if pat_args:
        if len(pat_args) not in (field_count, field_count + ih_count):
            raise ElabError("Constructor pattern arity mismatch", branch.span)
    field_pats = pat_args[:field_count]
    ih_pats = pat_args[field_count:]
    current_env = env
    for field_ty, pat in zip(field_types, field_pats, strict=False):
        name = None
        if isinstance(pat, EPatVar):
            name = normalize_binder_name(pat.name)
        current_env = current_env.push_binder(ElabType(field_ty), name=name)
    for ih_ty, pat in zip(ih_types, ih_pats, strict=False):
        name = None
        if isinstance(pat, EPatVar):
            name = normalize_binder_name(pat.name)
        current_env = current_env.push_binder(ElabType(ih_ty), name=name)
    codomain = mk_app(motive.shift(field_count), result_indices, scrut_like).shift(
        ih_count
    )
    rhs_term = elab_check(branch.rhs, ElabType(codomain), current_env, solver)
    case_term = rhs_term
    for ih_ty in reversed(ih_types):
        case_term = Lam(ih_ty, case_term)
    for field_ty in reversed(field_types):
        case_term = Lam(field_ty, case_term)
    return case_term


def _elim_info(
    ctor: Ctor,
    ind: Ind,
    levels: tuple[LevelExpr, ...],
    params_actual: Spine,
    motive: Term,
) -> tuple[Telescope, Telescope, Term, Spine]:
    p = len(ind.param_types)
    q = len(ind.index_types)
    field_types = ctor.field_schemas.inst_levels(levels).instantiate(params_actual)
    m = len(field_types)
    params_in_fields = params_actual.shift(m)
    motive_in_fields = motive.shift(m)
    field_vars = Spine.vars(m)
    scrut_like = mk_uapp(ctor, levels, params_in_fields, field_vars)
    result_indices = ctor.result_indices.inst_levels(levels).instantiate(params_actual, m)
    ihs: list[Term] = []
    for ri, j in enumerate(ctor.rps):
        rec_head, rec_levels, rec_args = decompose_uapp(
            field_types[j].shift(m - j)
        )
        assert rec_head == ind
        if levels and rec_levels and rec_levels != levels:
            raise ElabError("Recursive field uses mismatched universes", Span(0, 0))
        rec_indices = rec_args[p : p + q]
        ih_type = mk_app(motive_in_fields, rec_indices, field_vars[j])
        ihs.append(ih_type.shift(ri))
    return field_types, Telescope.of(*ihs), scrut_like, result_indices


def _elab_let(term: ELet, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    extended_env = env
    uarity = len(term.uparams)
    if term.ty is None:
        value_term, value_ty = elab_infer(term.val, extended_env, solver)
        arg_ty = value_ty.term
    else:
        arg_ty = _elab_type(term.ty, extended_env, solver)
        value_term = elab_check(term.val, ElabType(arg_ty), extended_env, solver)
        value_ty = ElabType(arg_ty)
    name = normalize_binder_name(term.name)
    body_env = env.push_let(value_ty, value_term, name=name, uarity=uarity)
    body_term, body_ty = elab_infer(term.body, body_env, solver)
    return Let(arg_ty=arg_ty, value=value_term, body=body_term), body_ty


def _elab_inductive_def(
    term: EInductiveDef, env: ElabEnv, solver: Solver
) -> tuple[Term, ElabType]:
    uarity = len(term.uparams)
    param_types: list[Term] = []
    param_specs: list[BinderSpec] = []
    param_env = env
    for binder in term.params:
        arg_ty = _elab_type(binder.ty, param_env, solver)
        name = normalize_binder_name(binder.name)
        param_specs.append(
            BinderSpec(name=name, implicit=binder.implicit, ty=arg_ty)
        )
        param_types.append(arg_ty)
        param_env = param_env.push_binder(ElabType(arg_ty), name=name)
    level_term = _elab_type(term.level, param_env, solver)
    index_types, ind_level = _split_indices(level_term, param_env.kenv)
    ind = Ind(
        name=term.name,
        param_types=Telescope.of(*param_types),
        index_types=Telescope.of(*index_types),
        level=ind_level,
        uarity=uarity,
    )
    ind_env = _extend_globals(
        env,
        {
            term.name: GlobalDecl(
                ty=ind.infer_type(env.kenv),
                value=ind,
                reducible=False,
                uarity=uarity,
            )
        },
        {
            term.name: ElabType(
                ind.infer_type(env.kenv),
                tuple(param_specs)
                + tuple(BinderSpec(name=None, implicit=False, ty=ty) for ty in index_types),
            )
        },
    )
    ctors: list[Ctor] = []
    ctor_entries: dict[str, GlobalDecl] = {}
    ctor_types: dict[str, ElabType] = {}
    for ctor_decl in term.ctors:
        ctor_env = ind_env
        field_types: list[Term] = []
        field_specs: list[BinderSpec] = []
        for field in ctor_decl.fields:
            field_ty = _elab_type(field.ty, ctor_env, solver)
            name = normalize_binder_name(field.name)
            field_specs.append(
                BinderSpec(name=name, implicit=field.implicit, ty=field_ty)
            )
            field_types.append(field_ty)
            ctor_env = ctor_env.push_binder(ElabType(field_ty), name=name)
        result_term, _ = elab_infer(ctor_decl.result, ctor_env, solver)
        head, levels, args = decompose_uapp(result_term)
        if isinstance(head, Const):
            decl = ind_env.lookup_global(head.name)
            if decl is not None and isinstance(decl.value, Ind):
                head = decl.value
        if head != ind:
            raise ElabError("Constructor result must target inductive", ctor_decl.span)
        if levels and len(levels) != uarity:
            raise ElabError("Constructor universe arity mismatch", ctor_decl.span)
        if len(args) < len(param_types):
            raise ElabError("Constructor result missing parameters", ctor_decl.span)
        params_actual = args[: len(param_types)]
        result_indices = args[len(param_types) :]
        ctor = Ctor(
            name=ctor_decl.name,
            inductive=ind,
            field_schemas=Telescope.of(*field_types),
            result_indices=result_indices,
            uarity=uarity,
        )
        ctors.append(ctor)
        full_name = f"{ind.name}.{ctor_decl.name}"
        ctor_entries[full_name] = GlobalDecl(
            ty=ctor.infer_type(env.kenv),
            value=ctor,
            reducible=False,
            uarity=uarity,
        )
        ctor_types[full_name] = ElabType(
            ctor.infer_type(env.kenv),
            tuple(param_specs) + tuple(field_specs),
        )
        if params_actual:
            _ = params_actual
        if result_indices:
            _ = result_indices
    object.__setattr__(ind, "constructors", tuple(ctors))
    extended_env = _extend_globals(ind_env, ctor_entries, ctor_types)
    return elab_infer(term.body, extended_env, solver)


def _split_indices(term: Term, env: Env) -> tuple[tuple[Term, ...], LevelExpr]:
    indices: list[Term] = []
    current = term
    while True:
        current_whnf = current.whnf(env)
        if isinstance(current_whnf, Pi):
            indices.append(current_whnf.arg_ty)
            env = env.push_binder(current_whnf.arg_ty)
            current = current_whnf.return_ty
            continue
        if isinstance(current_whnf, Univ):
            return tuple(indices), current_whnf.level
        raise ElabError("Inductive level must end in a universe", Span(0, 0))


def _extend_globals(
    env: ElabEnv,
    globals_entries: dict[str, GlobalDecl],
    elab_entries: dict[str, ElabType],
) -> ElabEnv:
    new_globals = dict(env.kenv.globals)
    new_globals.update(globals_entries)
    new_elab = dict(env.eglobals)
    new_elab.update(elab_entries)
    return ElabEnv(
        kenv=Env(binders=env.kenv.binders, globals=MappingProxyType(new_globals)),
        locals=env.locals,
        eglobals=new_elab,
    )
