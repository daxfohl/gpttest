"""Surface pattern matching and let-pattern sugar."""

from __future__ import annotations

from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabEnv, ElabType
from mltt.elab.sast import _expect_universe, elab_check, elab_infer
from mltt.kernel.ast import Lam, Term, UApp, Var
from mltt.kernel.env import Const, Env
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LevelExpr
from mltt.kernel.tel import ArgList, Telescope, decompose_uapp, mk_app, mk_lams, mk_uapp
from mltt.surface.sast import (
    Pat,
    PatCtor,
    PatTuple,
    PatVar,
    PatWild,
    SBranch,
    SLetPat,
    SMatch,
    Span,
    SurfaceError,
    SurfaceTerm,
)


def _resolve_inductive_head(env: Env, head: Term) -> Ind | None:
    if isinstance(head, Ind):
        return head
    if isinstance(head, UApp) and isinstance(head.head, Ind):
        return head.head
    if isinstance(head, Const):
        decl = env.lookup_global(head.name)
        if decl is not None:
            if isinstance(decl.value, Ind):
                return decl.value
            if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ind):
                return decl.value.head
    return None


def _looks_like_ctor(name: str) -> bool:
    return bool(name) and (name[0].isupper() or "." in name)


def _elab_scrutinee_info(
    scrutinee: SurfaceTerm, env: ElabEnv, state: ElabState
) -> tuple[Term, Term, Term, tuple[LevelExpr, ...], ArgList]:
    scrut_term, scrut_ty = elab_infer(scrutinee, env, state)
    scrut_ty_whnf = scrut_ty.term.whnf(env.kenv)
    head, level_actuals, args = decompose_uapp(scrut_ty_whnf)
    return scrut_term, scrut_ty_whnf, head, level_actuals, args


def _abstract_var(term: Term, target: int) -> Term:
    shifted = term.shift(1)

    def replace_var(t: Term, depth: int) -> Term:
        if isinstance(t, Var) and t.k == target + depth + 1:
            return Var(depth)
        return t._replace_terms(
            lambda sub, meta: replace_var(sub, depth + meta.binder_count)
        )

    return replace_var(shifted, 0)


def _apply_as_name(
    as_name: str | None, scrut: Term, scrut_ty: Term
) -> tuple[Term, Term]:
    if as_name is None:
        return scrut, scrut_ty
    return Var(0), scrut_ty.shift(1)


def _mk_ctor_indices(
    ind: Ind,
    ctor: Ctor,
    level_actuals: tuple[LevelExpr, ...],
    params_actual: ArgList,
    field_vars: ArgList,
) -> ArgList:
    return ArgList.of(
        *[
            t.inst_levels(level_actuals).instantiate(params_actual, len(field_vars))
            for t in ctor.result_indices
        ]
    )


def _match_branch_types(
    ind: Ind,
    ctor: Ctor,
    motive: Term,
    level_actuals: tuple[LevelExpr, ...],
    params_actual: ArgList,
    span: Span,
) -> tuple[Telescope, Term]:
    ctor_field_types = ctor.field_schemas.inst_levels(level_actuals).instantiate(
        params_actual
    )
    m = len(ctor_field_types)
    params_in_fields_ctx = params_actual.shift(m)
    motive_in_fields_ctx = motive.shift(m)
    field_vars = ArgList.vars(m)
    scrut_like = mk_uapp(ctor, level_actuals, params_in_fields_ctx, field_vars)
    result_indices = ArgList.of(
        *[
            t.inst_levels(level_actuals).instantiate(params_actual, m)
            for t in ctor.result_indices
        ]
    )
    p = len(ind.param_types)
    q = len(ind.index_types)
    ihs: list[Term] = []
    for ri, j in enumerate(ctor.rps):
        rec_head, rec_levels, rec_field_args = decompose_uapp(
            ctor_field_types[j].shift(m - j)
        )
        if rec_head != ind:
            raise SurfaceError("Recursive field head mismatch", span)
        if level_actuals and rec_levels and rec_levels != level_actuals:
            raise SurfaceError("Recursive field universe mismatch", span)
        rec_params = rec_field_args[:p]
        rec_indices = rec_field_args[p : p + q]
        _ = rec_params
        ih_type = mk_app(motive_in_fields_ctx, rec_indices, field_vars[j])
        ihs.append(ih_type.shift(ri))
    ih_types = Telescope.of(*ihs)
    codomain = mk_app(motive_in_fields_ctx, result_indices, scrut_like).shift(
        len(ih_types)
    )
    tel = ctor_field_types + ih_types
    return tel, codomain


def _branch_binders(
    pat: Pat,
    ctor: Ctor,
    tel: Telescope,
    field_count: int,
) -> list[tuple[str | None, Term]]:
    binders: list[tuple[str | None, Term]] = []
    if isinstance(pat, PatCtor):
        if len(pat.args) not in {field_count, len(tel)}:
            raise SurfaceError("Match pattern has wrong arity", pat.span)
        if len(pat.args) == field_count:
            full_args = pat.args + tuple(
                PatWild(pat.span) for _ in range(len(tel) - field_count)
            )
        else:
            full_args = pat.args
        for arg, ty in zip(full_args, tel, strict=True):
            if isinstance(arg, PatVar):
                binders.append((arg.name, ty))
            elif isinstance(arg, PatWild):
                binders.append((None, ty))
            else:
                raise SurfaceError("Match pattern must be constructor or _", arg.span)
        return binders
    if isinstance(pat, PatVar):
        if _looks_like_ctor(pat.name):
            if len(tel) == 0:
                return []
            raise SurfaceError("Constructor pattern needs fields", pat.span)
        return [(pat.name, ty) for ty in tel]
    if isinstance(pat, PatWild):
        return [(None, ty) for ty in tel]
    raise SurfaceError("Unsupported match pattern", pat.span)


def _branch_map(
    branches: tuple[SBranch, ...], env: ElabEnv, ind: Ind
) -> tuple[dict[str, SBranch], SurfaceTerm | None]:
    branch_map: dict[str, SBranch] = {}
    default_branch: SurfaceTerm | None = None
    for branch in branches:
        pat = branch.pat
        if isinstance(pat, PatWild):
            default_branch = branch.rhs
            continue
        if isinstance(pat, PatVar):
            ctor_name = pat.name
            if not _looks_like_ctor(ctor_name):
                raise SurfaceError(
                    "Match pattern must be constructor or _", branch.span
                )
            branch_map[ctor_name] = branch
            continue
        if isinstance(pat, PatCtor):
            branch_map[pat.ctor] = branch
            continue
        raise SurfaceError("Unsupported match pattern", branch.span)
    for ctor in ind.constructors:
        qualified = f"{ind.name}.{ctor.name}"
        if ctor.name not in branch_map and qualified in branch_map:
            branch_map[ctor.name] = branch_map[qualified]
    _ = env
    return branch_map, default_branch


def elab_match_infer(
    match: SMatch, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if len(match.scrutinees) != 1:
        raise SurfaceError("Match must be desugared to one scrutinee", match.span)
    if match.motive is None and all(n is None for n in match.as_names):
        raise SurfaceError("Cannot infer match result type; use check-mode", match.span)
    return _elab_match_with_motive(match, env, state)


def elab_match_check(
    match: SMatch, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    if len(match.scrutinees) != 1:
        raise SurfaceError("Match must be desugared to one scrutinee", match.span)
    if match.motive is not None or any(n is not None for n in match.as_names):
        term, term_ty = _elab_match_with_motive(match, env, state)
        state.add_constraint(env.kenv, term_ty.term, expected.term, match.span)
        return term
    return _elab_match_core(match, env, state, expected)


def _elab_match_core(
    match: SMatch, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
        match.scrutinees[0], env, state
    )
    ind = _resolve_inductive_head(env.kenv, head)
    if ind is None:
        raise SurfaceError("Match scrutinee is not an inductive type", match.span)
    p = len(ind.param_types)
    q = len(ind.index_types)
    if len(args) != p + q:
        raise SurfaceError("Match scrutinee has wrong arity", match.span)
    params_actual = args[:p]
    indices_actual = args[p:]
    branch_map, default_branch = _branch_map(match.branches, env, ind)
    cases: list[Term] = []
    scrut_ty_in_ctx = mk_uapp(
        ind, level_actuals, params_actual.shift(q), ArgList.vars(q)
    )
    motive = mk_lams(
        *ind.index_types,
        body=Lam(scrut_ty_in_ctx, expected.term.shift(q + 1)),
    )
    if q > 0 and isinstance(scrut_term, Var):
        index_vars = [idx for idx in indices_actual if isinstance(idx, Var)]
        if len(index_vars) != len(indices_actual):
            raise SurfaceError(
                "Cannot infer motive with non-variable indices", match.span
            )
        index_vars = [
            Var(idx.k - 1 if idx.k > scrut_term.k else idx.k) for idx in index_vars
        ]
        indices_actual = ArgList.of(*index_vars)
    for ctor in ind.constructors:
        branch = branch_map.get(ctor.name)
        tel, codomain = _match_branch_types(
            ind, ctor, motive, level_actuals, params_actual, match.span
        )
        field_count = len(ctor.field_schemas)
        binder_names: list[tuple[str | None, Term]]
        if branch is None and default_branch is not None:
            binder_names = [(None, ty) for ty in tel]
            branch_rhs = default_branch
        else:
            if branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", match.span
                )
            binder_names = _branch_binders(branch.pat, ctor, tel, field_count)
            branch_rhs = branch.rhs
        env_fields = env
        for name, ty in binder_names:
            env_fields = env_fields.push_binder(ElabType(ty), name=name)
        rhs_term = elab_check(branch_rhs, env_fields, state, ElabType(codomain))
        cases.append(mk_lams(*tel, body=rhs_term))
    match_term = Elim(ind, motive, tuple(cases), scrut_term)
    return match_term


def _elab_match_with_motive(
    match: SMatch, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if match.motive is None:
        raise SurfaceError("Match motive missing", match.span)
    if len(match.scrutinees) != 1:
        raise SurfaceError("Dependent match needs one scrutinee", match.span)
    scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
        match.scrutinees[0], env, state
    )
    ind = _resolve_inductive_head(env.kenv, head)
    if ind is None:
        raise SurfaceError("Match scrutinee is not an inductive type", match.span)
    p = len(ind.param_types)
    q = len(ind.index_types)
    if len(args) != p + q:
        raise SurfaceError("Match scrutinee has wrong arity", match.span)
    params_actual = args[:p]
    indices_actual = args[p:]
    as_name = match.as_names[0] if match.as_names else None
    env_motive = env
    if as_name is not None:
        env_motive = env_motive.push_binder(ElabType(scrut_ty_whnf), name=as_name)
    motive_term, motive_ty = elab_infer(match.motive, env_motive, state)
    _expect_universe(motive_ty.term, env_motive.kenv, match.motive.span)
    scrut_ty_in_ctx = mk_uapp(
        ind, level_actuals, params_actual.shift(q), ArgList.vars(q)
    )
    if as_name is not None:
        motive_body = motive_term.shift(q, cutoff=1)
    else:
        motive_body = motive_term.shift(q)
    motive_fn = mk_lams(*ind.index_types, body=Lam(scrut_ty_in_ctx, motive_body))
    branch_map, default_branch = _branch_map(match.branches, env, ind)
    cases: list[Term] = []
    for ctor in ind.constructors:
        branch = branch_map.get(ctor.name)
        tel, codomain = _match_branch_types(
            ind, ctor, motive_fn, level_actuals, params_actual, match.span
        )
        field_count = len(ctor.field_schemas)
        binder_names: list[tuple[str | None, Term]]
        if branch is None and default_branch is not None:
            binder_names = [(None, ty) for ty in tel]
            branch_rhs = default_branch
        else:
            if branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", match.span
                )
            binder_names = _branch_binders(branch.pat, ctor, tel, field_count)
            branch_rhs = branch.rhs
        env_fields = env
        for name, ty in binder_names:
            env_fields = env_fields.push_binder(ElabType(ty), name=name)
        rhs_term = elab_check(branch_rhs, env_fields, state, ElabType(codomain))
        cases.append(mk_lams(*tel, body=rhs_term))
    match_term = Elim(ind, motive_fn, tuple(cases), scrut_term)
    match_ty = mk_app(motive_fn, indices_actual, scrut_term)
    return match_term, ElabType(match_ty)


def elab_let_pat_infer(
    term: SLetPat, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    value_term, value_ty = elab_infer(term.value, env, state)
    value_ty_whnf = value_ty.term.whnf(env.kenv)
    if not _is_irrefutable(env.kenv, value_ty_whnf, term.pat):
        raise SurfaceError("Refutable pattern in let; use match", term.span)
    env_body = env
    for name, ty in _collect_binders(env.kenv, value_ty_whnf, term.pat):
        env_body = env_body.push_binder(ElabType(ty), name=name)
    body_term, body_ty = elab_infer(term.body, env_body, state)
    match_term = SMatch(
        span=term.span,
        scrutinees=(term.value,),
        as_names=(None,),
        motive=None,
        branches=(SBranch(term.pat, term.body, term.span),),
    )
    match_term_k = elab_check(match_term, env, state, body_ty)
    _ = value_term
    return match_term_k, body_ty


def _is_irrefutable(env: Env, scrut_ty: Term, pat: Pat) -> bool:
    if isinstance(pat, (PatVar, PatWild)):
        return True
    if isinstance(pat, PatTuple):
        raise SurfaceError("Tuple patterns must be desugared", pat.span)
    if not isinstance(pat, PatCtor):
        return False
    head, level_actuals, args = decompose_uapp(scrut_ty)
    ind = _resolve_inductive_head(env, head)
    if ind is None or len(ind.constructors) != 1:
        return False
    ctor = ind.constructors[0]
    if pat.ctor not in {ctor.name, f"{ind.name}.{ctor.name}"}:
        return False
    p = len(ind.param_types)
    params_actual = args[:p]
    field_tys = Telescope.of(
        *[
            t.inst_levels(level_actuals).instantiate(params_actual, depth_above=i)
            for i, t in enumerate(ctor.field_schemas)
        ]
    )
    if len(pat.args) != len(field_tys):
        return False
    for field_ty, subpat in zip(field_tys, pat.args, strict=True):
        if not _is_irrefutable(env, field_ty, subpat):
            return False
    return True


def _collect_binders(env: Env, scrut_ty: Term, pat: Pat) -> list[tuple[str, Term]]:
    if isinstance(pat, PatVar):
        return [(pat.name, scrut_ty)]
    if isinstance(pat, PatWild):
        return []
    if isinstance(pat, PatTuple):
        raise SurfaceError("Tuple patterns must be desugared", pat.span)
    if not isinstance(pat, PatCtor):
        return []
    head, level_actuals, args = decompose_uapp(scrut_ty)
    ind = _resolve_inductive_head(env, head)
    if ind is None or len(ind.constructors) != 1:
        return []
    ctor = ind.constructors[0]
    if pat.ctor not in {ctor.name, f"{ind.name}.{ctor.name}"}:
        return []
    p = len(ind.param_types)
    params_actual = args[:p]
    field_tys = Telescope.of(
        *[
            t.inst_levels(level_actuals).instantiate(params_actual, depth_above=i)
            for i, t in enumerate(ctor.field_schemas)
        ]
    )
    if len(pat.args) != len(field_tys):
        return []
    binders: list[tuple[str, Term]] = []
    for field_ty, subpat in zip(field_tys, pat.args, strict=True):
        binders.extend(_collect_binders(env, field_ty, subpat))
    return binders
