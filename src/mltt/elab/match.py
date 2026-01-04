"""Surface pattern matching."""

from __future__ import annotations

from mltt.elab.ast import EBranch, EMatch, EPat, EPatCtor, EPatVar, EPatWild, ETerm
from mltt.elab.state import ElabState
from mltt.elab.types import ElabEnv, ElabType
from mltt.elab.term import expect_universe, elab_check, elab_infer
from mltt.kernel.ast import Lam, Term, UApp, Univ, Var
from mltt.kernel.env import Const, Env
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LevelExpr
from mltt.kernel.tel import ArgList, Telescope, decompose_uapp, mk_app, mk_lams, mk_uapp
from mltt.elab.errors import ElabError
from mltt.common.span import Span


def resolve_inductive_head(env: Env, head: Term) -> Ind | None:
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
    scrutinee: ETerm, env: ElabEnv, state: ElabState
) -> tuple[Term, Term, Term, tuple[LevelExpr, ...], ArgList]:
    scrut_term, scrut_ty = elab_infer(scrutinee, env, state)
    scrut_ty_whnf = scrut_ty.term.whnf(env.kenv)
    head, level_actuals, args = decompose_uapp(scrut_ty_whnf)
    return scrut_term, scrut_ty_whnf, head, level_actuals, args


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
            raise ElabError("Recursive field head mismatch", span)
        if level_actuals and rec_levels and rec_levels != level_actuals:
            raise ElabError("Recursive field universe mismatch", span)
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
    pat: EPat,
    ctor: Ctor,
    tel: Telescope,
    field_count: int,
) -> list[tuple[str | None, Term]]:
    binders: list[tuple[str | None, Term]] = []
    match pat:
        case EPatCtor():
            if len(pat.args) not in {field_count, len(tel)}:
                raise ElabError("Match pattern has wrong arity", pat.span)
            if len(pat.args) == field_count:
                full_args = pat.args + tuple(
                    EPatWild(pat.span) for _ in range(len(tel) - field_count)
                )
            else:
                full_args = pat.args
            for arg, ty in zip(full_args, tel, strict=True):
                match arg:
                    case EPatVar():
                        binders.append((arg.name, ty))
                    case EPatWild():
                        binders.append((None, ty))
                    case _:
                        raise ElabError(
                            "Match pattern must be constructor or _", arg.span
                        )
            return binders
        case EPatVar():
            if _looks_like_ctor(pat.name):
                if len(tel) == 0:
                    return []
                raise ElabError("Constructor pattern needs fields", pat.span)
            return [(pat.name, ty) for ty in tel]
        case EPatWild():
            return [(None, ty) for ty in tel]
        case _:
            raise ElabError("Unsupported match pattern", pat.span)


def _branch_map(
    branches: tuple[EBranch, ...], env: ElabEnv, ind: Ind
) -> tuple[dict[str, EBranch], ETerm | None]:
    branch_map: dict[str, EBranch] = {}
    default_branch: ETerm | None = None
    for branch in branches:
        pat = branch.pat
        match pat:
            case EPatWild():
                default_branch = branch.rhs
            case EPatVar():
                ctor_name = pat.name
                if not _looks_like_ctor(ctor_name):
                    raise ElabError(
                        "Match pattern must be constructor or _", branch.span
                    )
                branch_map[ctor_name] = branch
            case EPatCtor():
                branch_map[pat.ctor] = branch
            case _:
                raise ElabError("Unsupported match pattern", branch.span)
    for ctor in ind.constructors:
        qualified = f"{ind.name}.{ctor.name}"
        if ctor.name not in branch_map and qualified in branch_map:
            branch_map[ctor.name] = branch_map[qualified]
    _ = env
    return branch_map, default_branch


def elab_match_infer(
    match: EMatch, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if match.motive is None:
        if len(match.branches) != 1:
            raise ElabError(
                "Cannot infer match result type; use check-mode", match.span
            )
        level = state.fresh_level_meta("type", match.span)
        expected_ty = Univ(level)
        expected = state.fresh_meta(env.kenv, expected_ty, match.span, kind="match")
        term = _elab_match_core(match, env, state, ElabType(expected))
        return term, ElabType(expected)
    return _elab_match_with_motive(match, env, state)


def elab_match_check(
    match: EMatch, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    if match.motive is not None:
        term, term_ty = _elab_match_with_motive(match, env, state)
        state.add_constraint(env.kenv, term_ty.term, expected.term, match.span)
        return term
    return _elab_match_core(match, env, state, expected)


def _elab_match_core(
    match: EMatch, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
        match.scrutinee, env, state
    )
    ind = resolve_inductive_head(env.kenv, head)
    if ind is None:
        raise ElabError("Match scrutinee is not an inductive type", match.span)
    p = len(ind.param_types)
    q = len(ind.index_types)
    if len(args) != p + q:
        raise ElabError("Match scrutinee has wrong arity", match.span)
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
            raise ElabError("Cannot infer motive with non-variable indices", match.span)
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
                raise ElabError(
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
    match: EMatch, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if match.motive is None:
        raise ElabError("Match motive missing", match.span)
    scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
        match.scrutinee, env, state
    )
    ind = resolve_inductive_head(env.kenv, head)
    if ind is None:
        raise ElabError("Match scrutinee is not an inductive type", match.span)
    p = len(ind.param_types)
    q = len(ind.index_types)
    if len(args) != p + q:
        raise ElabError("Match scrutinee has wrong arity", match.span)
    params_actual = args[:p]
    indices_actual = args[p:]
    motive_term, motive_ty = elab_infer(match.motive, env, state)
    expect_universe(motive_ty.term, env.kenv, match.motive.span)
    scrut_ty_in_ctx = mk_uapp(
        ind, level_actuals, params_actual.shift(q), ArgList.vars(q)
    )
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
                raise ElabError(
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
