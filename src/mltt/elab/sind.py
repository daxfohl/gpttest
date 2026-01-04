"""Surface inductive references and definitions."""

from __future__ import annotations

from types import MappingProxyType

from mltt.kernel.ast import Term, Univ, UApp, Var
from mltt.kernel.env import Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.tel import ArgList, Telescope, decompose_uapp, mk_uapp
from mltt.elab.elab_state import Constraint, ElabState
from mltt.elab.etype import ElabBinderInfo, ElabEnv, ElabType
from mltt.elab.match import _resolve_inductive_head
from mltt.elab.names import NameEnv
from mltt.elab.east import (
    EBinder,
    ECtor,
    EInd,
    EInductiveDef,
    EConstructorDecl,
    EPi,
    ETerm,
)
from mltt.elab.sast import (
    _elab_binders,
    _expect_universe,
    _require_global_info,
    elab_infer,
)
from mltt.elab.errors import ElabError
from mltt.common.span import Span


def _normalize_binder_name(name: str | None) -> str | None:
    if name == "_":
        return None
    return name


def elab_ind_infer(term: EInd, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
    decl, gty = _require_global_info(
        env, term.name, term.span, f"Unknown inductive {term.name}"
    )
    if decl.value is None:
        raise ElabError(f"Unknown inductive {term.name}", term.span)
    if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ind):
        ind = decl.value.head
    elif isinstance(decl.value, Ind):
        ind = decl.value
    else:
        raise ElabError(f"{term.name} is not an inductive", term.span)
    term_k, levels = state.apply_implicit_levels(ind, decl.uarity, term.span)
    ty = gty.inst_levels(levels)
    return term_k, ElabType(state.zonk(ty.term), ty.binders)


def elab_ctor_infer(
    term: ECtor, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    decl, gty = _require_global_info(
        env, term.name, term.span, f"Unknown constructor {term.name}"
    )
    if decl.value is None:
        raise ElabError(f"Unknown constructor {term.name}", term.span)
    if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ctor):
        ctor = decl.value.head
        term_k, levels = state.apply_implicit_levels(ctor, decl.uarity, term.span)
        ty = gty.inst_levels(levels)
        return term_k, ElabType(state.zonk(ty.term), ty.binders)
    if not isinstance(decl.value, Ctor):
        raise ElabError(f"{term.name} is not a constructor", term.span)
    ctor = decl.value
    term_k, levels = state.apply_implicit_levels(ctor, decl.uarity, term.span)
    ty = gty.inst_levels(levels)
    return term_k, ElabType(state.zonk(ty.term), ty.binders)


def elab_inductive_infer(
    term: EInductiveDef, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if env.lookup_global(term.name) is not None:
        raise ElabError(f"Duplicate inductive {term.name}", term.span)
    if len(set(term.uparams)) != len(term.uparams):
        raise ElabError("Duplicate universe binder", term.span)
    index_binders: tuple[EBinder, ...] = ()
    level_body = term.level
    if isinstance(term.level, EPi):
        index_binders = term.level.binders
        level_body = term.level.body
    for binder in index_binders:
        if binder.implicit:
            raise ElabError("Inductive indices cannot be implicit", binder.span)
    param_tys, _param_impls, _param_levels, env_params = _elab_binders(
        env, state, term.params
    )
    index_tys, _index_impls, _index_levels, env_indices = _elab_binders(
        env_params, state, index_binders
    )
    level_term, level_ty = elab_infer(level_body, env_indices, state)
    _expect_universe(level_ty.term, env_indices.kenv, level_body.span)
    if not isinstance(level_term, Univ):
        raise ElabError("Inductive level must be a Type", level_body.span)
    uarity = len(term.uparams)
    if uarity == 0:
        terms = [*param_tys, *index_tys, level_term]
        terms = state.merge_type_level_metas(terms)
        uarity, generalized = state.generalize_levels(terms)
        param_tys = generalized[: len(param_tys)]
        index_tys = generalized[len(param_tys) : len(param_tys) + len(index_tys)]
        level_term = generalized[-1]
    if not isinstance(level_term, Univ):
        raise ElabError("Inductive level must be a Type", level_body.span)
    ind = Ind(
        name=term.name,
        param_types=Telescope.of(*param_tys),
        index_types=Telescope.of(*index_tys),
        level=level_term.level,
        uarity=uarity,
    )
    param_infos = tuple(
        ElabBinderInfo(
            name=_normalize_binder_name(binder.name),
            implicit=binder.implicit,
        )
        for binder in term.params
    )
    index_infos = tuple(
        ElabBinderInfo(
            name=_normalize_binder_name(binder.name),
            implicit=False,
        )
        for binder in index_binders
    )
    binder_infos = param_infos + index_infos
    globals_dict = dict(env.kenv.globals)
    ind_ty = ind.infer_type(env.kenv)
    globals_dict[term.name] = GlobalDecl(
        ty=ind_ty,
        value=ind,
        reducible=False,
        uarity=ind.uarity,
    )
    env_with_ind = ElabEnv(
        kenv=Env(binders=env.kenv.binders, globals=MappingProxyType(globals_dict)),
        locals=env.locals,
        eglobals={
            **env.eglobals,
            term.name: ElabType(ind_ty, binder_infos),
        },
    )
    env_params_with_ind = ElabEnv(
        kenv=Env(binders=env_params.kenv.binders, globals=env_with_ind.kenv.globals),
        locals=env_params.locals,
        eglobals=env_with_ind.eglobals,
    )
    ctors: list[Ctor] = []
    ctor_infos_map: dict[str, tuple[ElabBinderInfo, ...]] = {}
    for ctor_decl in term.ctors:
        ctor_name = f"{term.name}.{ctor_decl.name}"
        if env.lookup_global(ctor_name) is not None:
            raise ElabError(f"Duplicate constructor {ctor_name}", ctor_decl.span)
        field_tys, _field_impls, _field_levels, env_fields = _elab_binders(
            env_params_with_ind, state, ctor_decl.fields
        )
        result_indices = ArgList.empty()
        result_term, result_ty = elab_infer(ctor_decl.result, env_fields, state)
        _expect_universe(result_ty.term, env_fields.kenv, ctor_decl.span)
        head, _levels, args = decompose_uapp(result_term)
        ind_head = _resolve_inductive_head(env_with_ind.kenv, head)
        if ind_head != ind:
            raise ElabError("Constructor result must be the inductive", ctor_decl.span)
        p = len(ind.param_types)
        q = len(ind.index_types)
        if len(args) != p + q:
            raise ElabError("Constructor result has wrong arity", ctor_decl.span)
        params_actual = args[:p]
        result_indices = args[p:]
        # Parameters may be implicit; allow non-var terms and rely on constraints.
        result_indices = ArgList.of(*result_indices)
        ctor = Ctor(
            name=ctor_decl.name,
            inductive=ind,
            field_schemas=Telescope.of(*field_tys),
            result_indices=result_indices,
            uarity=ind.uarity,
        )
        ctor_ty = ctor.infer_type(env_with_ind.kenv)
        ctors.append(ctor)
        ctor_infos = tuple(
            ElabBinderInfo(
                name=_normalize_binder_name(binder.name),
                implicit=binder.implicit,
            )
            for binder in ctor_decl.fields
        )
        ctor_infos_map[ctor_name] = param_infos + ctor_infos
        globals_dict[ctor_name] = GlobalDecl(
            ty=ctor_ty,
            value=ctor,
            reducible=False,
            uarity=ctor.uarity,
        )
    ind = Ind(
        name=term.name,
        param_types=Telescope.of(*param_tys),
        index_types=Telescope.of(*index_tys),
        constructors=tuple(ctors),
        level=level_term.level,
        uarity=uarity,
    )
    globals_dict[term.name] = GlobalDecl(
        ty=ind_ty,
        value=ind,
        reducible=False,
        uarity=ind.uarity,
    )
    env_full = ElabEnv(
        kenv=Env(binders=env.kenv.binders, globals=MappingProxyType(globals_dict)),
        locals=env.locals,
        eglobals={
            **env.eglobals,
            term.name: ElabType(ind_ty, binder_infos),
            **{
                name: ElabType(globals_dict[name].ty, ctor_infos_map[name])
                for name in ctor_infos_map
            },
        },
    )
    env1 = ElabEnv(
        kenv=env_full.kenv,
        locals=env.locals,
        eglobals=env_full.eglobals,
    )
    body_term, body_ty = elab_infer(term.body, env1, state)
    if isinstance(body_term, UApp) and isinstance(body_term.head, Ind):
        body_term = body_term.head
    if isinstance(body_term, Ind) and body_term.name == term.name:
        body_term = ind
    return body_term, body_ty
