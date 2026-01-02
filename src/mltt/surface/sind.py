"""Surface inductive references and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from mltt.kernel.ast import Term, Univ, UApp
from mltt.kernel.env import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.tel import ArgList, Telescope, decompose_uapp, mk_uapp
from mltt.surface.elab_state import Constraint, ElabState
from mltt.surface.etype import ElabEnv, ElabType
from mltt.surface.match import _resolve_inductive_head
from mltt.surface.sast import (
    NameEnv,
    SBinder,
    SPi,
    Span,
    SurfaceError,
    SurfaceTerm,
    _elab_binders,
    _expect_universe,
    _require_global_info,
)


@dataclass(frozen=True)
class SInd(SurfaceTerm):
    name: str

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        decl, gty = _require_global_info(
            env, self.name, self.span, f"Unknown inductive {self.name}"
        )
        if decl.value is None:
            raise SurfaceError(f"Unknown inductive {self.name}", self.span)
        if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ind):
            ind = decl.value.head
        elif isinstance(decl.value, Ind):
            ind = decl.value
        else:
            raise SurfaceError(f"{self.name} is not an inductive", self.span)
        term, levels = state.apply_implicit_levels(ind, decl.uarity, self.span)
        ty = gty.inst_levels(levels)
        return term, ElabType(state.zonk(ty.term), ty.implicit_spine)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Inductive references require elaboration", self.span)


@dataclass(frozen=True)
class SCtor(SurfaceTerm):
    name: str

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        decl, gty = _require_global_info(
            env, self.name, self.span, f"Unknown constructor {self.name}"
        )
        if decl.value is None:
            raise SurfaceError(f"Unknown constructor {self.name}", self.span)
        if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ctor):
            ctor = decl.value.head
            term, levels = state.apply_implicit_levels(ctor, decl.uarity, self.span)
            ty = gty.inst_levels(levels)
            return term, ElabType(state.zonk(ty.term), ty.implicit_spine)
        if not isinstance(decl.value, Ctor):
            raise SurfaceError(f"{self.name} is not a constructor", self.span)
        ctor = decl.value
        term, levels = state.apply_implicit_levels(ctor, decl.uarity, self.span)
        ty = gty.inst_levels(levels)
        return term, ElabType(state.zonk(ty.term), ty.implicit_spine)

    def elab_check(self, env: ElabEnv, state: ElabState, expected: ElabType) -> Term:
        term, term_ty = self.elab_infer(env, state)
        expected_whnf = expected.term.whnf(env.kenv)
        ctor: Ctor | None = None
        if isinstance(term, Ctor):
            ctor = term
        elif isinstance(term, UApp) and isinstance(term.head, Ctor):
            ctor = term.head
        if ctor is not None and not ctor.field_schemas:
            expected_head, levels, args = decompose_uapp(expected.term)
            if not levels:
                expected_head, levels, args = decompose_uapp(expected_whnf)
            param_count = len(ctor.inductive.param_types)
            if len(args) >= param_count:
                params = args[:param_count]
                applied = mk_uapp(ctor, levels, params)
                applied_ty = (
                    ctor.infer_type(env.kenv).inst_levels(levels).instantiate(params)
                )
                state.add_constraint(env.kenv, applied_ty, expected.term, self.span)
                return applied
        state.add_constraint(env.kenv, term_ty.term, expected.term, self.span)
        return term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Constructor references require elaboration", self.span)


@dataclass(frozen=True)
class SConstructorDecl:
    name: str
    fields: tuple[SBinder, ...]
    result: SurfaceTerm | None
    span: Span


@dataclass(frozen=True)
class SInductiveDef(SurfaceTerm):
    name: str
    uparams: tuple[str, ...]
    params: tuple[SBinder, ...]
    level: SurfaceTerm
    ctors: tuple[SConstructorDecl, ...]
    body: SurfaceTerm

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        if env.lookup_global(self.name) is not None:
            raise SurfaceError(f"Duplicate inductive {self.name}", self.span)
        if len(set(self.uparams)) != len(self.uparams):
            raise SurfaceError("Duplicate universe binder", self.span)
        index_binders: tuple[SBinder, ...] = ()
        level_body = self.level
        if isinstance(self.level, SPi):
            index_binders = self.level.binders
            level_body = self.level.body
        for binder in index_binders:
            if binder.implicit:
                raise SurfaceError("Inductive indices cannot be implicit", binder.span)
        old_level_names = state.level_names
        state.level_names = list(reversed(self.uparams)) + state.level_names
        param_tys, _param_impls, _param_levels, env_params = _elab_binders(
            env, state, self.params
        )
        index_tys, _index_impls, _index_levels, env_indices = _elab_binders(
            env_params, state, index_binders
        )
        level_term, level_ty = level_body.elab_infer(env_indices, state)
        _expect_universe(level_ty.term, env_indices.kenv, level_body.span)
        if not isinstance(level_term, Univ):
            raise SurfaceError("Inductive level must be a Type", level_body.span)
        uarity = len(self.uparams)
        if uarity == 0:
            terms = [*param_tys, *index_tys, level_term]
            terms = state.merge_type_level_metas(terms)
            uarity, generalized = state.generalize_levels(terms)
            param_tys = generalized[: len(param_tys)]
            index_tys = generalized[len(param_tys) : len(param_tys) + len(index_tys)]
            level_term = generalized[-1]
        if not isinstance(level_term, Univ):
            raise SurfaceError("Inductive level must be a Type", level_body.span)
        ind = Ind(
            name=self.name,
            param_types=Telescope.of(*param_tys),
            index_types=Telescope.of(*index_tys),
            level=level_term.level,
            uarity=uarity,
        )
        param_impls = tuple(binder.implicit for binder in self.params)
        mapping: dict[str, Term] = {self.name: ind}
        globals_dict = dict(env.kenv.globals)
        ind_ty = ind.infer_type(env.kenv)
        globals_dict[self.name] = GlobalDecl(
            ty=ind_ty,
            value=ind,
            reducible=False,
            uarity=ind.uarity,
        )
        env_with_ind = ElabEnv(
            kenv=Env(binders=env.kenv.binders, globals=MappingProxyType(globals_dict)),
            locals=env.locals,
            eglobals={**env.eglobals, self.name: ElabType(ind_ty, param_impls)},
        )
        env_params_with_ind = ElabEnv(
            kenv=Env(
                binders=env_params.kenv.binders, globals=env_with_ind.kenv.globals
            ),
            locals=env_params.locals,
            eglobals=env_with_ind.eglobals,
        )
        ctors: list[Ctor] = []
        ctor_impls_map: dict[str, tuple[bool, ...]] = {}
        for ctor_decl in self.ctors:
            ctor_name = f"{self.name}.{ctor_decl.name}"
            if env.lookup_global(ctor_name) is not None:
                raise SurfaceError(f"Duplicate constructor {ctor_name}", ctor_decl.span)
            field_tys, _field_impls, _field_levels, env_fields = _elab_binders(
                env_params_with_ind, state, ctor_decl.fields
            )
            result_indices = ArgList.empty()
            if len(index_tys) != 0 and ctor_decl.result is None:
                raise SurfaceError(
                    "Constructor must specify result indices", ctor_decl.span
                )
            if ctor_decl.result is not None:
                result_term, result_ty = ctor_decl.result.elab_infer(env_fields, state)
                _expect_universe(result_ty.term, env_fields.kenv, ctor_decl.span)
                head, _levels, args = decompose_uapp(result_term)
                ind_head = _resolve_inductive_head(env_with_ind.kenv, head)
                if ind_head != ind:
                    raise SurfaceError(
                        "Constructor result must be the inductive", ctor_decl.span
                    )
                p = len(ind.param_types)
                q = len(ind.index_types)
                if len(args) != p + q:
                    raise SurfaceError(
                        "Constructor result has wrong arity", ctor_decl.span
                    )
                param_vars = ArgList.vars(p, len(field_tys))
                for arg, expected in zip(args[:p], param_vars):
                    state.add_constraint(env_fields.kenv, arg, expected, ctor_decl.span)
                result_indices = args[p:]
            field_tys = [
                self._replace_defined(field_ty, mapping) for field_ty in field_tys
            ]
            result_indices = ArgList.of(
                *(self._replace_defined(idx, mapping) for idx in result_indices)
            )
            ctors.append(
                Ctor(
                    name=ctor_decl.name,
                    inductive=ind,
                    field_schemas=Telescope.of(*field_tys),
                    result_indices=result_indices,
                    uarity=ind.uarity,
                )
            )
            ctor_impls_map[ctor_decl.name] = param_impls + tuple(
                binder.implicit for binder in ctor_decl.fields
            )
        object.__setattr__(ind, "constructors", tuple(ctors))
        state.level_names = old_level_names
        for ctor in ctors:
            ctor_name = f"{self.name}.{ctor.name}"
            mapping[ctor_name] = ctor
            ctor_ty = ctor.infer_type(env.kenv)
            globals_dict[ctor_name] = GlobalDecl(
                ty=ctor_ty,
                value=ctor,
                reducible=False,
                uarity=ctor.uarity,
            )
        env1 = ElabEnv(
            kenv=Env(binders=env.kenv.binders, globals=MappingProxyType(globals_dict)),
            locals=env.locals,
            eglobals={
                **env.eglobals,
                self.name: ElabType(ind_ty, param_impls),
                **{
                    f"{self.name}.{ctor.name}": ElabType(
                        ctor.infer_type(env.kenv),
                        ctor_impls_map.get(ctor.name, ()),
                    )
                    for ctor in ctors
                },
            },
        )
        body_term, body_ty = self.body.elab_infer(env1, state)
        body_term = self._replace_defined(body_term, mapping)
        body_ty = ElabType(
            self._replace_defined(body_ty.term, mapping), body_ty.implicit_spine
        )
        state.constraints = [
            Constraint(
                c.ctx_len,
                self._replace_defined(c.lhs, mapping),
                self._replace_defined(c.rhs, mapping),
                c.span,
                c.kind,
            )
            for c in state.constraints
        ]
        for meta in state.metas.values():
            meta.ty = self._replace_defined(meta.ty, mapping)
            if meta.solution is not None:
                meta.solution = self._replace_defined(meta.solution, mapping)
        return body_term, body_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Inductive definitions require elaboration", self.span)

    def _replace_defined(self, term: Term, mapping: dict[str, Term]) -> Term:
        if isinstance(term, Const) and term.name in mapping:
            return mapping[term.name]
        if isinstance(term, UApp) and isinstance(term.head, Const):
            if term.head.name in mapping:
                return UApp(mapping[term.head.name], term.levels)
        return term._replace_terms(lambda sub, _m: self._replace_defined(sub, mapping))
