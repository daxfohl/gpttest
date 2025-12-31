"""Surface inductive references and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from mltt.kernel.ast import Term, Univ, UApp
from mltt.kernel.env import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.telescope import ArgList, Telescope, decompose_uapp
from mltt.surface.elab_state import Constraint, ElabState
from mltt.surface.match import _resolve_inductive_head
from mltt.surface.sast import (
    NameEnv,
    SBinder,
    SPi,
    Span,
    SurfaceError,
    SurfaceTerm,
    _elab_binders,
)


@dataclass(frozen=True)
class SInd(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        decl = env.lookup_global(self.name)
        if decl is None or decl.value is None:
            raise SurfaceError(f"Unknown inductive {self.name}", self.span)
        if not isinstance(decl.value, Ind):
            raise SurfaceError(f"{self.name} is not an inductive", self.span)
        ind = decl.value
        if ind.uarity:
            levels = tuple(
                state.fresh_level_meta("implicit", self.span) for _ in range(ind.uarity)
            )
            term = UApp(ind, levels)
            return term, ind.infer_type(env).inst_levels(levels)
        return ind, ind.infer_type(env)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Inductive references require elaboration", self.span)


@dataclass(frozen=True)
class SCtor(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        decl = env.lookup_global(self.name)
        if decl is None or decl.value is None:
            raise SurfaceError(f"Unknown constructor {self.name}", self.span)
        if isinstance(decl.value, UApp) and isinstance(decl.value.head, Ctor):
            return decl.value, decl.ty
        if not isinstance(decl.value, Ctor):
            raise SurfaceError(f"{self.name} is not a constructor", self.span)
        ctor = decl.value
        if ctor.uarity:
            levels = tuple(
                state.fresh_level_meta("implicit", self.span)
                for _ in range(ctor.uarity)
            )
            term = UApp(ctor, levels)
            return term, ctor.infer_type(env).inst_levels(levels)
        return ctor, ctor.infer_type(env)

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

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        if env.lookup_global(self.name) is not None:
            raise SurfaceError(f"Duplicate inductive {self.name}", self.span)
        for binder in self.params:
            if binder.implicit:
                raise SurfaceError(
                    "Inductive parameters cannot be implicit", binder.span
                )
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
        param_tys, _param_impls, env_params = _elab_binders(env, state, self.params)
        index_tys, _index_impls, env_indices = _elab_binders(
            env_params, state, index_binders
        )
        level_term, level_ty = level_body.elab_infer(env_indices, state)
        level_ty_whnf = level_ty.whnf(env_indices)
        if not isinstance(level_ty_whnf, Univ):
            raise SurfaceError("Inductive level must be a universe", level_body.span)
        if not isinstance(level_term, Univ):
            raise SurfaceError("Inductive level must be a Type", level_body.span)
        ind = Ind(
            name=self.name,
            param_types=Telescope.of(*param_tys),
            index_types=Telescope.of(*index_tys),
            level=level_term.level,
            uarity=len(self.uparams),
        )
        mapping: dict[str, Term] = {self.name: ind}
        globals_dict = dict(env.globals)
        ind_ty = ind.infer_type(env)
        globals_dict[self.name] = GlobalDecl(
            ty=ind_ty, value=ind, reducible=True, uarity=ind.uarity
        )
        env_with_ind = Env(binders=env.binders, globals=MappingProxyType(globals_dict))
        env_params_with_ind = Env(
            binders=env_params.binders, globals=env_with_ind.globals
        )
        ctors: list[Ctor] = []
        for ctor_decl in self.ctors:
            ctor_name = f"{self.name}.{ctor_decl.name}"
            if env.lookup_global(ctor_name) is not None:
                raise SurfaceError(f"Duplicate constructor {ctor_name}", ctor_decl.span)
            field_tys, _field_impls, env_fields = _elab_binders(
                env_params_with_ind, state, ctor_decl.fields
            )
            result_indices = ArgList.empty()
            if len(index_tys) != 0 and ctor_decl.result is None:
                raise SurfaceError(
                    "Constructor must specify result indices", ctor_decl.span
                )
            if ctor_decl.result is not None:
                result_term, result_ty = ctor_decl.result.elab_infer(env_fields, state)
                result_ty_whnf = result_ty.whnf(env_fields)
                if not isinstance(result_ty_whnf, Univ):
                    raise SurfaceError(
                        "Constructor result must be a universe", ctor_decl.span
                    )
                head, _levels, args = decompose_uapp(result_term)
                ind_head = _resolve_inductive_head(env_with_ind, head)
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
                if args[:p] != param_vars:
                    raise SurfaceError(
                        "Constructor result parameters must be unchanged",
                        ctor_decl.span,
                    )
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
        object.__setattr__(ind, "constructors", tuple(ctors))
        state.level_names = old_level_names
        for ctor in ctors:
            ctor_name = f"{self.name}.{ctor.name}"
            mapping[ctor_name] = ctor
            ctor_ty = ctor.infer_type(env)
            globals_dict[ctor_name] = GlobalDecl(
                ty=ctor_ty, value=ctor, reducible=True, uarity=ctor.uarity
            )
        env1 = Env(binders=env.binders, globals=MappingProxyType(globals_dict))
        body_term, body_ty = self.body.elab_infer(env1, state)
        body_term = self._replace_defined(body_term, mapping)
        body_ty = self._replace_defined(body_ty, mapping)
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
