"""Surface inductive references and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from mltt.kernel.ast import Lam, Term, Univ, UApp
from mltt.kernel.environment import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.telescope import Telescope, decompose_uapp
from mltt.surface.elab_state import Constraint, ElabState
from mltt.surface.sast import (
    NameEnv,
    SBinder,
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
class SPatBinder:
    name: str | None
    span: Span


@dataclass(frozen=True)
class SBranch:
    ctor: str
    binders: tuple[SPatBinder, ...]
    rhs: SurfaceTerm
    span: Span


@dataclass(frozen=True)
class SMatch(SurfaceTerm):
    scrutinee: SurfaceTerm
    as_name: str | None
    motive: SurfaceTerm | None
    branches: tuple[SBranch, ...]

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        if self.motive is not None or self.as_name is not None:
            raise SurfaceError("Dependent match not implemented", self.span)
        raise SurfaceError("Cannot infer match result type; use check-mode", self.span)

    def elab_check(self, env: Env, state: ElabState, expected: Term) -> Term:
        if self.motive is not None or self.as_name is not None:
            raise SurfaceError("Dependent match not implemented", self.span)
        scrut_term, scrut_ty = self.scrutinee.elab_infer(env, state)
        scrut_ty_whnf = scrut_ty.whnf(env)
        head, level_actuals, args = decompose_uapp(scrut_ty_whnf)
        ind = self._resolve_inductive_head(env, head)
        if ind is None:
            raise SurfaceError("Match scrutinee is not an inductive type", self.span)
        p = len(ind.param_types)
        q = len(ind.index_types)
        if len(args) != p + q:
            raise SurfaceError("Match scrutinee has wrong arity", self.span)
        params_actual = args[:p]
        branch_map = self._branch_map(env, ind)
        cases: list[Term] = []
        for ctor in ind.constructors:
            branch = branch_map.get(ctor)
            if branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", self.span
                )
            field_tys = ctor.field_schemas.inst_levels(level_actuals).instantiate(
                params_actual
            )
            if len(branch.binders) != len(field_tys):
                raise SurfaceError(
                    f"Wrong number of binders for {ctor.name}", branch.span
                )
            env_branch = env
            for binder, field_ty in zip(branch.binders, field_tys, strict=True):
                env_branch = env_branch.push_binder(
                    field_ty, name=binder.name if binder.name is not None else None
                )
            rhs_term = branch.rhs.elab_check(
                env_branch, state, expected.shift(len(field_tys))
            )
            case_term = rhs_term
            for field_ty in reversed(list(field_tys)):
                case_term = Lam(field_ty, case_term)
            cases.append(case_term)
        motive = Lam(scrut_ty_whnf, expected.shift(1))
        return Elim(ind, motive, tuple(cases), scrut_term)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Match requires elaboration", self.span)

    def _resolve_inductive_head(self, env: Env, head: Term) -> Ind | None:
        if isinstance(head, Ind):
            return head
        if isinstance(head, Const):
            decl = env.lookup_global(head.name)
            if decl is not None and isinstance(decl.value, Ind):
                return decl.value
        return None

    def _branch_map(self, env: Env, ind: Ind) -> dict[Ctor, SBranch]:
        branches: dict[Ctor, SBranch] = {}
        ctor_by_name = {ctor.name: ctor for ctor in ind.constructors}
        ctor_by_qual = {f"{ind.name}.{ctor.name}": ctor for ctor in ind.constructors}
        seen: set[Ctor] = set()
        for branch in self.branches:
            ctor = None
            if branch.ctor in ctor_by_qual:
                ctor = ctor_by_qual[branch.ctor]
            elif branch.ctor in ctor_by_name:
                ctor = ctor_by_name[branch.ctor]
            else:
                decl = env.lookup_global(branch.ctor)
                if decl is not None and isinstance(decl.value, Ctor):
                    ctor = decl.value
            if ctor is None or ctor.inductive != ind:
                raise SurfaceError(f"Unknown constructor {branch.ctor}", branch.span)
            if ctor in seen:
                raise SurfaceError(
                    f"Duplicate branch for constructor {ctor.name}", branch.span
                )
            seen.add(ctor)
            branches[ctor] = branch
        return branches


@dataclass(frozen=True)
class SConstructorDecl:
    name: str
    fields: tuple[SBinder, ...]
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
        old_level_names = state.level_names
        state.level_names = list(reversed(self.uparams)) + state.level_names
        param_tys, _param_impls, env_params = _elab_binders(env, state, self.params)
        level_term, level_ty = self.level.elab_infer(env, state)
        level_ty_whnf = level_ty.whnf(env)
        if not isinstance(level_ty_whnf, Univ):
            raise SurfaceError("Inductive level must be a universe", self.level.span)
        if not isinstance(level_term, Univ):
            raise SurfaceError("Inductive level must be a Type", self.level.span)
        ind = Ind(
            name=self.name,
            param_types=Telescope.of(*param_tys),
            level=level_term.level,
            uarity=len(self.uparams),
        )
        ctors: list[Ctor] = []
        for ctor_decl in self.ctors:
            ctor_name = f"{self.name}.{ctor_decl.name}"
            if env.lookup_global(ctor_name) is not None:
                raise SurfaceError(f"Duplicate constructor {ctor_name}", ctor_decl.span)
            field_tys, _field_impls, _ = _elab_binders(
                env_params, state, ctor_decl.fields
            )
            ctors.append(
                Ctor(
                    name=ctor_decl.name,
                    inductive=ind,
                    field_schemas=Telescope.of(*field_tys),
                    uarity=ind.uarity,
                )
            )
        object.__setattr__(ind, "constructors", tuple(ctors))
        state.level_names = old_level_names
        mapping: dict[str, Term] = {self.name: ind}
        globals_dict = dict(env.globals)
        ind_ty = ind.infer_type(env)
        globals_dict[self.name] = GlobalDecl(
            ty=ind_ty, value=ind, reducible=True, uarity=ind.uarity
        )
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
