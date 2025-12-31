"""Surface pattern matching and let-pattern sugar."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.kernel.ast import App, Lam, Pi, Term, Univ
from mltt.kernel.env import Const, Env
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LevelExpr
from mltt.kernel.tel import (
    ArgList,
    Telescope,
    decompose_uapp,
    mk_app,
    mk_lams,
    mk_uapp,
)
from mltt.surface.elab_state import ElabState
from mltt.surface.etype import ElabEnv, ElabType
from mltt.surface.sast import (
    NameEnv,
    SVar,
    Span,
    SurfaceError,
    SurfaceTerm,
    _expect_universe,
)


def _resolve_inductive_head(env: Env, head: Term) -> Ind | None:
    if isinstance(head, Ind):
        return head
    if isinstance(head, Const):
        decl = env.lookup_global(head.name)
        if decl is not None and isinstance(decl.value, Ind):
            return decl.value
    return None


def _expand_tuple_pat(pat: Pat) -> Pat:
    if isinstance(pat, PatTuple):
        if len(pat.elts) < 2:
            return _expand_tuple_pat(pat.elts[0])
        left = _expand_tuple_pat(pat.elts[0])
        right = _expand_tuple_pat(
            PatTuple(span=pat.span, elts=pat.elts[1:])
            if len(pat.elts) > 2
            else pat.elts[1]
        )
        return PatCtor(span=pat.span, ctor="Pair", args=(left, right))
    if isinstance(pat, PatCtor):
        return PatCtor(
            span=pat.span,
            ctor=pat.ctor,
            args=tuple(_expand_tuple_pat(arg) for arg in pat.args),
        )
    return pat


def _looks_like_ctor(name: str) -> bool:
    return bool(name) and (name[0].isupper() or "." in name)


def _elab_scrutinee_info(
    scrutinee: SurfaceTerm, env: ElabEnv, state: ElabState
) -> tuple[Term, Term, Term, tuple[LevelExpr, ...], ArgList]:
    scrut_term, scrut_ty = scrutinee.elab_infer(env, state)
    scrut_ty_whnf = scrut_ty.term.whnf(env.kenv)
    head, level_actuals, args = decompose_uapp(scrut_ty_whnf)
    return scrut_term, scrut_ty_whnf, head, level_actuals, args


@dataclass(frozen=True)
class Pat:
    span: Span


@dataclass(frozen=True)
class PatVar(Pat):
    name: str


@dataclass(frozen=True)
class PatWild(Pat):
    pass


@dataclass(frozen=True)
class PatCtor(Pat):
    ctor: str
    args: tuple[Pat, ...]


@dataclass(frozen=True)
class PatTuple(Pat):
    elts: tuple[Pat, ...]


@dataclass(frozen=True)
class SBranch:
    pat: Pat
    rhs: SurfaceTerm
    span: Span


@dataclass(frozen=True)
class SMatch(SurfaceTerm):
    scrutinees: tuple[SurfaceTerm, ...]
    as_names: tuple[str | None, ...]
    motive: SurfaceTerm | None
    branches: tuple[SBranch, ...]

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        if len(self.scrutinees) != 1:
            return self._desugar().elab_infer(env, state)
        if self.motive is None and all(n is None for n in self.as_names):
            raise SurfaceError(
                "Cannot infer match result type; use check-mode", self.span
            )
        return self._elab_with_motive(env, state)

    def elab_check(self, env: ElabEnv, state: ElabState, expected: ElabType) -> Term:
        if len(self.scrutinees) != 1:
            return self._desugar().elab_check(env, state, expected)
        if self.motive is not None or any(n is not None for n in self.as_names):
            term, term_ty = self._elab_with_motive(env, state)
            state.add_constraint(env.kenv, term_ty.term, expected.term, self.span)
            return term
        desugared = self._desugar()
        if desugared is not self:
            return desugared.elab_check(env, state, expected)
        return self._elab_core(env, state, expected)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Match requires elaboration", self.span)

    def _elab_core(self, env: ElabEnv, state: ElabState, expected: ElabType) -> Term:
        self._check_duplicate_binders(self.branches)
        scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
            self.scrutinees[0], env, state
        )
        ind = _resolve_inductive_head(env.kenv, head)
        if ind is None:
            raise SurfaceError("Match scrutinee is not an inductive type", self.span)
        p = len(ind.param_types)
        q = len(ind.index_types)
        if len(args) != p + q:
            raise SurfaceError("Match scrutinee has wrong arity", self.span)
        params_actual = args[:p]
        branch_map, default_branch = self._branch_map(env, ind)
        cases: list[Term] = []
        for ctor in ind.constructors:
            branch = branch_map.get(ctor) or default_branch
            if branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", self.span
                )
            field_tys = self._field_types(ctor, level_actuals, params_actual)
            binder_names = self._branch_binders(branch.pat, ctor, field_tys)
            env_branch = env
            for binder_name, field_ty in zip(binder_names, field_tys, strict=True):
                env_branch = env_branch.push_binder(
                    ElabType(field_ty), name=binder_name
                )
            rhs_term = branch.rhs.elab_check(
                env_branch,
                state,
                ElabType(expected.term.shift(len(field_tys)), expected.implicit_spine),
            )
            case_term = rhs_term
            for field_ty in reversed(list(field_tys)):
                case_term = Lam(field_ty, case_term)
            cases.append(case_term)
        motive = Lam(scrut_ty_whnf, expected.term.shift(1))
        return Elim(ind, motive, tuple(cases), scrut_term)

    def _elab_with_motive(
        self, env: ElabEnv, state: ElabState
    ) -> tuple[Term, ElabType]:
        if self.motive is None:
            raise SurfaceError("Match motive missing", self.span)
        if len(self.scrutinees) != 1:
            raise SurfaceError("Dependent match needs one scrutinee", self.span)
        self._check_duplicate_binders(self.branches)
        scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
            self.scrutinees[0], env, state
        )
        ind = _resolve_inductive_head(env.kenv, head)
        if ind is None:
            raise SurfaceError("Match scrutinee is not an inductive type", self.span)
        p = len(ind.param_types)
        q = len(ind.index_types)
        if len(args) != p + q:
            raise SurfaceError("Match scrutinee has wrong arity", self.span)
        params_actual = args[:p]
        as_name = self.as_names[0] if self.as_names else None
        env_motive = env.push_binder(ElabType(scrut_ty_whnf), name=as_name or "_")
        motive_term, motive_ty = self.motive.elab_infer(env_motive, state)
        _expect_universe(motive_ty.term, env_motive.kenv, self.motive.span)
        motive = Lam(scrut_ty_whnf, motive_term)
        match_ty = App(motive, scrut_term)
        branch_map, default_branch = self._branch_map(env, ind)
        cases: list[Term] = []
        for ctor in ind.constructors:
            branch = branch_map.get(ctor) or default_branch
            if branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", self.span
                )
            field_tys = self._field_types(ctor, level_actuals, params_actual)
            m = len(field_tys)
            binder_names = self._branch_binders(branch.pat, ctor, field_tys)
            env_branch = env
            for binder_name, field_ty in zip(binder_names, field_tys, strict=True):
                env_branch = env_branch.push_binder(
                    ElabType(field_ty), name=binder_name
                )
            params_in_fields_ctx = params_actual.shift(m)
            field_vars = ArgList.vars(m)
            scrut_like = mk_uapp(
                ctor,
                level_actuals,
                params_in_fields_ctx,
                field_vars,
            )
            expected_branch = motive_term.shift(m).subst(scrut_like, m)
            rhs_term = branch.rhs.elab_check(
                env_branch, state, ElabType(expected_branch)
            )
            case_term = rhs_term
            for field_ty in reversed(list(field_tys)):
                case_term = Lam(field_ty, case_term)
            cases.append(case_term)
        return Elim(ind, motive, tuple(cases), scrut_term), ElabType(match_ty)

    def _field_types(
        self,
        ctor: Ctor,
        level_actuals: tuple[LevelExpr, ...],
        params_actual: ArgList,
    ) -> Telescope:
        return Telescope.of(
            *[
                t.inst_levels(level_actuals).instantiate(params_actual, depth_above=i)
                for i, t in enumerate(ctor.field_schemas)
            ]
        )

    def _branch_map(
        self, env: ElabEnv, ind: Ind
    ) -> tuple[dict[Ctor, SBranch], SBranch | None]:
        branches: dict[Ctor, SBranch] = {}
        default: SBranch | None = None
        ctor_by_name = {ctor.name: ctor for ctor in ind.constructors}
        ctor_by_qual = {f"{ind.name}.{ctor.name}": ctor for ctor in ind.constructors}
        seen: set[Ctor] = set()
        for branch in self.branches:
            if isinstance(branch.pat, PatWild):
                if default is not None:
                    raise SurfaceError("Duplicate wildcard branch", branch.span)
                default = branch
                continue
            pat = branch.pat
            if isinstance(pat, PatVar):
                ctor = ctor_by_name.get(pat.name) or ctor_by_qual.get(pat.name)
                if ctor is None or len(ctor.field_schemas) != 0:
                    raise SurfaceError(
                        "Match pattern must be constructor or _", branch.span
                    )
                pat = PatCtor(span=pat.span, ctor=pat.name, args=())
            if not isinstance(pat, PatCtor):
                raise SurfaceError("Unsupported match pattern", branch.span)
            ctor = None
            if pat.ctor in ctor_by_qual:
                ctor = ctor_by_qual[pat.ctor]
            elif pat.ctor in ctor_by_name:
                ctor = ctor_by_name[pat.ctor]
            else:
                decl = env.lookup_global(pat.ctor)
                if decl is not None and isinstance(decl.value, Ctor):
                    ctor = decl.value
            if ctor is None or ctor.inductive != ind:
                raise SurfaceError(f"Unknown constructor {pat.ctor}", branch.span)
            if ctor in seen:
                raise SurfaceError(
                    f"Duplicate branch for constructor {ctor.name}", branch.span
                )
            seen.add(ctor)
            branches[ctor] = branch
        return branches, default

    def _branch_binders(
        self, pat: Pat, ctor: Ctor, field_tys: Telescope
    ) -> list[str | None]:
        if isinstance(pat, PatWild):
            return [None for _ in field_tys]
        if isinstance(pat, PatVar):
            if len(field_tys) != 0:
                raise SurfaceError("Match pattern must be constructor or _", pat.span)
            return []
        if isinstance(pat, PatCtor):
            if len(pat.args) != len(field_tys):
                raise SurfaceError(f"Wrong number of binders for {ctor.name}", pat.span)
            names: list[str | None] = []
            for arg in pat.args:
                if isinstance(arg, PatVar):
                    names.append(arg.name)
                elif isinstance(arg, PatWild):
                    names.append(None)
                else:
                    raise SurfaceError("Nested patterns must be desugared", arg.span)
            return names
        raise SurfaceError("Unsupported match pattern", pat.span)

    def _desugar(self) -> SurfaceTerm:
        if len(self.scrutinees) != 1:
            return self._desugar_multi()
        expanded = self._expand_tuple_branches(self.branches)
        if self._needs_nested(expanded):
            return self._compile_nested(self.scrutinees[0], expanded)
        if expanded != self.branches:
            return SMatch(
                span=self.span,
                scrutinees=self.scrutinees,
                as_names=self.as_names,
                motive=self.motive,
                branches=expanded,
            )
        return self

    def _expand_tuple_branches(
        self, branches: tuple[SBranch, ...]
    ) -> tuple[SBranch, ...]:
        return tuple(
            SBranch(_expand_tuple_pat(branch.pat), branch.rhs, branch.span)
            for branch in branches
        )

    def _needs_nested(self, branches: tuple[SBranch, ...]) -> bool:
        return any(self._pat_nested(branch.pat) for branch in branches)

    def _pat_nested(self, pat: Pat) -> bool:
        if isinstance(pat, PatCtor):
            for arg in pat.args:
                if isinstance(arg, (PatCtor, PatTuple)):
                    return True
                if self._pat_nested(arg):
                    return True
        if isinstance(pat, PatTuple):
            return True
        return False

    def _compile_nested(
        self, scrutinee: SurfaceTerm, branches: tuple[SBranch, ...]
    ) -> SurfaceTerm:
        self._check_duplicate_binders(branches)
        default = self._extract_default(branches)
        if default is None:
            raise SurfaceError("Nested patterns require a final '_' branch", self.span)
        return self._compile_branches(scrutinee, branches[:-1], default.rhs)

    def _check_duplicate_binders(self, branches: tuple[SBranch, ...]) -> None:
        for branch in branches:
            seen: set[str] = set()
            for name in self._pat_bindings(branch.pat):
                if name in seen:
                    raise SurfaceError(
                        f"Duplicate binder {name} in pattern", branch.span
                    )
                seen.add(name)

    def _pat_bindings(self, pat: Pat) -> list[str]:
        if isinstance(pat, PatVar):
            return [pat.name]
        if isinstance(pat, PatCtor):
            names: list[str] = []
            for arg in pat.args:
                names.extend(self._pat_bindings(arg))
            return names
        if isinstance(pat, PatTuple):
            tuple_names: list[str] = []
            for elt in pat.elts:
                tuple_names.extend(self._pat_bindings(elt))
            return tuple_names
        return []

    def _extract_default(self, branches: tuple[SBranch, ...]) -> SBranch | None:
        if not branches:
            return None
        last = branches[-1]
        return last if isinstance(last.pat, PatWild) else None

    def _compile_branches(
        self,
        scrutinee: SurfaceTerm,
        branches: tuple[SBranch, ...],
        fallback: SurfaceTerm,
    ) -> SurfaceTerm:
        if not branches:
            return fallback
        head, *rest = branches
        next_fallback = self._compile_branches(scrutinee, tuple(rest), fallback)
        return self._compile_pat(scrutinee, head.pat, head.rhs, next_fallback)

    def _compile_pat(
        self,
        scrutinee: SurfaceTerm,
        pat: Pat,
        success: SurfaceTerm,
        failure: SurfaceTerm,
    ) -> SurfaceTerm:
        if isinstance(pat, PatWild):
            return success
        if isinstance(pat, PatVar):
            if _looks_like_ctor(pat.name):
                branch = SBranch(pat, success, pat.span)
                wildcard = SBranch(PatWild(pat.span), failure, pat.span)
                return SMatch(
                    span=pat.span,
                    scrutinees=(scrutinee,),
                    as_names=(None,),
                    motive=None,
                    branches=(branch, wildcard),
                )
            return SLetPat(span=pat.span, pat=pat, value=scrutinee, body=success)
        if isinstance(pat, PatCtor):
            nested: list[tuple[str, Pat]] = []
            flat_args: list[Pat] = []
            for arg in pat.args:
                if isinstance(arg, (PatVar, PatWild)):
                    flat_args.append(arg)
                else:
                    fresh = self._fresh_name()
                    flat_args.append(PatVar(name=fresh, span=arg.span))
                    nested.append((fresh, arg))
            inner = success
            for name, subpat in reversed(nested):
                inner = self._compile_pat(
                    SVar(span=subpat.span, name=name), subpat, inner, failure
                )
            branch = SBranch(
                PatCtor(span=pat.span, ctor=pat.ctor, args=tuple(flat_args)),
                inner,
                pat.span,
            )
            wildcard = SBranch(PatWild(pat.span), failure, pat.span)
            return SMatch(
                span=pat.span,
                scrutinees=(scrutinee,),
                as_names=(None,),
                motive=None,
                branches=(branch, wildcard),
            )
        raise SurfaceError("Unsupported pattern", pat.span)

    def _desugar_multi(self) -> SurfaceTerm:
        if self.motive is not None or any(n is not None for n in self.as_names):
            raise SurfaceError("Dependent match needs one scrutinee", self.span)
        self._check_duplicate_binders(self.branches)
        scrutinees = self.scrutinees
        branches = self._expand_tuple_in_multi(self.branches)
        if len(scrutinees) == 1:
            return SMatch(
                span=self.span,
                scrutinees=scrutinees,
                as_names=self.as_names,
                motive=self.motive,
                branches=branches,
            )
        default = self._extract_default(branches)
        if default is None:
            raise SurfaceError(
                "Multi-scrutinee match requires a final '_' branch", self.span
            )
        return self._compile_multi(scrutinees, branches[:-1], default.rhs)

    def _expand_tuple_in_multi(
        self, branches: tuple[SBranch, ...]
    ) -> tuple[SBranch, ...]:
        return tuple(
            SBranch(self._expand_tuple_multi_pat(branch.pat), branch.rhs, branch.span)
            for branch in branches
        )

    def _expand_tuple_multi_pat(self, pat: Pat) -> Pat:
        if isinstance(pat, PatTuple):
            return PatTuple(
                span=pat.span,
                elts=tuple(_expand_tuple_pat(elt) for elt in pat.elts),
            )
        return _expand_tuple_pat(pat)

    def _compile_multi(
        self,
        scrutinees: tuple[SurfaceTerm, ...],
        branches: tuple[SBranch, ...],
        fallback: SurfaceTerm,
    ) -> SurfaceTerm:
        if len(scrutinees) == 1:
            return self._compile_branches(scrutinees[0], branches, fallback)
        scrutinee = scrutinees[0]

        def compile_branch_list(
            branch_list: tuple[SBranch, ...], default_term: SurfaceTerm
        ) -> SurfaceTerm:
            if not branch_list:
                return default_term
            head, *rest = branch_list
            pat = head.pat
            if isinstance(pat, PatTuple):
                if len(pat.elts) != len(scrutinees):
                    raise SurfaceError("Tuple pattern arity mismatch", head.span)
                first_pat = pat.elts[0]
                rest_pat = (
                    pat.elts[1]
                    if len(pat.elts) == 2
                    else PatTuple(span=pat.span, elts=pat.elts[1:])
                )
            elif isinstance(pat, PatWild):
                first_pat = PatWild(pat.span)
                rest_pat = PatWild(pat.span)
            else:
                raise SurfaceError(
                    "Multi-scrutinee patterns must be tuple or _", head.span
                )
            inner = self._compile_multi(
                scrutinees[1:], (SBranch(rest_pat, head.rhs, head.span),), default_term
            )
            next_default = compile_branch_list(tuple(rest), default_term)
            return self._compile_pat(scrutinee, first_pat, inner, next_default)

        return compile_branch_list(branches, fallback)

    def _fresh_name(self) -> str:
        counter = getattr(self, "_pat_counter", 0)
        name = f"_pat{counter}"
        object.__setattr__(self, "_pat_counter", counter + 1)
        return name


@dataclass(frozen=True)
class SLetPat(SurfaceTerm):
    pat: Pat
    value: SurfaceTerm
    body: SurfaceTerm

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        value_term, value_ty = self.value.elab_infer(env, state)
        value_ty_whnf = value_ty.term.whnf(env.kenv)
        if not self._is_irrefutable(env.kenv, value_ty_whnf, self.pat):
            raise SurfaceError("Refutable pattern in let; use match", self.span)
        env_body = env
        for name, ty in self._collect_binders(env.kenv, value_ty_whnf, self.pat):
            env_body = env_body.push_binder(ElabType(ty), name=name)
        body_term, body_ty = self.body.elab_infer(env_body, state)
        match_term = SMatch(
            span=self.span,
            scrutinees=(self.value,),
            as_names=(None,),
            motive=None,
            branches=(SBranch(self.pat, self.body, self.span),),
        )
        match_term_k = match_term.elab_check(env, state, body_ty)
        return match_term_k, body_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Let-pattern requires elaboration", self.span)

    def _is_irrefutable(self, env: Env, scrut_ty: Term, pat: Pat) -> bool:
        pat = _expand_tuple_pat(pat)
        if isinstance(pat, (PatVar, PatWild)):
            return True
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
            if not self._is_irrefutable(env, field_ty, subpat):
                return False
        return True

    def _collect_binders(
        self, env: Env, scrut_ty: Term, pat: Pat
    ) -> list[tuple[str, Term]]:
        pat = _expand_tuple_pat(pat)
        if isinstance(pat, PatVar):
            return [(pat.name, scrut_ty)]
        if isinstance(pat, PatWild):
            return []
        if not isinstance(pat, PatCtor):
            return []
        head, level_actuals, args = decompose_uapp(scrut_ty)
        ind = _resolve_inductive_head(env, head)
        if ind is None or len(ind.constructors) != 1:
            return []
        ctor = ind.constructors[0]
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
            binders.extend(self._collect_binders(env, field_ty, subpat))
        return binders


@dataclass(frozen=True)
class SElimBinder:
    name: str | None
    span: Span


@dataclass(frozen=True)
class SElimBranch:
    ctor: str
    binders: tuple[SElimBinder, ...]
    rhs: SurfaceTerm
    span: Span


@dataclass(frozen=True)
class SElim(SurfaceTerm):
    scrutinee: SurfaceTerm
    as_name: str | None
    motive: SurfaceTerm
    branches: tuple[SElimBranch, ...]

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        scrut_term, scrut_ty_whnf, head, level_actuals, args = _elab_scrutinee_info(
            self.scrutinee, env, state
        )
        ind = _resolve_inductive_head(env.kenv, head)
        if ind is None:
            raise SurfaceError(
                "Eliminator scrutinee is not an inductive type", self.span
            )
        p = len(ind.param_types)
        q = len(ind.index_types)
        if len(args) != p + q:
            raise SurfaceError("Eliminator scrutinee has wrong arity", self.span)
        params_actual = args[:p]
        indices_actual = args[p:]
        motive_term = self._elab_motive(
            env,
            state,
            ind,
            level_actuals,
            params_actual,
            indices_actual,
            scrut_ty_whnf,
        )
        cases = self._elab_cases(
            env,
            state,
            ind,
            level_actuals,
            params_actual,
            indices_actual,
            motive_term,
        )
        elim_term = Elim(
            inductive=ind, motive=motive_term, cases=cases, scrutinee=scrut_term
        )
        elim_ty = mk_app(motive_term, indices_actual, scrut_term)
        return elim_term, ElabType(elim_ty)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Eliminator requires elaboration", self.span)

    def _elab_motive(
        self,
        env: ElabEnv,
        state: ElabState,
        ind: Ind,
        level_actuals: tuple[LevelExpr, ...],
        params_actual: ArgList,
        indices_actual: ArgList,
        scrut_ty: Term,
    ) -> Term:
        index_tys = ind.index_types.inst_levels(level_actuals).instantiate(
            params_actual
        )
        q = len(index_tys)
        params_in_indices_ctx = params_actual.shift(q)
        index_vars = ArgList.vars(q)
        scrut_in_indices_ctx = mk_uapp(
            ind, level_actuals, params_in_indices_ctx, index_vars
        )
        if self.as_name is not None:
            env_motive = env
            for idx_ty in index_tys:
                env_motive = env_motive.push_binder(ElabType(idx_ty))
            env_motive = env_motive.push_binder(
                ElabType(scrut_in_indices_ctx), name=self.as_name
            )
            motive_term, motive_ty = self.motive.elab_infer(env_motive, state)
            _expect_universe(motive_ty.term, env_motive.kenv, self.motive.span)
            body = motive_term.shift(q + 1)
            return mk_lams(*index_tys, scrut_in_indices_ctx, body=body)
        motive_term, motive_ty = self.motive.elab_infer(env, state)
        motive_ty_whnf = motive_ty.term.whnf(env.kenv)
        if isinstance(motive_ty_whnf, Univ):
            body = motive_term.shift(q + 1)
            return mk_lams(*index_tys, scrut_in_indices_ctx, body=body)
        motive_pi = motive_ty_whnf
        for _ in range(q + 1):
            if not isinstance(motive_pi, Pi):
                raise SurfaceError(
                    "Eliminator motive must be a function over indices and scrutinee",
                    self.motive.span,
                )
            motive_pi = motive_pi.return_ty.whnf(env.kenv)
        if not isinstance(motive_pi, Univ):
            raise SurfaceError(
                "Eliminator motive must return a universe", self.motive.span
            )
        return motive_term

    def _elab_cases(
        self,
        env: ElabEnv,
        state: ElabState,
        ind: Ind,
        level_actuals: tuple[LevelExpr, ...],
        params_actual: ArgList,
        indices_actual: ArgList,
        motive: Term,
    ) -> tuple[Term, ...]:
        branches: dict[Ctor, SElimBranch] = {}
        seen: set[Ctor] = set()
        ctor_by_name = {ctor.name: ctor for ctor in ind.constructors}
        ctor_by_qual = {f"{ind.name}.{ctor.name}": ctor for ctor in ind.constructors}
        for branch in self.branches:
            ctor = ctor_by_qual.get(branch.ctor) or ctor_by_name.get(branch.ctor)
            if ctor is None:
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
        cases: list[Term] = []
        for ctor in ind.constructors:
            case_branch = branches.get(ctor)
            if case_branch is None:
                raise SurfaceError(
                    f"Missing branch for constructor {ctor.name}", self.span
                )
            tel, codomain = self._case_telescope(
                ind,
                ctor,
                level_actuals,
                params_actual,
                indices_actual,
                motive,
            )
            binders = list(case_branch.binders)
            if len(binders) < len(tel):
                binders.extend(
                    SElimBinder(name=None, span=case_branch.span)
                    for _ in range(len(tel) - len(binders))
                )
            if len(binders) != len(tel):
                raise SurfaceError(
                    f"Wrong number of binders for {ctor.name}", case_branch.span
                )
            env_branch = env
            for binder, binder_ty in zip(binders, tel, strict=True):
                env_branch = env_branch.push_binder(
                    ElabType(binder_ty), name=binder.name
                )
            rhs_term = case_branch.rhs.elab_check(
                env_branch, state, ElabType(codomain.shift(len(tel)))
            )
            case_term = rhs_term
            for binder_ty in reversed(list(tel)):
                case_term = Lam(binder_ty, case_term)
            cases.append(case_term)
        return tuple(cases)

    def _case_telescope(
        self,
        ind: Ind,
        ctor: Ctor,
        level_actuals: tuple[LevelExpr, ...],
        params_actual: ArgList,
        indices_actual: ArgList,
        motive: Term,
    ) -> tuple[Telescope, Term]:
        p = len(ind.param_types)
        q = len(ind.index_types)
        ctor_field_types = Telescope.of(
            *[
                t.inst_levels(level_actuals).instantiate(params_actual, depth_above=i)
                for i, t in enumerate(ctor.field_schemas)
            ]
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
        ihs: list[Term] = []
        for ri, j in enumerate(ctor.rps):
            rec_head, rec_levels, rec_field_args = decompose_uapp(
                ctor_field_types[j].shift(m - j)
            )
            if rec_head != ind:
                raise SurfaceError("Recursive field head mismatch", self.span)
            if level_actuals and rec_levels and rec_levels != level_actuals:
                raise SurfaceError("Recursive field universe mismatch", self.span)
            rec_params = rec_field_args[:p]
            rec_indices = rec_field_args[p : p + q]
            ih_type = mk_app(motive_in_fields_ctx, rec_indices, field_vars[j])
            ihs.append(ih_type.shift(ri))
        ih_types = Telescope.of(*ihs)
        codomain = mk_app(motive_in_fields_ctx, result_indices, scrut_like).shift(
            len(ih_types)
        )
        tel = ctor_field_types + ih_types
        return tel, codomain
