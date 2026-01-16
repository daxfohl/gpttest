"""Constraint solver for elaboration-time metavariables and universe levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

from mltt.common.span import Span
from mltt.kernel.ast import (
    App,
    Lam,
    Let,
    MetaVar,
    Pi,
    Term,
    UApp,
    Univ,
    Var,
)
from mltt.kernel.env import Const, Env
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.levels import LConst, LevelExpr, LMax, LMeta, LSucc, LVar
from mltt.kernel.tel import Spine
from mltt.solver.constraints import Constraint
from mltt.solver.errors import SolverError
from mltt.solver.levels import LMetaInfo, LevelConstraint
from mltt.solver.meta import Meta


@dataclass(frozen=True)
class _ArgDecision:
    kind: str
    arg: Term | None = None


class Solver:
    """Stateful solver for term and level constraints."""

    def __init__(self) -> None:
        self.metas: dict[int, Meta] = {}
        self.constraints: list[Constraint] = []
        self.level_metas: dict[int, LMetaInfo] = {}
        self.level_constraints: list[LevelConstraint] = []
        self._next_meta = 0
        self._next_level_meta = 0

    # --- Metas -------------------------------------------------------------
    def fresh_meta(self, env: Env, ty: Term, span: Span, kind: str) -> MetaVar:
        mid = self._next_meta
        self._next_meta += 1
        self.metas[mid] = Meta(ctx_len=len(env.binders), ty=ty, span=span, kind=kind)
        return MetaVar(mid=mid)

    def fresh_level_meta(self, origin: str, span: Span) -> LMeta:
        mid = self._next_level_meta
        self._next_level_meta += 1
        self.level_metas[mid] = LMetaInfo(span=span, origin=origin)
        return LMeta(mid)

    # --- Constraints -------------------------------------------------------
    def add_constraint(self, env: Env, lhs: Term, rhs: Term, span: Span) -> None:
        if not self._try_unify(env, lhs, rhs, span):
            self.constraints.append(Constraint(len(env.binders), lhs, rhs, span))

    def add_level_constraint(
        self, lhs: LevelExpr, rhs: LevelExpr, span: Span, reason: str | None = None
    ) -> None:
        self.level_constraints.append(
            LevelConstraint(lhs=lhs, rhs=rhs, span=span, reason=reason)
        )

    # --- Solve -------------------------------------------------------------
    def solve(self, env: Env) -> None:
        made_progress = True
        while made_progress and self.constraints:
            made_progress = False
            remaining: list[Constraint] = []
            for c in self.constraints:
                span = c.span or Span(0, 0)
                if self._try_unify(env, c.lhs, c.rhs, span):
                    made_progress = True
                else:
                    remaining.append(c)
            self.constraints = remaining
        self._solve_levels()

    def ensure_solved(self) -> None:
        for meta in self.metas.values():
            if meta.solution is not None:
                continue
            if meta.kind == "hole":
                span = meta.span or Span(0, 0)
                raise SolverError("Cannot synthesize value for hole", span)
            span = meta.span or Span(0, 0)
            raise SolverError("Cannot synthesize implicit argument", span)

    # --- Zonk --------------------------------------------------------------
    def zonk(self, term: Term) -> Term:
        match term:
            case MetaVar(mid=mid, args=args):
                meta = self.metas.get(mid)
                if meta is None or meta.solution is None:
                    return MetaVar(mid=mid, args=self._zonk_spine(args))
                solution = self.zonk(meta.solution)
                applied: Term = solution
                for arg in args:
                    applied = App(applied, self.zonk(arg))
                return applied
            case UApp(head=head, levels=levels):
                return UApp(head=self.zonk(head), levels=self._zonk_levels(levels))
            case Univ(level=level):
                return Univ(self._zonk_level(level))
            case _:
                return term._replace_terms(lambda t, _m: self.zonk(t))

    # --- Unification -------------------------------------------------------
    def _try_unify(self, env: Env, lhs: Term, rhs: Term, span: Span) -> bool:
        lhs = self.zonk(lhs).whnf(env)
        rhs = self.zonk(rhs).whnf(env)
        if lhs == rhs:
            return True
        if isinstance(lhs, MetaVar):
            return self._solve_meta(env, lhs, rhs, span)
        if isinstance(rhs, MetaVar):
            return self._solve_meta(env, rhs, lhs, span)
        match lhs, rhs:
            case Var(k1), Var(k2):
                if k1 != k2:
                    raise SolverError("Cannot unify terms", span)
                return True
            case Const(name=left_name), Const(name=right_name):
                if left_name != right_name:
                    raise SolverError("Cannot unify terms", span)
                return True
            case Ind(), Ind():
                if lhs != rhs:
                    raise SolverError("Cannot unify terms", span)
                return True
            case Ctor(), Ctor():
                if lhs != rhs:
                    raise SolverError("Cannot unify terms", span)
                return True
            case Lam(arg_ty=a1, body=b1), Lam(arg_ty=a2, body=b2):
                self._try_unify(env, a1, a2, span)
                env2 = env.push_binder(a1)
                self._try_unify(env2, b1, b2, span)
                return True
            case Pi(arg_ty=a1, return_ty=b1), Pi(arg_ty=a2, return_ty=b2):
                self._try_unify(env, a1, a2, span)
                env2 = env.push_binder(a1)
                self._try_unify(env2, b1, b2, span)
                return True
            case App(func=f1, arg=a1), App(func=f2, arg=a2):
                self._try_unify(env, f1, f2, span)
                self._try_unify(env, a1, a2, span)
                return True
            case Let(arg_ty=a1, value=v1, body=b1), Let(arg_ty=a2, value=v2, body=b2):
                self._try_unify(env, a1, a2, span)
                self._try_unify(env, v1, v2, span)
                env2 = env.push_let(a1, v1)
                self._try_unify(env2, b1, b2, span)
                return True
            case UApp(head=h1, levels=lv1), UApp(head=h2, levels=lv2):
                if len(lv1) != len(lv2):
                    raise SolverError("Universe application arity mismatch", span)
                self._try_unify(env, h1, h2, span)
                for left_level, right_level in zip(lv1, lv2, strict=True):
                    self._unify_levels(left_level, right_level, span)
                return True
            case Univ(level=l1), Univ(level=l2):
                self._unify_levels(l1, l2, span)
                return True
            case Elim(inductive=i1, motive=m1, cases=c1, scrutinee=s1), Elim(
                inductive=i2, motive=m2, cases=c2, scrutinee=s2
            ):
                if i1 != i2 or len(c1) != len(c2):
                    raise SolverError("Cannot unify terms", span)
                self._try_unify(env, m1, m2, span)
                for a, b in zip(c1, c2, strict=True):
                    self._try_unify(env, a, b, span)
                self._try_unify(env, s1, s2, span)
                return True
        raise SolverError("Cannot unify terms", span)

    def _solve_meta(self, env: Env, meta_term: MetaVar, rhs: Term, span: Span) -> bool:
        meta = self.metas[meta_term.mid]
        if meta.solution is not None:
            return self._try_unify(env, meta.solution, rhs, span)
        if isinstance(rhs, MetaVar) and rhs.mid == meta_term.mid:
            return True
        if self._occurs(meta_term.mid, rhs):
            raise SolverError("occurs check failed", span)
        args = list(meta_term.args)
        if not self._args_solvable(args):
            return False
        arg_vars = cast(list[Var], args)
        if not self._vars_subset(rhs, arg_vars):
            return False
        arg_types = self._pi_arg_types(meta.ty, len(args), env)
        if arg_types is None:
            return False
        body = self._abstract_over_args(rhs, arg_vars)
        solution = body
        for arg_ty in reversed(arg_types):
            solution = Lam(arg_ty, solution)
        self.metas[meta_term.mid] = Meta(
            ctx_len=meta.ctx_len,
            ty=meta.ty,
            solution=solution,
            span=meta.span,
            kind=meta.kind,
        )
        return True

    # --- Level solving ------------------------------------------------------
    def _solve_levels(self) -> None:
        for constraint in self.level_constraints:
            lhs = self._zonk_level(constraint.lhs)
            rhs = self._zonk_level(constraint.rhs)
            self._record_level_constraint(lhs, rhs, constraint)
        for mid, info in self.level_metas.items():
            if info.solution is not None:
                continue
            lower = self._zonk_level(info.lower_bound)
            upper_vals = [self._zonk_level(u) for u in info.upper_bounds]
            lower_eval = lower.eval()
            if lower_eval is None:
                continue
            for upper in upper_vals:
                upper_eval = upper.eval()
                if upper_eval is None:
                    continue
                if upper_eval < lower_eval:
                    span = info.span or Span(0, 0)
                    raise SolverError("Universe level constraint failed", span)
            info.solution = lower

    def _record_level_constraint(
        self, lhs: LevelExpr, rhs: LevelExpr, constraint: LevelConstraint
    ) -> None:
        match lhs, rhs:
            case LConst(left_const), LConst(right_const):
                if left_const > right_const:
                    span = constraint.span or Span(0, 0)
                    raise SolverError("Universe level constraint failed", span)
            case LMeta(mid), _:
                info = self.level_metas[mid]
                info.lower_bound = info.lower_bound.max(rhs)
            case _, LMeta(mid):
                info = self.level_metas[mid]
                info.upper_bounds.append(lhs)
            case _:
                if not rhs >= lhs:
                    span = constraint.span or Span(0, 0)
                    raise SolverError("Universe level constraint failed", span)

    def _unify_levels(self, lhs: LevelExpr, rhs: LevelExpr, span: Span) -> None:
        if lhs == rhs:
            return
        self.add_level_constraint(lhs, rhs, span)
        self.add_level_constraint(rhs, lhs, span)

    # --- Helpers ------------------------------------------------------------
    def _args_solvable(self, args: Iterable[Term]) -> bool:
        seen: set[int] = set()
        for arg in args:
            if not isinstance(arg, Var):
                return False
            if arg.k in seen:
                return False
            seen.add(arg.k)
        return True

    def _vars_subset(self, term: Term, args: list[Var]) -> bool:
        allowed = {arg.k for arg in args}
        return self._free_vars(term, 0).issubset(allowed)

    def _free_vars(self, term: Term, depth: int) -> set[int]:
        match term:
            case Var(k):
                return {k - depth} if k >= depth else set()
            case Lam(arg_ty=arg_ty, body=body):
                return self._free_vars(arg_ty, depth) | self._free_vars(body, depth + 1)
            case Pi(arg_ty=arg_ty, return_ty=body):
                return self._free_vars(arg_ty, depth) | self._free_vars(body, depth + 1)
            case Let(arg_ty=arg_ty, value=value, body=body):
                return (
                    self._free_vars(arg_ty, depth)
                    | self._free_vars(value, depth)
                    | self._free_vars(body, depth + 1)
                )
            case App(func=func, arg=arg):
                return self._free_vars(func, depth) | self._free_vars(arg, depth)
            case UApp(head=head, levels=_levels):
                return self._free_vars(head, depth)
            case Univ():
                return set()
            case MetaVar(args=args):
                free: set[int] = set()
                for arg in args:
                    free |= self._free_vars(arg, depth)
                return free
            case Elim(motive=motive, cases=cases, scrutinee=scrutinee):
                free = self._free_vars(motive, depth)
                for case in cases:
                    free |= self._free_vars(case, depth)
                free |= self._free_vars(scrutinee, depth)
                return free
            case _:
                return set()

    def _pi_arg_types(self, ty: Term, count: int, env: Env) -> list[Term] | None:
        arg_types: list[Term] = []
        current = ty
        for _ in range(count):
            current_whnf = current.whnf(env)
            if not isinstance(current_whnf, Pi):
                return None
            arg_types.append(current_whnf.arg_ty)
            current = current_whnf.return_ty
        return arg_types

    def _abstract_over_args(self, term: Term, args: list[Var]) -> Term:
        if not args:
            return term
        n = len(args)
        body = term.shift(n)
        for i, arg in enumerate(args):
            body = self._replace_var(body, arg.k + n, Var(n - 1 - i), 0)
        return body

    def _replace_var(
        self, term: Term, target: int, replacement: Term, depth: int
    ) -> Term:
        match term:
            case Var(k):
                if k == target + depth:
                    return replacement.shift(depth)
                return term
            case Lam(arg_ty=arg_ty, body=body):
                return Lam(
                    self._replace_var(arg_ty, target, replacement, depth),
                    self._replace_var(body, target, replacement, depth + 1),
                )
            case Pi(arg_ty=arg_ty, return_ty=body):
                return Pi(
                    self._replace_var(arg_ty, target, replacement, depth),
                    self._replace_var(body, target, replacement, depth + 1),
                )
            case Let(arg_ty=arg_ty, value=value, body=body):
                return Let(
                    arg_ty=self._replace_var(arg_ty, target, replacement, depth),
                    value=self._replace_var(value, target, replacement, depth),
                    body=self._replace_var(body, target, replacement, depth + 1),
                )
            case App(func=func, arg=arg):
                return App(
                    self._replace_var(func, target, replacement, depth),
                    self._replace_var(arg, target, replacement, depth),
                )
            case UApp(head=head, levels=levels):
                return UApp(
                    head=self._replace_var(head, target, replacement, depth),
                    levels=levels,
                )
            case MetaVar(mid=mid, args=args):
                return MetaVar(
                    mid=mid,
                    args=Spine.of(
                        *(
                            self._replace_var(arg, target, replacement, depth)
                            for arg in args
                        )
                    ),
                )
            case Elim(inductive=ind, motive=motive, cases=cases, scrutinee=scrutinee):
                return Elim(
                    inductive=ind,
                    motive=self._replace_var(motive, target, replacement, depth),
                    cases=tuple(
                        self._replace_var(case, target, replacement, depth)
                        for case in cases
                    ),
                    scrutinee=self._replace_var(scrutinee, target, replacement, depth),
                )
            case _:
                return term

    def _occurs(self, mid: int, term: Term) -> bool:
        match term:
            case MetaVar(mid=other):
                return mid == other
            case Lam(arg_ty=arg_ty, body=body):
                return self._occurs(mid, arg_ty) or self._occurs(mid, body)
            case Pi(arg_ty=arg_ty, return_ty=body):
                return self._occurs(mid, arg_ty) or self._occurs(mid, body)
            case Let(arg_ty=arg_ty, value=value, body=body):
                return (
                    self._occurs(mid, arg_ty)
                    or self._occurs(mid, value)
                    or self._occurs(mid, body)
                )
            case App(func=func, arg=arg):
                return self._occurs(mid, func) or self._occurs(mid, arg)
            case UApp(head=head, levels=_levels):
                return self._occurs(mid, head)
            case Elim(motive=motive, cases=cases, scrutinee=scrutinee):
                if self._occurs(mid, motive):
                    return True
                if any(self._occurs(mid, case) for case in cases):
                    return True
                return self._occurs(mid, scrutinee)
            case _:
                return False

    def _zonk_spine(self, args: Spine) -> Spine:
        return Spine.of(*(self.zonk(arg) for arg in args))

    def _zonk_levels(self, levels: tuple[LevelExpr, ...]) -> tuple[LevelExpr, ...]:
        return tuple(self._zonk_level(level) for level in levels)

    def _zonk_level(self, level: LevelExpr) -> LevelExpr:
        match level:
            case LMeta(mid=mid):
                info = self.level_metas.get(mid)
                if info is None or info.solution is None:
                    return level
                return self._zonk_level(info.solution)
            case LSucc(e):
                return LSucc(self._zonk_level(e))
            case LMax(a, b):
                return LMax(self._zonk_level(a), self._zonk_level(b))
            case LVar():
                return level
            case LConst():
                return level
        return level
