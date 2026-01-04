"""Constraint solving and zonking helpers."""

from __future__ import annotations

from collections import deque

from mltt.common.span import Span
from mltt.elab.errors import ElabError
from mltt.elab.state import Constraint, ElabState, LevelConstraint
from mltt.kernel.ast import App, Lam, MetaVar, Pi, Term, Univ, Var, UApp
from mltt.kernel.env import Env
from mltt.kernel.ind import Elim
from mltt.kernel.levels import LMax, LMeta, LSucc, LevelExpr


def solve(state: ElabState, env: Env) -> None:
    _ConstraintSolver(state).solve(env)


def ensure_solved(state: ElabState) -> None:
    _ConstraintSolver(state).ensure_solved()


def zonk(state: ElabState, term: Term) -> Term:
    return _ConstraintSolver(state).zonk(term)


def zonk_level(state: ElabState, level: LevelExpr) -> LevelExpr:
    return _ConstraintSolver(state).zonk_level(level)


def solve_meta(
    state: ElabState,
    env: Env,
    constraint: Constraint,
    lhs: Term,
    rhs: Term,
) -> bool:
    return _ConstraintSolver(state)._solve_meta(env, constraint, lhs, rhs)


class _ConstraintSolver:
    def __init__(self, state: ElabState) -> None:
        self._state = state

    def solve(self, env: Env) -> None:
        queue: deque[Constraint] = deque(self._state.constraints)
        postponed: list[Constraint] = []
        self._state.constraints = []
        progress = True
        while queue or progress:
            progress = False
            while queue:
                constraint = queue.popleft()
                result = self._solve_constraint(env, constraint)
                if self._state.constraints:
                    queue.extendleft(
                        reversed(
                            [
                                c
                                for c in self._state.constraints
                                if self._constraint_has_meta(c)
                            ]
                        )
                    )
                    self._state.constraints = []
                if result == "solved":
                    progress = True
                    continue
                if result == "progress":
                    progress = True
                    queue.extendleft(postponed)
                    postponed = []
                    continue
                postponed.append(constraint)
                if not queue:
                    if postponed:
                        queue.extend(postponed)
                        postponed = []
                        if all(self._is_stuck(env, c) for c in queue):
                            smallest = min(
                                queue, key=lambda c: c.span.start if c.span else 0
                            )
                            span = smallest.span or Span(0, 0)
                            lhs = self.zonk(smallest.lhs)
                            rhs = self.zonk(smallest.rhs)
                            raise ElabError(f"Stuck constraint: {lhs} ≡ {rhs}", span)
            if self._state.constraints:
                queue.extendleft(
                    reversed(
                        [
                            c
                            for c in self._state.constraints
                            if self._constraint_has_meta(c)
                        ]
                    )
                )
                self._state.constraints = []
                progress = True
            progress = self._solve_level_constraints() or progress

    def zonk(self, term: Term) -> Term:
        cache: dict[int, Term] = {}

        def walk(t: Term) -> Term:
            if isinstance(t, MetaVar):
                if t.mid in cache:
                    return cache[t.mid]
                meta = self._state.metas.get(t.mid)
                if meta is not None and meta.solution is not None:
                    cache[t.mid] = walk(meta.solution)
                    return cache[t.mid]
                return t
            if isinstance(t, Univ):
                return Univ(self.zonk_level(t.level))
            if isinstance(t, UApp):
                head = walk(t.head)
                levels = tuple(self.zonk_level(level) for level in t.levels)
                return UApp(head, levels)
            replaced = t._replace_terms(lambda sub, _m: walk(sub))
            return self._zonk_levels(replaced)

        return walk(term)

    def zonk_level(self, level: LevelExpr) -> LevelExpr:
        match level:
            case LMeta(mid):
                meta = self._state.level_metas.get(mid)
                if meta is not None and meta.solution is not None:
                    return self.zonk_level(meta.solution)
                return level
            case LSucc(e):
                return LSucc(self.zonk_level(e))
            case LMax(a, b):
                return LMax(self.zonk_level(a), self.zonk_level(b))
            case _:
                return level

    def ensure_solved(self) -> None:
        unsolved = [
            (mid, meta)
            for mid, meta in self._state.metas.items()
            if meta.solution is None
        ]
        if unsolved:
            mid, meta = unsolved[0]
            ty = self.zonk(meta.ty)
            span = meta.span or Span(0, 0)
            if meta.kind == "implicit":
                message = f"Cannot infer implicit argument ?m{mid}; expected type {ty}"
            else:
                message = (
                    f"Cannot synthesize value for hole ?m{mid}; expected type {ty}"
                )
            raise ElabError(message, span)
        for mid, info in self._state.level_metas.items():
            if info.solution is None:
                span = info.span or Span(0, 0)
                raise ElabError(f"Cannot infer universe level ?u{mid}", span)

    def _solve_constraint(self, env: Env, constraint: Constraint) -> str:
        ctx_env = self._env_for_ctx(env, constraint.ctx_len)
        lhs = self.zonk(constraint.lhs).whnf(ctx_env)
        rhs = self.zonk(constraint.rhs).whnf(ctx_env)
        if lhs == rhs:
            return "solved"
        if self._solve_meta(ctx_env, constraint, lhs, rhs):
            return "progress"
        if isinstance(lhs, Pi):
            from mltt.kernel.tel import decompose_app
            from mltt.kernel.tel import decompose_uapp

            head, args = decompose_app(rhs)
            _ = head
            pi_count = 0
            pi_cursor: Term = lhs
            while isinstance(pi_cursor, Pi):
                pi_count += 1
                pi_cursor = pi_cursor.return_ty
            if len(args) >= pi_count:
                rhs_whnf = rhs.whnf(ctx_env)
                lhs_body = pi_cursor.whnf(ctx_env)
                lhs_head, _, _ = decompose_uapp(lhs_body)
                rhs_head, _, _ = decompose_uapp(rhs_whnf)
                if lhs_head == rhs_head:
                    return "progress"
        if isinstance(lhs, Univ) and isinstance(rhs, Univ):
            self._state.add_level_constraint(lhs.level, rhs.level, constraint.span)
            self._state.add_level_constraint(rhs.level, lhs.level, constraint.span)
            return "progress"
        if type(lhs) is not type(rhs):
            raise ElabError(
                f"Cannot unify {lhs} with {rhs}", constraint.span or Span(0, 0)
            )
        if isinstance(lhs, Pi) and isinstance(rhs, Pi):
            self._state.constraints.append(
                Constraint(constraint.ctx_len, lhs.arg_ty, rhs.arg_ty, constraint.span)
            )
            self._state.constraints.append(
                Constraint(
                    constraint.ctx_len + 1,
                    lhs.return_ty,
                    rhs.return_ty,
                    constraint.span,
                )
            )
            return "progress"
        if isinstance(lhs, Lam) and isinstance(rhs, Lam):
            self._state.constraints.append(
                Constraint(constraint.ctx_len + 1, lhs.body, rhs.body, constraint.span)
            )
            return "progress"
        if isinstance(lhs, App) and isinstance(rhs, App):
            self._state.constraints.append(
                Constraint(constraint.ctx_len, lhs.func, rhs.func, constraint.span)
            )
            self._state.constraints.append(
                Constraint(constraint.ctx_len, lhs.arg, rhs.arg, constraint.span)
            )
            return "progress"
        if (
            isinstance(lhs, Elim)
            and isinstance(rhs, Elim)
            and lhs.inductive == rhs.inductive
        ):
            self._state.constraints.append(
                Constraint(constraint.ctx_len, lhs.motive, rhs.motive, constraint.span)
            )
            self._state.constraints.append(
                Constraint(
                    constraint.ctx_len, lhs.scrutinee, rhs.scrutinee, constraint.span
                )
            )
            for l_case, r_case in zip(lhs.cases, rhs.cases, strict=True):
                self._state.constraints.append(
                    Constraint(constraint.ctx_len, l_case, r_case, constraint.span)
                )
            return "progress"
        raise ElabError(f"Cannot unify {lhs} with {rhs}", constraint.span or Span(0, 0))

    def _solve_meta(
        self, env: Env, constraint: Constraint, lhs: Term, rhs: Term
    ) -> bool:
        if isinstance(lhs, MetaVar) and self._try_solve_meta(env, lhs, rhs):
            return True
        if isinstance(rhs, MetaVar) and self._try_solve_meta(env, rhs, lhs):
            return True
        lhs_head, lhs_spine = self._decompose_app(lhs)
        rhs_head, rhs_spine = self._decompose_app(rhs)
        if isinstance(lhs_head, MetaVar) and self._try_solve_spine(
            env, lhs_head, lhs_spine, rhs, constraint
        ):
            return True
        if isinstance(rhs_head, MetaVar) and self._try_solve_spine(
            env, rhs_head, rhs_spine, lhs, constraint
        ):
            return True
        return False

    def _constraint_has_meta(self, constraint: Constraint) -> bool:
        return self._term_has_meta(constraint.lhs) or self._term_has_meta(
            constraint.rhs
        )

    def _term_has_meta(self, term: Term) -> bool:
        if isinstance(term, MetaVar):
            return True
        found = False

        def walk(sub: Term, _meta: object) -> Term:
            nonlocal found
            if found:
                return sub
            if isinstance(sub, MetaVar):
                found = True
                return sub
            return sub._replace_terms(walk)

        term._replace_terms(walk)
        return found

    def _try_solve_meta(self, env: Env, meta_term: MetaVar, rhs: Term) -> bool:
        meta = self._state.metas.get(meta_term.mid)
        if meta is None or meta.solution is not None:
            return False
        rhs = self.zonk(rhs)
        if self._occurs(meta_term.mid, rhs):
            raise ElabError(
                "Cannot solve hole: occurs check failed", meta.span or Span(0, 0)
            )
        adapted = self._adapt_to_ctx(rhs, len(env.binders), meta.ctx_len)
        meta.solution = adapted
        return True

    def _try_solve_spine(
        self,
        env: Env,
        meta_term: MetaVar,
        spine: list[Term],
        rhs: Term,
        constraint: Constraint,
    ) -> bool:
        meta = self._state.metas.get(meta_term.mid)
        if meta is None or meta.solution is not None:
            return False
        if not spine:
            return False
        var_indices: list[int] = []
        for arg in spine:
            if not isinstance(arg, Var):
                return False
            var_indices.append(arg.k)
        if len(set(var_indices)) != len(var_indices):
            return False
        ctx_len = len(env.binders)
        diff = meta.ctx_len - ctx_len
        restricted_rhs = self._adapt_to_ctx(rhs, ctx_len, meta.ctx_len)
        adjusted_indices: list[int] = []
        for k in var_indices:
            if diff < 0:
                drop = -diff
                if k < drop:
                    return False
                adjusted_indices.append(k - drop)
            else:
                adjusted_indices.append(k + diff)
        arg_tys = self._pi_spine(meta.ty, meta.ctx_len, len(spine), env)
        term = restricted_rhs
        for offset, (k, arg_ty) in enumerate(
            zip(reversed(adjusted_indices), reversed(arg_tys))
        ):
            term = self._abstract_var(term, k + offset)
            term = Lam(arg_ty, term)
        meta.solution = term
        return True

    def _pi_spine(self, ty: Term, ctx_len: int, count: int, env: Env) -> list[Term]:
        arg_tys: list[Term] = []
        current = ty
        ctx_env = self._env_for_ctx(env, ctx_len)
        for _ in range(count):
            current = self.zonk(current).whnf(ctx_env)
            if not isinstance(current, Pi):
                raise ElabError("Cannot solve meta: type is not a function", Span(0, 0))
            arg_tys.append(current.arg_ty)
            current = current.return_ty
        return arg_tys

    def _solve_level_constraints(self) -> bool:
        progress = False
        queue: deque[LevelConstraint] = deque(self._state.level_constraints)
        self._state.level_constraints = []
        while queue:
            constraint = queue.popleft()
            lhs = self.zonk_level(constraint.lhs)
            rhs = self.zonk_level(constraint.rhs)
            match rhs:
                case LMeta(mid):
                    info = self._state.level_metas[mid]
                    new_lb = self._level_max(info.lower_bound, lhs)
                    if new_lb != info.lower_bound:
                        info.lower_bound = new_lb
                        progress = True
                    continue
            match lhs:
                case LMeta(mid):
                    info = self._state.level_metas[mid]
                    info.upper_bounds.append(rhs)
                    continue
                case LMax(a, b):
                    queue.append(
                        LevelConstraint(a, rhs, constraint.span, constraint.reason)
                    )
                    queue.append(
                        LevelConstraint(b, rhs, constraint.span, constraint.reason)
                    )
                    continue
            lhs_val = self._level_eval(lhs)
            rhs_val = self._level_eval(rhs)
            if lhs_val is not None and rhs_val is not None:
                if lhs_val > rhs_val:
                    span = constraint.span or Span(0, 0)
                    raise ElabError(
                        f"Universe level mismatch: {lhs} ≤ {rhs} does not hold", span
                    )
        for mid, info in self._state.level_metas.items():
            if info.solution is None:
                candidate = self.zonk_level(info.lower_bound)
                for ub in info.upper_bounds:
                    ub_val = self._level_eval(self.zonk_level(ub))
                    cand_val = self._level_eval(candidate)
                    if ub_val is not None and cand_val is not None:
                        if cand_val > ub_val:
                            span = info.span or Span(0, 0)
                            raise ElabError(
                                "Universe level mismatch: lower bound exceeds upper",
                                span,
                            )
                info.solution = candidate
                progress = True
        return progress

    def _level_eval(self, level: LevelExpr) -> int | None:
        return level.eval()

    def _level_max(self, a: LevelExpr, b: LevelExpr) -> LevelExpr:
        return a.max(b)

    def _is_stuck(self, env: Env, constraint: Constraint) -> bool:
        ctx_env = self._env_for_ctx(env, constraint.ctx_len)
        lhs = self.zonk(constraint.lhs).whnf(ctx_env)
        rhs = self.zonk(constraint.rhs).whnf(ctx_env)
        lhs_head, _ = self._decompose_app(lhs)
        rhs_head, _ = self._decompose_app(rhs)
        return isinstance(lhs_head, MetaVar) or isinstance(rhs_head, MetaVar)

    def _decompose_app(self, term: Term) -> tuple[Term, list[Term]]:
        spine: list[Term] = []
        head = term
        while isinstance(head, App):
            spine.append(head.arg)
            head = head.func
        spine.reverse()
        return head, spine

    def _env_for_ctx(self, env: Env, ctx_len: int) -> Env:
        if ctx_len == len(env.binders):
            return env
        if ctx_len < len(env.binders):
            return Env(binders=env.binders[:ctx_len], globals=env.globals)
        extra = ctx_len - len(env.binders)
        dummy = Env(globals=env.globals)
        for _ in range(extra):
            dummy = dummy.push_binder(Univ(0))
        return Env(binders=dummy.binders + env.binders, globals=env.globals)

    def _abstract_var(self, term: Term, target: int) -> Term:
        shifted = term.shift(1)

        def replace_var(t: Term, depth: int) -> Term:
            if isinstance(t, Var) and t.k == target + depth + 1:
                return Var(depth)
            return t._replace_terms(
                lambda sub, meta: replace_var(sub, depth + meta.binder_count)
            )

        return replace_var(shifted, 0)

    def _occurs(self, mid: int, term: Term) -> bool:
        if isinstance(term, MetaVar):
            if term.mid == mid:
                return True
            meta = self._state.metas.get(term.mid)
            if meta is not None and meta.solution is not None:
                return self._occurs(mid, meta.solution)
            return False
        found = False

        def check(sub: Term, _m: object) -> Term:
            nonlocal found
            if not found and self._occurs(mid, sub):
                found = True
            return sub

        term._replace_terms(check)
        return found

    def _adapt_to_ctx(self, term: Term, ctx_len: int, keep_len: int) -> Term:
        if ctx_len == keep_len:
            return term
        if ctx_len < keep_len:
            return term.shift(keep_len - ctx_len)
        drop = ctx_len - keep_len

        def restrict(t: Term, depth: int) -> Term:
            if isinstance(t, Var):
                if t.k < depth:
                    return t
                idx = t.k - depth
                if idx < drop:
                    raise ElabError(
                        "Cannot solve hole: solution mentions locals out of scope",
                        Span(0, 0),
                    )
                return Var(t.k - drop)
            return t._replace_terms(
                lambda sub, meta: restrict(sub, depth + meta.binder_count)
            )

        return restrict(term, 0)

    def _zonk_levels(self, term: Term) -> Term:
        match term:
            case Univ(level):
                return Univ(self.zonk_level(level))
            case UApp(head, levels):
                return UApp(head, tuple(self.zonk_level(level) for level in levels))
            case _:
                return term
