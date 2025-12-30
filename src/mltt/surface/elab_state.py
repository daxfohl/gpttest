"""Metavariable and constraint state for elaboration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mltt.kernel.ast import App, Lam, MetaVar, Pi, Term, Univ, Var
from mltt.kernel.environment import Env
from mltt.kernel.ind import Elim
from mltt.surface.sast import Span, SurfaceError


@dataclass
class Meta:
    ctx_len: int
    ty: Term
    solution: Term | None = None
    span: Span | None = None
    kind: str = "hole"


@dataclass
class Constraint:
    ctx_len: int
    lhs: Term
    rhs: Term
    span: Span | None = None
    kind: str = "term_eq"


@dataclass
class ElabState:
    metas: dict[int, Meta] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    next_id: int = 0

    def fresh_meta(
        self, env: Env, expected: Term, span: Span | None, *, kind: str
    ) -> MetaVar:
        mid = self.next_id
        self.next_id += 1
        self.metas[mid] = Meta(
            ctx_len=len(env.binders), ty=expected, span=span, kind=kind
        )
        return MetaVar(mid)

    def add_constraint(self, env: Env, lhs: Term, rhs: Term, span: Span | None) -> None:
        self.constraints.append(Constraint(len(env.binders), lhs, rhs, span))

    def solve(self, env: Env) -> None:
        queue: deque[Constraint] = deque(self.constraints)
        postponed: list[Constraint] = []
        self.constraints = []
        while queue:
            constraint = queue.popleft()
            result = self._solve_constraint(env, constraint)
            if result == "solved":
                continue
            if result == "progress":
                queue.extendleft(postponed)
                postponed = []
                continue
            postponed.append(constraint)
            if not queue:
                if not postponed:
                    break
                queue.extend(postponed)
                postponed = []
                if all(self._is_stuck(env, c) for c in queue):
                    smallest = min(queue, key=lambda c: c.span.start if c.span else 0)
                    span = smallest.span or Span(0, 0)
                    lhs = self.zonk(smallest.lhs)
                    rhs = self.zonk(smallest.rhs)
                    raise SurfaceError(f"Stuck constraint: {lhs} ≡ {rhs}", span)

    def zonk(self, term: Term) -> Term:
        cache: dict[int, Term] = {}

        def walk(t: Term) -> Term:
            if isinstance(t, MetaVar):
                if t.mid in cache:
                    return cache[t.mid]
                meta = self.metas.get(t.mid)
                if meta is not None and meta.solution is not None:
                    cache[t.mid] = walk(meta.solution)
                    return cache[t.mid]
                return t
            return t._replace_terms(lambda sub, _m: walk(sub))

        return walk(term)

    def ensure_solved(self) -> None:
        unsolved = [
            (mid, meta) for mid, meta in self.metas.items() if meta.solution is None
        ]
        if not unsolved:
            return
        mid, meta = unsolved[0]
        ty = self.zonk(meta.ty)
        span = meta.span or Span(0, 0)
        if meta.kind == "implicit":
            message = f"Cannot infer implicit argument ?m{mid}; expected type {ty}"
        else:
            message = f"Cannot synthesize value for hole ?m{mid}; expected type {ty}"
        raise SurfaceError(message, span)

    def _solve_constraint(self, env: Env, constraint: Constraint) -> str:
        ctx_env = self._env_for_ctx(env, constraint.ctx_len)
        lhs = self.zonk(constraint.lhs).whnf(ctx_env)
        rhs = self.zonk(constraint.rhs).whnf(ctx_env)
        if lhs == rhs:
            return "solved"
        if self._solve_meta(ctx_env, constraint, lhs, rhs):
            return "progress"
        if type(lhs) is not type(rhs):
            raise SurfaceError(
                f"Cannot unify {lhs} with {rhs}", constraint.span or Span(0, 0)
            )
        if isinstance(lhs, Pi) and isinstance(rhs, Pi) and lhs.implicit == rhs.implicit:
            self.constraints.append(
                Constraint(constraint.ctx_len, lhs.arg_ty, rhs.arg_ty, constraint.span)
            )
            self.constraints.append(
                Constraint(
                    constraint.ctx_len + 1,
                    lhs.return_ty,
                    rhs.return_ty,
                    constraint.span,
                )
            )
            return "progress"
        if (
            isinstance(lhs, Lam)
            and isinstance(rhs, Lam)
            and lhs.implicit == rhs.implicit
        ):
            self.constraints.append(
                Constraint(constraint.ctx_len + 1, lhs.body, rhs.body, constraint.span)
            )
            return "progress"
        if (
            isinstance(lhs, App)
            and isinstance(rhs, App)
            and lhs.implicit == rhs.implicit
        ):
            self.constraints.append(
                Constraint(constraint.ctx_len, lhs.func, rhs.func, constraint.span)
            )
            self.constraints.append(
                Constraint(constraint.ctx_len, lhs.arg, rhs.arg, constraint.span)
            )
            return "progress"
        if isinstance(lhs, Univ) and isinstance(rhs, Univ) and lhs.level == rhs.level:
            return "solved"
        if (
            isinstance(lhs, Elim)
            and isinstance(rhs, Elim)
            and lhs.inductive == rhs.inductive
        ):
            self.constraints.append(
                Constraint(constraint.ctx_len, lhs.motive, rhs.motive, constraint.span)
            )
            self.constraints.append(
                Constraint(
                    constraint.ctx_len, lhs.scrutinee, rhs.scrutinee, constraint.span
                )
            )
            for l_case, r_case in zip(lhs.cases, rhs.cases, strict=True):
                self.constraints.append(
                    Constraint(constraint.ctx_len, l_case, r_case, constraint.span)
                )
            return "progress"
        raise SurfaceError(
            f"Cannot unify {lhs} with {rhs}", constraint.span or Span(0, 0)
        )

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

    def _try_solve_meta(self, env: Env, meta_term: MetaVar, rhs: Term) -> bool:
        meta = self.metas.get(meta_term.mid)
        if meta is None or meta.solution is not None:
            return False
        rhs = self.zonk(rhs)
        if self._occurs(meta_term.mid, rhs):
            raise SurfaceError(
                "Cannot solve hole: occurs check failed", meta.span or Span(0, 0)
            )
        adapted = self._adapt_to_ctx(rhs, len(env.binders), meta.ctx_len)
        meta.solution = adapted
        return True

    def _try_solve_spine(
        self,
        env: Env,
        meta_term: MetaVar,
        spine: list[tuple[Term, bool]],
        rhs: Term,
        constraint: Constraint,
    ) -> bool:
        meta = self.metas.get(meta_term.mid)
        if meta is None or meta.solution is not None:
            return False
        if not spine:
            return False
        var_indices: list[int] = []
        for arg, _implicit in spine:
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
        arg_tys, arg_impls = self._pi_spine(meta.ty, meta.ctx_len, len(spine), env)
        term = restricted_rhs
        for offset, (k, arg_ty, implicit) in enumerate(
            zip(reversed(adjusted_indices), reversed(arg_tys), reversed(arg_impls))
        ):
            term = self._abstract_var(term, k + offset)
            term = Lam(arg_ty, term, implicit=implicit)
        meta.solution = term
        return True

    def _pi_spine(
        self, ty: Term, ctx_len: int, count: int, env: Env
    ) -> tuple[list[Term], list[bool]]:
        arg_tys: list[Term] = []
        arg_impls: list[bool] = []
        current = ty
        ctx_env = self._env_for_ctx(env, ctx_len)
        for _ in range(count):
            current = self.zonk(current).whnf(ctx_env)
            if not isinstance(current, Pi):
                raise SurfaceError(
                    "Cannot solve meta: type is not a function", Span(0, 0)
                )
            arg_tys.append(current.arg_ty)
            arg_impls.append(current.implicit)
            current = current.return_ty
        return arg_tys, arg_impls

    def _is_stuck(self, env: Env, constraint: Constraint) -> bool:
        ctx_env = self._env_for_ctx(env, constraint.ctx_len)
        lhs = self.zonk(constraint.lhs).whnf(ctx_env)
        rhs = self.zonk(constraint.rhs).whnf(ctx_env)
        lhs_head, _ = self._decompose_app(lhs)
        rhs_head, _ = self._decompose_app(rhs)
        return isinstance(lhs_head, MetaVar) or isinstance(rhs_head, MetaVar)

    def _decompose_app(self, term: Term) -> tuple[Term, list[tuple[Term, bool]]]:
        spine: list[tuple[Term, bool]] = []
        head = term
        while isinstance(head, App):
            spine.append((head.arg, head.implicit))
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
            meta = self.metas.get(term.mid)
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
                    raise SurfaceError(
                        "Cannot solve hole: solution mentions locals out of scope",
                        Span(0, 0),
                    )
                return Var(t.k - drop)
            return t._replace_terms(
                lambda sub, meta: restrict(sub, depth + meta.binder_count)
            )

        return restrict(term, 0)
