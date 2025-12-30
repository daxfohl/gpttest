"""Metavariable and constraint state for Milestone 2 elaboration."""

from __future__ import annotations

from dataclasses import dataclass, field

from mltt.kernel.ast import App, MetaVar, Pi, Term, Var
from mltt.kernel.environment import Env
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
        changed = True
        while changed:
            changed = False
            remaining: list[Constraint] = []
            for constraint in self.constraints:
                if self._solve_constraint(env, constraint):
                    changed = True
                else:
                    remaining.append(constraint)
            self.constraints = remaining

    def zonk(self, term: Term) -> Term:
        if isinstance(term, MetaVar):
            meta = self.metas.get(term.mid)
            if meta is not None and meta.solution is not None:
                return self.zonk(meta.solution)
            return term
        return term._replace_terms(lambda t, _m: self.zonk(t))

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

    def _solve_constraint(self, env: Env, constraint: Constraint) -> bool:
        lhs = self.zonk(constraint.lhs)
        rhs = self.zonk(constraint.rhs)
        if lhs == rhs:
            return True
        if isinstance(lhs, MetaVar) and self._try_solve(lhs.mid, rhs):
            return True
        if isinstance(rhs, MetaVar) and self._try_solve(rhs.mid, lhs):
            return True
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
            return True
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
            return True
        return False

    def _try_solve(self, mid: int, rhs: Term) -> bool:
        meta = self.metas.get(mid)
        if meta is None or meta.solution is not None:
            return False
        rhs = self.zonk(rhs)
        if self._occurs(mid, rhs):
            raise SurfaceError(
                "Cannot solve hole: occurs check failed", meta.span or Span(0, 0)
            )
        max_var = self._max_var_index(rhs)
        if max_var >= meta.ctx_len:
            raise SurfaceError(
                "Cannot solve hole: solution mentions locals out of scope",
                meta.span or Span(0, 0),
            )
        meta.solution = rhs
        return True

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

    def _max_var_index(self, term: Term) -> int:
        if isinstance(term, Var):
            return term.k
        if isinstance(term, MetaVar):
            meta = self.metas.get(term.mid)
            if meta is not None and meta.solution is not None:
                return self._max_var_index(meta.solution)
            return -1
        max_idx = -1

        def check(sub: Term, _m: object) -> Term:
            nonlocal max_idx
            max_idx = max(max_idx, self._max_var_index(sub))
            return sub

        term._replace_terms(check)
        return max_idx
