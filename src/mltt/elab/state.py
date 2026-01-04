"""Metavariable and constraint state for elaboration."""

from __future__ import annotations

from dataclasses import dataclass, field

from mltt.common.span import Span
from mltt.kernel.ast import MetaVar, Term, Univ, UApp
from mltt.kernel.env import Env
from mltt.kernel.ind import Ind
from mltt.kernel.levels import LConst, LMax, LMeta, LSucc, LVar, LevelExpr


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
class LMetaInfo:
    solution: LevelExpr | None = None
    span: Span | None = None
    origin: str = "type"
    lower_bound: LevelExpr = field(default_factory=lambda: LConst(0))
    upper_bounds: list[LevelExpr] = field(default_factory=list)


@dataclass
class LevelConstraint:
    lhs: LevelExpr
    rhs: LevelExpr
    span: Span | None = None
    reason: str | None = None


@dataclass
class ElabState:
    metas: dict[int, Meta] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    next_id: int = 0
    level_metas: dict[int, LMetaInfo] = field(default_factory=dict)
    level_constraints: list[LevelConstraint] = field(default_factory=list)
    next_level_id: int = 0

    def fresh_meta(
        self, env: Env, expected: Term, span: Span | None, *, kind: str
    ) -> MetaVar:
        mid = self.next_id
        self.next_id += 1
        self.metas[mid] = Meta(
            ctx_len=len(env.binders), ty=expected, span=span, kind=kind
        )
        return MetaVar(mid)

    def fresh_level_meta(self, origin: str, span: Span | None) -> LMeta:
        mid = self.next_level_id
        self.next_level_id += 1
        self.level_metas[mid] = LMetaInfo(span=span, origin=origin)
        return LMeta(mid)

    def apply_implicit_levels(
        self, head: Term, uarity: int, span: Span
    ) -> tuple[Term, tuple[LevelExpr, ...]]:
        if uarity <= 0:
            return head, ()
        levels = tuple(self.fresh_level_meta("implicit", span) for _ in range(uarity))
        return UApp(head, levels), levels

    def add_constraint(self, env: Env, lhs: Term, rhs: Term, span: Span | None) -> None:
        constraint = Constraint(len(env.binders), lhs, rhs, span)
        from mltt.elab import solver as elab_solver

        if elab_solver.solve_meta(self, env, constraint, lhs, rhs):
            return
        self.constraints.append(constraint)

    def add_level_constraint(
        self,
        lhs: LevelExpr,
        rhs: LevelExpr,
        span: Span | None,
        reason: str | None = None,
    ) -> None:
        self.level_constraints.append(LevelConstraint(lhs, rhs, span, reason))

    def generalize_let(self, ty: Term, value: Term) -> tuple[int, Term, Term]:
        meta_ids = self._collect_level_metas(ty) | self._collect_level_metas(value)
        if not meta_ids:
            return 0, ty, value
        equalities = self._level_meta_equalities()
        parent: dict[int, int] = {mid: mid for mid in meta_ids}

        def find(mid: int) -> int:
            while parent[mid] != mid:
                parent[mid] = parent[parent[mid]]
                mid = parent[mid]
            return mid

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in equalities:
            if a in parent and b in parent:
                union(a, b)

        groups: dict[int, list[int]] = {}
        for mid in meta_ids:
            groups.setdefault(find(mid), []).append(mid)

        to_generalize: list[list[int]] = []
        for members in groups.values():
            infos = [self.level_metas.get(mid) for mid in members]
            if any(info is None or info.solution is not None for info in infos):
                continue
            if any(info.upper_bounds for info in infos if info is not None):
                continue
            if any(info.lower_bound != LConst(0) for info in infos if info is not None):
                continue
            to_generalize.append(sorted(members))

        if not to_generalize:
            return 0, ty, value

        mapping: dict[int, LevelExpr] = {}
        for idx, group in enumerate(to_generalize):
            for mid in group:
                mapping[mid] = LVar(idx)
        ty_gen = self._replace_level_metas(ty, mapping)
        value_gen = self._replace_level_metas(value, mapping)
        if mapping:
            self.constraints = [
                Constraint(
                    c.ctx_len,
                    self._replace_level_metas(c.lhs, mapping),
                    self._replace_level_metas(c.rhs, mapping),
                    c.span,
                    c.kind,
                )
                for c in self.constraints
            ]
            self.level_constraints = [
                LevelConstraint(
                    self._replace_level_expr(c.lhs, mapping),
                    self._replace_level_expr(c.rhs, mapping),
                    c.span,
                    c.reason,
                )
                for c in self.level_constraints
            ]
        for group in to_generalize:
            for mid in group:
                self.level_metas.pop(mid, None)
        return len(to_generalize), ty_gen, value_gen

    def merge_type_level_metas(self, terms: list[Term]) -> list[Term]:
        meta_ids: list[int] = []
        for term in terms:
            meta_ids.extend(self._collect_level_metas(term))
        seen: set[int] = set()
        type_metas: list[int] = []
        for mid in meta_ids:
            if mid in seen:
                continue
            info = self.level_metas.get(mid)
            if info is None or info.solution is not None or info.origin != "type":
                continue
            seen.add(mid)
            type_metas.append(mid)
        if len(type_metas) <= 1:
            return terms
        root = type_metas[0]
        mapping: dict[int, LevelExpr] = {mid: LMeta(root) for mid in type_metas[1:]}
        terms = [self._replace_level_metas(term, mapping) for term in terms]
        if mapping:
            self.constraints = [
                Constraint(
                    c.ctx_len,
                    self._replace_level_metas(c.lhs, mapping),
                    self._replace_level_metas(c.rhs, mapping),
                    c.span,
                    c.kind,
                )
                for c in self.constraints
            ]
            self.level_constraints = [
                LevelConstraint(
                    self._replace_level_expr(c.lhs, mapping),
                    self._replace_level_expr(c.rhs, mapping),
                    c.span,
                    c.reason,
                )
                for c in self.level_constraints
            ]
        for mid in type_metas[1:]:
            self.level_metas.pop(mid, None)
        return terms

    def generalize_levels(self, terms: list[Term]) -> tuple[int, list[Term]]:
        meta_ids: set[int] = set()
        for term in terms:
            meta_ids |= self._collect_level_metas(term)
        if not meta_ids:
            return 0, terms
        equalities = self._level_meta_equalities()
        parent: dict[int, int] = {mid: mid for mid in meta_ids}

        def find(mid: int) -> int:
            while parent[mid] != mid:
                parent[mid] = parent[parent[mid]]
                mid = parent[mid]
            return mid

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in equalities:
            if a in parent and b in parent:
                union(a, b)

        groups: dict[int, list[int]] = {}
        for mid in meta_ids:
            groups.setdefault(find(mid), []).append(mid)

        to_generalize: list[list[int]] = []
        for members in groups.values():
            infos = [self.level_metas.get(mid) for mid in members]
            if any(info is None or info.solution is not None for info in infos):
                continue
            if any(info.upper_bounds for info in infos if info is not None):
                continue
            if any(info.lower_bound != LConst(0) for info in infos if info is not None):
                continue
            to_generalize.append(sorted(members))

        if not to_generalize:
            return 0, terms

        mapping: dict[int, LevelExpr] = {}
        for idx, group in enumerate(to_generalize):
            for mid in group:
                mapping[mid] = LVar(idx)
        terms_gen = [self._replace_level_metas(term, mapping) for term in terms]
        if mapping:
            self.constraints = [
                Constraint(
                    c.ctx_len,
                    self._replace_level_metas(c.lhs, mapping),
                    self._replace_level_metas(c.rhs, mapping),
                    c.span,
                    c.kind,
                )
                for c in self.constraints
            ]
            self.level_constraints = [
                LevelConstraint(
                    self._replace_level_expr(c.lhs, mapping),
                    self._replace_level_expr(c.rhs, mapping),
                    c.span,
                    c.reason,
                )
                for c in self.level_constraints
            ]
        for group in to_generalize:
            for mid in group:
                self.level_metas.pop(mid, None)
        return len(to_generalize), terms_gen

    def solve(self, env: Env) -> None:
        from mltt.elab import solver as elab_solver

        elab_solver.solve(self, env)

    def zonk(self, term: Term) -> Term:
        from mltt.elab import solver as elab_solver

        return elab_solver.zonk(self, term)

    def zonk_level(self, level: LevelExpr) -> LevelExpr:
        from mltt.elab import solver as elab_solver

        return elab_solver.zonk_level(self, level)

    def ensure_solved(self) -> None:
        from mltt.elab import solver as elab_solver

        elab_solver.ensure_solved(self)

    def _collect_level_metas(self, term: Term) -> set[int]:
        found: set[int] = set()

        def collect_level(level: LevelExpr) -> None:
            match level:
                case LMeta(mid):
                    found.add(mid)
                case LSucc(e):
                    collect_level(e)
                case LMax(a, b):
                    collect_level(a)
                    collect_level(b)
                case _:
                    return

        def walk(t: Term) -> None:
            match t:
                case Univ(level):
                    collect_level(level)
                case UApp(head, levels):
                    walk(head)
                    for level in levels:
                        collect_level(level)
                case Ind(level=level):
                    collect_level(level)
                case _:

                    def visit(sub: Term, _m: object) -> Term:
                        walk(sub)
                        return sub

                    _ = t._replace_terms(visit)
                    return None

        walk(term)
        return found

    def _level_meta_equalities(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()

        def meta_id(term: Term) -> int | None:
            if isinstance(term, Univ) and isinstance(term.level, LMeta):
                return term.level.mid
            return None

        for constraint in self.constraints:
            lhs_id = meta_id(constraint.lhs)
            rhs_id = meta_id(constraint.rhs)
            if lhs_id is not None and rhs_id is not None and lhs_id != rhs_id:
                a, b = sorted((lhs_id, rhs_id))
                pairs.add((a, b))
        return pairs

    def _replace_level_metas(self, term: Term, mapping: dict[int, LevelExpr]) -> Term:
        def replace_level(level: LevelExpr) -> LevelExpr:
            match level:
                case LMeta(mid) if mid in mapping:
                    return mapping[mid]
                case LSucc(e):
                    return LSucc(replace_level(e))
                case LMax(a, b):
                    return LMax(replace_level(a), replace_level(b))
                case _:
                    return level

        def walk(t: Term) -> Term:
            match t:
                case Univ(level):
                    return Univ(replace_level(level))
                case UApp(head, levels):
                    head_term = walk(head)
                    new_levels = tuple(replace_level(level) for level in levels)
                    return UApp(head_term, new_levels)
                case Ind(level=level):
                    return t
                case _:
                    return t._replace_terms(lambda sub, _m: walk(sub))

        return walk(term)

    def _replace_level_expr(
        self, level: LevelExpr, mapping: dict[int, LevelExpr]
    ) -> LevelExpr:
        match level:
            case LMeta(mid) if mid in mapping:
                return mapping[mid]
            case LSucc(e):
                return LSucc(self._replace_level_expr(e, mapping))
            case LMax(a, b):
                return LMax(
                    self._replace_level_expr(a, mapping),
                    self._replace_level_expr(b, mapping),
                )
            case _:
                return level
