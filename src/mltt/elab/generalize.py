"""Elaboration-time generalization policy for universe levels."""

from __future__ import annotations

from mltt.kernel.ast import Term, Univ, UApp
from mltt.kernel.ind import Ind
from mltt.kernel.levels import LConst, LMax, LMeta, LSucc, LVar, LevelExpr
from mltt.solver.constraints import Constraint
from mltt.solver.levels import LevelConstraint
from mltt.solver.solver import Solver


def generalize_let(solver: Solver, ty: Term, value: Term) -> tuple[int, Term, Term]:
    meta_ids = _collect_level_metas(ty) | _collect_level_metas(value)
    if not meta_ids:
        return 0, ty, value
    equalities = _level_meta_equalities(solver)
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
        infos = [solver.level_metas.get(mid) for mid in members]
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
    ty_gen = _replace_level_metas(ty, mapping)
    value_gen = _replace_level_metas(value, mapping)
    if mapping:
        solver.constraints = [
            Constraint(
                c.ctx_len,
                _replace_level_metas(c.lhs, mapping),
                _replace_level_metas(c.rhs, mapping),
                c.span,
                c.kind,
            )
            for c in solver.constraints
        ]
        solver.level_constraints = [
            LevelConstraint(
                _replace_level_expr(c.lhs, mapping),
                _replace_level_expr(c.rhs, mapping),
                c.span,
                c.reason,
            )
            for c in solver.level_constraints
        ]
    for group in to_generalize:
        for mid in group:
            solver.level_metas.pop(mid, None)
    return len(to_generalize), ty_gen, value_gen


def merge_type_level_metas(solver: Solver, terms: list[Term]) -> list[Term]:
    meta_ids: list[int] = []
    for term in terms:
        meta_ids.extend(_collect_level_metas(term))
    seen: set[int] = set()
    type_metas: list[int] = []
    for mid in meta_ids:
        if mid in seen:
            continue
        info = solver.level_metas.get(mid)
        if info is None or info.solution is not None or info.origin != "type":
            continue
        seen.add(mid)
        type_metas.append(mid)
    if len(type_metas) <= 1:
        return terms
    root = type_metas[0]
    mapping: dict[int, LevelExpr] = {mid: LMeta(root) for mid in type_metas[1:]}
    terms = [_replace_level_metas(term, mapping) for term in terms]
    if mapping:
        solver.constraints = [
            Constraint(
                c.ctx_len,
                _replace_level_metas(c.lhs, mapping),
                _replace_level_metas(c.rhs, mapping),
                c.span,
                c.kind,
            )
            for c in solver.constraints
        ]
        solver.level_constraints = [
            LevelConstraint(
                _replace_level_expr(c.lhs, mapping),
                _replace_level_expr(c.rhs, mapping),
                c.span,
                c.reason,
            )
            for c in solver.level_constraints
        ]
    for mid in type_metas[1:]:
        solver.level_metas.pop(mid, None)
    return terms


def generalize_levels(solver: Solver, terms: list[Term]) -> tuple[int, list[Term]]:
    meta_ids: set[int] = set()
    for term in terms:
        meta_ids |= _collect_level_metas(term)
    if not meta_ids:
        return 0, terms
    equalities = _level_meta_equalities(solver)
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
        infos = [solver.level_metas.get(mid) for mid in members]
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
    terms_gen = [_replace_level_metas(term, mapping) for term in terms]
    if mapping:
        solver.constraints = [
            Constraint(
                c.ctx_len,
                _replace_level_metas(c.lhs, mapping),
                _replace_level_metas(c.rhs, mapping),
                c.span,
                c.kind,
            )
            for c in solver.constraints
        ]
        solver.level_constraints = [
            LevelConstraint(
                _replace_level_expr(c.lhs, mapping),
                _replace_level_expr(c.rhs, mapping),
                c.span,
                c.reason,
            )
            for c in solver.level_constraints
        ]
    for group in to_generalize:
        for mid in group:
            solver.level_metas.pop(mid, None)
    return len(to_generalize), terms_gen


def _collect_level_metas(term: Term) -> set[int]:
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
                _ = level
                collect_level(level)
            case _:

                def visit(sub: Term, _m: object) -> Term:
                    walk(sub)
                    return sub

                _ = t._replace_terms(visit)
                return None

    walk(term)
    return found


def _level_meta_equalities(solver: Solver) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()

    def meta_id(term: Term) -> int | None:
        match term:
            case Univ(level=LMeta(mid)):
                return mid
            case _:
                return None

    for constraint in solver.constraints:
        lhs_id = meta_id(constraint.lhs)
        rhs_id = meta_id(constraint.rhs)
        if lhs_id is not None and rhs_id is not None and lhs_id != rhs_id:
            a, b = sorted((lhs_id, rhs_id))
            pairs.add((a, b))
    return pairs


def _replace_level_metas(term: Term, mapping: dict[int, LevelExpr]) -> Term:
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
                _ = level
                return t
            case _:
                return t._replace_terms(lambda sub, _m: walk(sub))

    return walk(term)


def _replace_level_expr(level: LevelExpr, mapping: dict[int, LevelExpr]) -> LevelExpr:
    match level:
        case LMeta(mid) if mid in mapping:
            return mapping[mid]
        case LSucc(e):
            return LSucc(_replace_level_expr(e, mapping))
        case LMax(a, b):
            return LMax(
                _replace_level_expr(a, mapping),
                _replace_level_expr(b, mapping),
            )
        case _:
            return level
