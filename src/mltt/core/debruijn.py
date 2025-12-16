"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .ast import (
    App,
    Ctor,
    Elim,
    I,
    Lam,
    Pi,
    Term,
    Univ,
    Var,
)


@dataclass(frozen=True)
class CtxEntry:
    """Single context entry containing the type of a bound variable."""

    ty: Term


@dataclass(frozen=True)
class Ctx:
    """
    Typing context for de Bruijn-indexed terms.

    Representation:
        The context is stored as a sequence of entries, where index 0 refers to
        the *innermost (most recently introduced) binder*, index 1 to the next
        outer binder, and so on.

    Invariant:
        Every stored entry type is a well-scoped term in the *entire current
        context*. In particular, when a new binder is added, all existing entry
        types (and the new binder type itself) are shifted so that references to
        previously bound variables continue to point at the same syntactic
        entities after extension.

    Extension discipline:
        `prepend(t)` prepends a new binder at index 0 and shifts all stored types
        by 1. `prepend(t0, t1, ..., tn)` is equivalent to repeated single-binder
        extension:
            prepend(t0, t1, ..., tn) == prepend(t0).prepend(t1)...prepend(tn)
        where binder types are ordered outermost → innermost, matching the order
        of arguments to `nested_pi` / `nested_lam`.

    Interpretation:
        Given a context Γ and an index k, `Var(k)` refers to the binder whose
        type is stored at Γ[k]. The shifting performed during extension ensures
        that this interpretation is stable across context growth.

    Notes:
        This design chooses to eagerly maintain well-scoped entry types under
        extension, rather than interpreting entry types in a relative tail
        context. This simplifies lookup and type checking at the cost of extra
        shifting during context extension.
    """

    entries: tuple[CtxEntry, ...] = ()

    def __iter__(self) -> Iterator[CtxEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> CtxEntry:
        return self.entries[idx]

    def prepend_each(self, *tys: Term) -> Ctx:
        """Prepend binders to the context.

        Args:
            tys: Binder types ordered outermost → innermost, like nested_pi/nested_lam.

        Semantics:
            prepend(t0, t1, ..., tk) == prepend(t0).prepend(t1)...prepend(tk)

        De Bruijn invariant:
            The newest binder is at index 0. All stored entry types are maintained
            as well-scoped in the resulting (extended) context by shifting as needed.
        """

        k = len(tys)
        # Shift existing entries by k because k new binders are inserted in front
        existing = ((shift(entry.ty, k)) for entry in self)

        # Now compute the new entries exactly as repeated prepend would.
        # Insert from innermost to outermost (reverse order), and shift each ty by 1,
        # then 2, ... as it gets placed under previously inserted binders.
        tys1 = ((shift(ty, depth)) for depth, ty in enumerate(reversed(tys), start=1))

        # new_entries currently is [innermost shifted by1, ..., outermost shifted by k]
        # but those should appear *before* existing entries, and in the same order
        # as ctx.entries (index 0 is innermost), so keep as built:
        return Ctx.as_ctx(*tys1, *existing)

    @staticmethod
    def as_ctx(*ctx: CtxEntry | Term) -> Ctx:
        """Coerce a sequence of entries or terms into a ``Ctx`` of ``CtxEntry``."""
        return Ctx(tuple(Ctx.as_ctx_entry(entry) for entry in ctx))

    @staticmethod
    def as_ctx_entry(entry: CtxEntry | Term) -> CtxEntry:
        """Coerce an entry or term into a ``CtxEntry``."""

        return entry if isinstance(entry, CtxEntry) else CtxEntry(entry)

    @property
    def types(self) -> tuple[Term, ...]:
        return tuple(e.ty for e in self.entries)


def shift(term: Term, by: int, cutoff: int = 0) -> Term:
    """Shift free variables in ``term`` by ``by`` starting at ``cutoff``.

    De Bruijn convention: index 0 refers to the innermost binder. When adding
    a binder, shift outer references up to keep them pointing at the same
    syntactic entity. ``cutoff`` shields inner binders so they are unaffected.
    """

    match term:
        case Var(k):
            return Var(k + by if k >= cutoff else k)
        case Lam(ty, body):
            return Lam(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case Pi(ty, body):
            return Pi(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case App(f, a):
            return App(shift(f, by, cutoff), shift(a, by, cutoff))
        case Elim(inductive, motive, cases, scrutinee):
            return Elim(
                inductive,
                shift(motive, by, cutoff),
                tuple(shift(case, by, cutoff) for case in cases),
                shift(scrutinee, by, cutoff),
            )
        case Univ() | I() | Ctor():
            return term

    raise TypeError(f"Unexpected term in shift: {term!r}")


def subst(term: Term, sub: Term, j: int = 0) -> Term:
    """Substitute ``sub`` for ``Var(j)`` inside ``term``, and squash it.

    Standard de Bruijn substitution: replacing ``Var(j)`` drops indices above
    ``j`` by 1 to fill the gap and shifts ``sub`` when descending under binders
    so its free variables remain aligned.
    """
    match term:
        case Var(k):
            if k == j:
                return sub
            elif k > j:
                return Var(k - 1)
            else:
                return term
        case Lam(ty, body):
            return Lam(
                subst(ty, sub, j),
                subst(body, shift(sub, 1, 0), j + 1),
            )
        case Pi(ty, body):
            return Pi(
                subst(ty, sub, j),
                subst(body, shift(sub, 1, 0), j + 1),
            )
        case App(f, a):
            return App(subst(f, sub, j), subst(a, sub, j))
        case Elim(inductive, motive, cases, scrutinee):
            return Elim(
                inductive,
                subst(motive, sub, j),
                tuple(subst(case, sub, j) for case in cases),
                subst(scrutinee, sub, j),
            )
        case Univ() | I() | Ctor():
            return term

    raise TypeError(f"Unexpected term in subst: {term!r}")


__all__ = ["Ctx", "CtxEntry", "shift", "subst"]
