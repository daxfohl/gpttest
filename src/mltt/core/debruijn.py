"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .ast import (
    App,
    Ctor,
    Id,
    IdElim,
    Elim,
    I,
    Lam,
    Pi,
    Refl,
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
    """Immutable context wrapper with convenient accessors."""

    entries: tuple[CtxEntry, ...] = ()

    def __iter__(self) -> Iterator[CtxEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> CtxEntry:
        return self.entries[idx]

    def extend(self, ty: Term) -> Ctx:
        """Extend ``ctx`` with ``ty`` while keeping indices for outer vars stable.

        The new binder lives at index 0; every stored type, including ``ty``, is
        shifted by one so references to outer binders keep pointing at the same
        definitions after widening the context.
        """

        prepended = (CtxEntry(ty), *self.entries)
        return Ctx.as_ctx(CtxEntry(shift(entry.ty, 1)) for entry in prepended)

    @staticmethod
    def as_ctx(ctx: Iterable[CtxEntry | Term]) -> Ctx:
        """Coerce a sequence of entries or terms into a ``Ctx`` of ``CtxEntry``."""
        return Ctx(tuple(Ctx.as_ctx_entry(entry) for entry in ctx))

    @staticmethod
    def as_ctx_entry(entry: CtxEntry | Term) -> CtxEntry:
        """Coerce an entry or term into a ``CtxEntry``."""

        return entry if isinstance(entry, CtxEntry) else CtxEntry(entry)


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
                [shift(branch, by, cutoff) for branch in cases],
                shift(scrutinee, by, cutoff),
            )
        case Id(ty, l, r):
            return Id(shift(ty, by, cutoff), shift(l, by, cutoff), shift(r, by, cutoff))
        case Refl(ty, t):
            return Refl(shift(ty, by, cutoff), shift(t, by, cutoff))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                shift(A, by, cutoff),
                shift(x, by, cutoff),
                shift(P, by, cutoff),
                shift(d, by, cutoff),
                shift(y, by, cutoff),
                shift(p, by, cutoff),
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
                [subst(branch, sub, j) for branch in cases],
                subst(scrutinee, sub, j),
            )
        case Id(ty, l, r):
            return Id(
                subst(ty, sub, j),
                subst(l, sub, j),
                subst(r, sub, j),
            )
        case Refl(ty, t):
            return Refl(subst(ty, sub, j), subst(t, sub, j))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                subst(A, sub, j),
                subst(x, sub, j),
                subst(P, sub, j),
                subst(d, sub, j),
                subst(y, sub, j),
                subst(p, sub, j),
            )
        case Univ() | I() | Ctor():
            return term

    raise TypeError(f"Unexpected term in subst: {term!r}")


__all__ = ["Ctx", "CtxEntry", "shift", "subst"]
