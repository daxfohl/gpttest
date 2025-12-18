"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .ast import Term


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
        Each stored entry type is scoped in the *tail context* beneath it. The
        type of Var(0) is stored in the context of Var(1..), the type of Var(1)
        is stored in the context of Var(2..), and so on.

    Extension discipline:
        `prepend(t)` prepends a new binder at index 0 without rewriting existing
        entry types. `prepend(t0, t1, ..., tn)` is equivalent to repeated
        single-binder extension:
            prepend(t0, t1, ..., tn) == prepend(t0).prepend(t1)...prepend(tn)
        where binder types are ordered outermost → innermost, matching the order
        of arguments to `nested_pi` / `nested_lam`.

    Interpretation:
        Given a context Γ and an index k, `Var(k)` refers to the binder whose
        type is stored at Γ[k]. Lookup shifts the stored type by k to account
        for the binders in front of it.

    Notes:
        This design keeps entry types relative to their tails and shifts on
        lookup, matching standard de Bruijn conventions for Pi/Lam codomains.
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
            The newest binder is at index 0. Stored entry types are not rewritten
            on extension; lookup shifts by index instead.
        """
        # prepend(t0, ..., tk) == prepend(t0).prepend(t1)...prepend(tk)
        # so insert from innermost to outermost (reverse order).
        return Ctx.as_ctx(*reversed(tys), *self.entries)

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

    def __str__(self) -> str:
        if len(self.entries) < 2:
            return f"Ctx{self.entries}"
        return f"Ctx(\n{"".join([f"  #{i}: {e.ty}\n" for i, e in enumerate(self)])})"


def shift(term: Term, by: int, cutoff: int = 0) -> Term:
    """Shift free variables in ``term`` by ``by`` starting at ``cutoff``."""

    return term.shift(by, cutoff)


def subst(term: Term, sub: Term, j: int = 0) -> Term:
    """Substitute ``sub`` for ``Var(j)`` inside ``term``."""

    return term.subst(sub, j)


__all__ = ["Ctx", "CtxEntry", "shift", "subst"]
