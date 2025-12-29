from __future__ import annotations

from dataclasses import dataclass

from mltt.kernel.ast import Term


@dataclass(frozen=True)
class Env:
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

    binders: tuple[Binder, ...] = ()

    def push_binders(self, *tys: Term) -> Env:
        """Prepend binders to the context.

        Args:
            tys: Binder types ordered outermost → innermost, like nested_pi/nested_lam.

        Semantics:
            prepend(t0, t1, ..., tk) == prepend(t0).prepend(t1)...prepend(tk)

        De Bruijn invariant:
            The newest binder is at index 0. Stored entry types are not rewritten
            on extension; lookup shifts by index instead.
        """
        # push(t0, ..., tk) == push(t0).push(t1)...push(tk)
        # so insert from innermost to outermost (reverse order).
        return Env.of(*reversed(tys), *self.binders)

    @staticmethod
    def of(*env: Binder | Term) -> Env:
        """Coerce a sequence of entries or terms into a ``Ctx`` of ``CtxEntry``."""
        return Env(tuple(Binder.of(entry) for entry in env))

    def __str__(self) -> str:
        if len(self.binders) < 2:
            return f"Ctx{self.binders}"
        return f"Ctx(\n{"".join([f"  #{i}: {e.ty}\n" for i, e in enumerate(self.binders)])})"


@dataclass(frozen=True)
class Binder:
    """Single context entry containing the type of a bound variable."""

    ty: Term

    @staticmethod
    def of(entry: Binder | Term) -> Binder:
        """Coerce an entry or term into a ``CtxEntry``."""

        return entry if isinstance(entry, Binder) else Binder(entry)
