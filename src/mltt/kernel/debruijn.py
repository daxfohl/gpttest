"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar, overload, Self, Sequence, Iterable, Callable

from mltt.kernel.ast import Term, App, Lam, Pi, Var

T = TypeVar("T")


@dataclass(frozen=True)
class SeqBase(Sequence[T], Generic[T]):
    _data: tuple[T, ...] = ()

    @classmethod
    def of(cls: type[Self], *items: T) -> Self:
        return cls(items)

    # ---- Sequence contract ----
    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, i: int) -> T: ...
    @overload
    def __getitem__(self, s: slice) -> Self: ...

    def __getitem__(self, idx: int | slice) -> T | Self:
        if isinstance(idx, slice):
            return self.of(*self._data[idx])
        return self._data[idx]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._data)!r})"

    # ---- a couple generic helpers you might want everywhere ----
    def _map(self, f: Callable[[T], T]) -> Self:
        return self.of(*(f(x) for x in self._data))

    def _mapi(self, f: Callable[[int, T], T]) -> Self:
        return self.of(*(f(i, x) for i, x in enumerate(self._data)))


class ArgList(SeqBase[Term]):
    @classmethod
    def of(cls, *items: Term) -> Self:
        return cls(items)

    @classmethod
    def empty(cls) -> Self:
        return cls.of()

    def __add__(self, other: Iterable[Term]) -> Self:
        return self.of(*self, *other)

    @staticmethod
    def vars(count: int, offset: int = 0) -> ArgList:
        return ArgList(tuple(Var(i) for i in reversed(range(offset, offset + count))))

    def instantiate(self, actuals: ArgList, depth_above: int = 0) -> Self:
        return self._map(lambda t: t.instantiate(actuals, depth_above).whnf())

    def shift(self, i: int) -> Self:
        return self._map(lambda e: e.shift(i))


class Telescope(SeqBase[Term]):
    @classmethod
    def of(cls, *items: Term) -> Self:
        return cls(items)

    @classmethod
    def empty(cls) -> Self:
        return cls.of()

    def __add__(self, other: Iterable[Term]) -> Self:
        return self.of(*self, *other)

    def instantiate(self, actuals: ArgList, depth_above: int = 0) -> Self:
        return self._mapi(lambda i, t: t.instantiate(actuals, depth_above + i).whnf())


@dataclass(frozen=True)
class CtxEntry:
    """Single context entry containing the type of a bound variable."""

    ty: Term

    @staticmethod
    def of(entry: CtxEntry | Term) -> CtxEntry:
        """Coerce an entry or term into a ``CtxEntry``."""

        return entry if isinstance(entry, CtxEntry) else CtxEntry(entry)


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

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CtxEntry]:
        return iter(self.entries)

    def __getitem__(self, idx: int) -> CtxEntry:
        return self.entries[idx]

    def push(self, *tys: Term) -> Ctx:
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
        return Ctx.of(*reversed(tys), *self.entries)

    @staticmethod
    def of(*ctx: CtxEntry | Term) -> Ctx:
        """Coerce a sequence of entries or terms into a ``Ctx`` of ``CtxEntry``."""
        return Ctx(tuple(CtxEntry.of(entry) for entry in ctx))

    def __str__(self) -> str:
        if len(self.entries) < 2:
            return f"Ctx{self.entries}"
        return f"Ctx(\n{"".join([f"  #{i}: {e.ty}\n" for i, e in enumerate(self)])})"

    def as_telescope(self) -> Telescope:
        return Telescope.of(*(e.ty for e in reversed(self.entries)))


def mk_app(fn: Term, *args: Term | ArgList) -> Term:
    """Apply ``args`` to ``term`` left-associatively.

    Constructors and inductive type heads are stored unapplied; callers often
    need to thread parameters, indices, and payloads in order. This helper
    keeps those call sites readable and centralizes the left-associative
    application pattern.

    Args:
        fn: Function being applied.
        *args: Arguments to apply, ordered left-to-right.

    Returns:
        The left-associated application ``(((term arg0) arg1) ...)``.
    """
    #  e.g. fn = \x->(\y->z). args = [x, y]
    result: Term = fn
    for arg in args:
        if isinstance(arg, ArgList):
            result = mk_app(result, *arg)
        else:
            result = App(result, arg)
    return result


def mk_lams(*param_tys: Term | Telescope, body: Term) -> Term:
    """Build a right-nested lambda chain over ``param_tys`` ending in ``body``.

    Each element of ``param_tys`` becomes one binder, with the first argument
    binding outermost and the last argument closest to ``body``. This mirrors
    the left-to-right order of parameters in source syntax while keeping the
    resulting AST compact and easy to construct programmatically.

    Args:
        *param_tys: Parameter types, ordered from outermost to innermost.
        body: The lambda body that closes over the introduced binders.

    Returns:
        A ``Lam`` chain whose body is ``body`` and whose binders match
        ``param_tys`` in order.
    """
    fn: Term = body
    for param_ty in reversed(param_tys):
        if isinstance(param_ty, Telescope):
            fn = mk_lams(*param_ty, body=fn)
        else:
            fn = Lam(param_ty, fn)
    return fn


def mk_pis(*param_tys: Term | Telescope, return_ty: Term) -> Term:
    """Build a right-nested Pi chain over ``param_tys`` ending in ``return_ty``.

    Like ``nested_lam``, the outermost quantifier corresponds to the first
    element of ``param_tys`` while the last element binds closest to
    ``return_ty``. This helper centralizes the repetitive Pi-tower pattern
    used throughout the inductive definitions.

    Args:
        *param_tys: Parameter types, ordered from outermost to innermost.
        return_ty: The codomain of the innermost Pi.

    Returns:
        A ``Pi`` chain whose codomain is ``return_ty`` and whose binders match
        ``param_tys`` in order.
    """
    pi: Term = return_ty
    for param_ty in reversed(param_tys):
        if isinstance(param_ty, Telescope):
            pi = mk_pis(*param_ty, return_ty=pi)
        else:
            pi = Pi(param_ty, pi)
    return pi


def decompose_app(term: Term) -> tuple[Term, ArgList]:
    """Split an application into its head and argument tuple.

    This is the inverse of ``apply_term`` and is used by eliminator matching.
    It peels applications from the outside in, yielding the ultimate head
    (which may itself be an inductive type or constructor) and the ordered
    argument tuple.

    Args:
        term: Term to break apart. Non-application terms return themselves as
            the head with an empty argument tuple.

    Returns:
        A pair ``(head, args)`` where ``head`` is the unapplied function and
        ``args`` are in the same order they were originally applied.
    """
    #  e.g. input = ((((\x->(\y->z)) x) y). output: [\x->(\y->z), [x, y]]
    args: list[Term] = []
    while isinstance(term, App):
        args.append(term.arg)
        term = term.func
    return term, ArgList.of(*reversed(args))
