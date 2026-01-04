"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload, Self, Sequence, Iterable, Callable

from mltt.kernel.ast import Term, App, Lam, Pi, Var, UApp
from mltt.kernel.levels import LevelExpr


@dataclass(frozen=True)
class SeqBase[T](Sequence[T]):
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


class Spine(SeqBase[Term]):
    @classmethod
    def of(cls, *items: Term) -> Self:
        return cls(items)

    @classmethod
    def empty(cls) -> Self:
        return cls.of()

    def __add__(self, other: Iterable[Term]) -> Self:
        return self.of(*self, *other)

    @staticmethod
    def vars(count: int, offset: int = 0) -> Spine:
        return Spine(tuple(Var(i) for i in reversed(range(offset, offset + count))))

    def instantiate(self, actuals: Spine, depth_above: int = 0) -> Self:
        return self._map(lambda t: t.instantiate(actuals, depth_above).whnf())

    def shift(self, i: int) -> Self:
        return self._map(lambda e: e.shift(i))

    def inst_levels(self, actuals: tuple[LevelExpr, ...]) -> Self:
        return self._map(lambda e: e.inst_levels(actuals))


class Telescope(SeqBase[Term]):
    @classmethod
    def of(cls, *items: Term) -> Self:
        return cls(items)

    @classmethod
    def empty(cls) -> Self:
        return cls.of()

    def __add__(self, other: Iterable[Term]) -> Self:
        return self.of(*self, *other)

    def instantiate(self, actuals: Spine, depth_above: int = 0) -> Self:
        return self._mapi(lambda i, t: t.instantiate(actuals, depth_above + i).whnf())

    def inst_levels(self, actuals: tuple[LevelExpr, ...]) -> Self:
        return self._map(lambda t: t.inst_levels(actuals))


def mk_app(fn: Term, *args: Term | Spine) -> Term:
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
        if isinstance(arg, Spine):
            result = mk_app(result, *arg)
        else:
            result = App(result, arg)
    return result


def mk_uapp(head: Term, levels: tuple[LevelExpr, ...], *args: Term | Spine) -> Term:
    """Apply universe levels to ``head`` and then apply term arguments."""
    applied = UApp(head, levels) if levels else head
    return mk_app(applied, *args)


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


def decompose_app(term: Term) -> tuple[Term, Spine]:
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
    return term, Spine.of(*reversed(args))


def decompose_uapp(term: Term) -> tuple[Term, tuple[LevelExpr, ...], Spine]:
    """Split an application into head, universe levels, and term arguments."""
    head, args = decompose_app(term)
    if isinstance(head, UApp):
        return head.head, head.levels, args
    return head, (), args
