from __future__ import annotations

from mltt.core.ast import Term, App, Pi, Lam


def apply_term(term: Term, *args: Term) -> Term:
    """Apply ``args`` to ``term`` left-associatively.

    Constructors and inductive type heads are stored unapplied; callers often
    need to thread parameters, indices, and payloads in order. This helper
    keeps those call sites readable and centralizes the left-associative
    application pattern.

    Args:
        term: Function being applied.
        *args: Arguments to apply, ordered left-to-right.

    Returns:
        The left-associated application ``(((term arg0) arg1) ...)``.
    """
    #  e.g. term = \x->(\y->z). args = [x, y]
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def nested_pi(*param_tys: Term, return_ty: Term) -> Term:
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
        pi = Pi(param_ty, pi)
    return pi


def decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
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
    return term, tuple(reversed(args))


def decompose_lam(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    while isinstance(term, Lam):
        args.append(term.arg_ty)
        term = term.body
    return term, tuple(args)


def decompose_pi(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    while isinstance(term, Pi):
        args.append(term.arg_ty)
        term = term.return_ty
    return term, tuple(args)


def nested_lam(*param_tys: Term, body: Term) -> Term:
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
        fn = Lam(param_ty, fn)
    return fn
