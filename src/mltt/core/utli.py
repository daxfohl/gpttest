from __future__ import annotations

from mltt.core.ast import Term, Lam


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
