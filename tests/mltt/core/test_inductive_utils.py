from mltt.core.ast import App, Var
from mltt.core.inductive_utils import instantiate_params_indices


def test_instantiates_params_outermost_first() -> None:
    # Two parameters, no indices. Outermost param is Var(1), inner is Var(0).
    # When we substitute Var(11) for Var(0), all vars > 0 drop by one.
    params = (Var(10), Var(11))
    a = Var(1)
    b = Var(0)
    term = App(a, b)

    instantiated = instantiate_params_indices(term, params, ())

    a1 = Var(9)
    b1 = params[1]
    assert instantiated == App(a1, b1)


def test_instantiates_indices_inner_to_outer() -> None:
    # Two indices, no params. Outermost index is Var(1), inner is Var(0).
    # The second substitution removes Var(0) and shifts Var(20) down to Var(19).
    indices = (Var(20), Var(21))
    a = Var(1)
    b = Var(0)
    term = App(a, b)

    instantiated = instantiate_params_indices(term, (), indices)

    a1 = Var(19)
    b1 = indices[1]
    assert instantiated == App(a1, b1)


def test_instantiates_params_and_indices() -> None:
    # Binder order (outer -> inner): param0, param1, index0.
    params = (Var(30), Var(31))
    indices = (Var(32),)
    a = Var(1)
    b = Var(0)
    a2 = Var(2)
    b2 = App(a, b)
    term = App(a2, b2)

    instantiated = instantiate_params_indices(term, params, indices)

    # After all substitutions, free variables above each removed binder shift down.
    a1 = Var(30)
    b1 = indices[0]
    a3 = Var(28)
    b3 = App(a1, b1)
    assert instantiated == App(a3, b3)


def test_respects_offset_and_leaves_innermost_binders() -> None:
    # Offset=1 introduces an unrelated innermost binder at Var(0).
    # Binder order (outer -> inner): param0, index0, [offset binder].
    params = (Var(40),)
    indices = (Var(41),)
    a = Var(2)
    b = Var(0)
    term = App(a, b)

    instantiated = instantiate_params_indices(term, params, indices, offset=1)

    a1 = Var(39)
    b1 = Var(0)
    assert instantiated == App(a1, b1)
