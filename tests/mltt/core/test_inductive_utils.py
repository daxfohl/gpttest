from mltt.core.ast import App, I, Var
from mltt.core.debruijn import shift
from mltt.core.inductive_utils import (
    instantiate_for_inductive,
    instantiate_params_indices,
    instantiate_into,
)


def test_instantiates_params_outermost_first() -> None:
    # Two parameters, no indices. Outermost param is Var(1), inner is Var(0).
    # When we substitute Var(11) for Var(0), all vars > 0 drop by one.
    params = (Var(10), Var(11))
    term = App(Var(1), Var(0))

    instantiated = instantiate_params_indices(term, params, ())

    assert instantiated == App(Var(9), params[1])


def test_instantiates_indices_inner_to_outer() -> None:
    # Two indices, no params. Outermost index is Var(1), inner is Var(0).
    # The second substitution removes Var(0) and shifts Var(20) down to Var(19).
    indices = (Var(20), Var(21))
    term = App(Var(1), Var(0))

    instantiated = instantiate_params_indices(term, (), indices)

    assert instantiated == App(Var(19), indices[1])


def test_instantiates_params_and_indices() -> None:
    # Binder order (outer -> inner): param0, param1, index0.
    params = (Var(30), Var(31))
    indices = (Var(32),)
    term = App(Var(2), App(Var(1), Var(0)))

    instantiated = instantiate_params_indices(term, params, indices)

    # After all substitutions, free variables above each removed binder shift down.
    assert instantiated == App(Var(28), App(Var(30), indices[0]))


def test_respects_offset_and_leaves_innermost_binders() -> None:
    # Offset=1 introduces an unrelated innermost binder at Var(0).
    # Binder order (outer -> inner): param0, index0, [offset binder].
    params = (Var(40),)
    indices = (Var(41),)
    term = App(Var(2), Var(0))

    instantiated = instantiate_params_indices(term, params, indices, offset=1)

    assert instantiated == App(Var(39), Var(0))


def test_instantiate_for_inductive_matches_shifted_pattern() -> None:
    ind = I(name="Dummy", param_types=(Var(0),), index_types=(Var(0), Var(0)))
    params = (Var(10),)
    indices = (Var(20), Var(21))
    args = (Var(30),)
    targets = (App(App(Var(3), Var(1)), Var(0)),)

    shifted = tuple(shift(arg, len(ind.index_types)) for arg in (*params, *indices))
    expected = instantiate_into((*shifted, *args), targets)

    assert (
        instantiate_for_inductive(ind, params, indices, targets, args=args) == expected
    )
