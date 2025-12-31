import pytest

from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError
from mltt.surface.etype import ElabEnv
from mltt.surface.prelude import prelude_env


def elab_with_state(src: str) -> ElabState:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term(src)
    term.elab_infer(env, state)
    state.solve(env.kenv)
    return state


def test_typed_hole_unsolved() -> None:
    state = elab_with_state("let x : Nat := (_ : Nat); x")
    with pytest.raises(SurfaceError, match="Cannot synthesize value for hole"):
        state.ensure_solved()


def test_hole_in_check_mode_lambda_body() -> None:
    state = elab_with_state("let k : Nat -> Nat := fun n => _; k")
    with pytest.raises(SurfaceError, match="Cannot synthesize value for hole"):
        state.ensure_solved()


def test_reject_hole_in_infer_mode() -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term("_")
    with pytest.raises(SurfaceError, match="Hole needs expected type"):
        term.elab_infer(env, state)
