import pytest

from mltt.elab.etype import ElabEnv
from mltt.elab.elab_state import ElabState
from mltt.elab.sast import elab_infer
from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError
from mltt.kernel.prelude import prelude_env
from elab_helpers import elab_with_state


def test_typed_hole_unsolved() -> None:
    state = elab_with_state("let x: Nat := (_: Nat); x")
    with pytest.raises(SurfaceError, match="Cannot synthesize value for hole"):
        state.ensure_solved()


def test_hole_in_check_mode_lambda_body() -> None:
    state = elab_with_state("let k(n: Nat): Nat := _; k")
    with pytest.raises(SurfaceError, match="Cannot synthesize value for hole"):
        state.ensure_solved()


def test_reject_hole_in_infer_mode() -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term("_")
    with pytest.raises(SurfaceError, match="Hole needs expected type"):
        elab_infer(term, env, state)
