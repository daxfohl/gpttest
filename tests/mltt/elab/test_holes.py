import pytest

from mltt.elab.elab_helpers import elab_with_state
from mltt.elab.errors import ElabError
from mltt.solver.solver import Solver
from mltt.elab.term import elab_infer
from mltt.elab.types import ElabEnv
from mltt.kernel.prelude import prelude_env
from mltt.surface.parse import parse_elab_term


def test_typed_hole_unsolved() -> None:
    solver = elab_with_state("let x: Nat := (_: Nat); x")
    with pytest.raises(ElabError, match="Cannot synthesize value for hole"):
        solver.ensure_solved()


def test_hole_in_check_mode_lambda_body() -> None:
    solver = elab_with_state("let k(n: Nat): Nat := _; k")
    with pytest.raises(ElabError, match="Cannot synthesize value for hole"):
        solver.ensure_solved()


def test_reject_hole_in_infer_mode() -> None:
    env = ElabEnv.from_env(prelude_env())
    solver = Solver()
    term = parse_elab_term("_")
    with pytest.raises(ElabError, match="Hole needs expected type"):
        elab_infer(term, env, solver)
