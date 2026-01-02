import pytest

from mltt.surface.elab_state import ElabState
from mltt.surface.etype import ElabEnv
from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError
from mltt.surface.prelude import prelude_env


def elab_ok(src: str) -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = term.elab_infer(env, state)
    state.solve(env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def test_explicit_id() -> None:
    src = """
    let id(A: Type 0, x: A): A := x;
    id
    """
    elab_ok(src)


def test_arrow_sugar() -> None:
    src = """
    let k(A: Type 0, B: Type 0, a: A, b: B): A := a;
    k
    """
    elab_ok(src)


def test_check_mode_lambda() -> None:
    src = """
    let id2(A: Type 0): A -> A := fun (x: A) => x;
    id2
    """
    elab_ok(src)


def test_lambda_type_params() -> None:
    src = """
    let id2: (impl A: Type 0) -> A -> A := fun<A>(impl x: A) => x;
    id2
    """
    elab_ok(src)


def test_reject_infer_mode_unannotated_lambda() -> None:
    src = "fun x => x"
    with pytest.raises(SurfaceError, match="Unexpected token"):
        parse_term(src)


def test_typed_let() -> None:
    src = "let A: Type 1 := Type 0; A"
    elab_ok(src)


def test_const_syntax() -> None:
    elab_ok("const Nat")


def test_ind_ctor_syntax() -> None:
    elab_ok("ind Nat")
    elab_ok("ctor Nat.Zero")


def test_uapp_syntax() -> None:
    elab_ok("Vec_U@{0}")
