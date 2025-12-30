import pytest

from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError
from mltt.surface.prelude import prelude_env


def test_explicit_id() -> None:
    src = """
    let id : (A : Type 0) -> A -> A :=
      fun (A : Type 0) => fun (x : A) => x;
    id
    """
    term = parse_term(src)
    env = prelude_env()
    term.elab_infer(env)


def test_arrow_sugar() -> None:
    src = """
    let k : (A : Type 0) -> (B : Type 0) -> A -> B -> A :=
      fun (A : Type 0) (B : Type 0) => fun (a : A) (b : B) => a;
    k
    """
    term = parse_term(src)
    env = prelude_env()
    term.elab_infer(env)


def test_check_mode_unannotated_lambda() -> None:
    src = """
    let id2 : (A : Type 0) -> A -> A :=
      fun (A : Type 0) => fun x => x;
    id2
    """
    term = parse_term(src)
    env = prelude_env()
    term.elab_infer(env)


def test_reject_infer_mode_unannotated_lambda() -> None:
    src = "fun x => x"
    term = parse_term(src)
    env = prelude_env()
    with pytest.raises(SurfaceError, match="Cannot infer unannotated lambda"):
        term.elab_infer(env)


def test_typed_let() -> None:
    src = "let A : Type 1 := Type 0; A"
    term = parse_term(src)
    env = prelude_env()
    term.elab_infer(env)


def test_const_syntax() -> None:
    term = parse_term("const Nat")
    env = prelude_env()
    term.elab_infer(env)


def test_ind_ctor_syntax() -> None:
    env = prelude_env()
    parse_term("ind Nat").elab_infer(env)
    parse_term("ctor Nat.Zero").elab_infer(env)


def test_uapp_syntax() -> None:
    env = prelude_env()
    parse_term("Vec_U@{0}").elab_infer(env)
