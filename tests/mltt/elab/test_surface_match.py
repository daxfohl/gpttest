import pytest

from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabEnv
from mltt.elab.sast import elab_infer
from mltt.surface.sast import SurfaceError
from mltt.surface.parse import parse_term, parse_term_raw
from mltt.kernel.prelude import prelude_env


def elab_ok(src: str) -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = elab_infer(term, env, state)
    state.solve(env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def elab_fails(src: str) -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    try:
        term = parse_term(src)
        elab_infer(term, env, state)
    except SurfaceError:
        return
    raise AssertionError("Expected surface error")


def test_match_pred() -> None:
    src = """
    let pred(n: Nat): Nat := 
      match n with
      | Zero => ctor Nat.Zero
      | Succ k => k;
    pred
    """
    elab_ok(src)


def test_match_pred_dependent() -> None:
    src = """
    let pred(n: Nat): Nat := 
      match n return Nat with
      | Zero => ctor Nat.Zero
      | Succ k => k;
    pred
    """
    elab_ok(src)


def test_match_drop_succ() -> None:
    src = """
    let drop(n: Nat): Nat := 
      match n with
      | Zero => ctor Nat.Zero
      | Succ _ => ctor Nat.Zero;
    drop
    """
    elab_ok(src)


def test_match_missing_branch() -> None:
    src = """
    let drop(n: Nat): Nat := 
      match n with
      | Zero => ctor Nat.Zero;
    drop
    """
    elab_fails(src)


def test_let_destruct_sigma() -> None:
    src = """
    let p := ctor Sigma.Pair(Nat, fun (x: Nat) => Nat, Nat.Zero, Nat.Zero);
    let (a, b) := p;
    a
    """
    elab_ok(src)


def test_match_nested_pattern() -> None:
    src = """
    let xs := ctor List.Cons(Nat, Nat.Zero, ctor List.Cons(Nat, Nat.Zero, ctor List.Nil(Nat)));
    let y: Nat := 
      match xs with
      | Cons x (Cons y ys) => y
      | _ => Nat.Zero;
    y
    """
    elab_ok(src)


def test_match_multi_scrutinee() -> None:
    src = """
    let n := Nat.Zero;
    let b := ctor Bool.True;
    let m: Nat := 
      match n, b with
      | (Zero, True) => Nat.Zero
      | _ => Nat.Succ(Nat.Zero);
    m
    """
    elab_ok(src)


def test_match_duplicate_binder_error() -> None:
    src = """
    let xs := ctor List.Nil(Nat);
    let y := 
      match xs return Nat with
      | Cons x x => Nat.Zero
      | _ => Nat.Zero;
    y
    """
    elab_fails(src)


def test_let_destruct_refutable_error() -> None:
    src = """
    let xs := ctor List.Nil(Nat);
    let Cons x ys := xs;
    x
    """
    elab_fails(src)


def test_match_requires_desugared_single_scrutinee() -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term_raw(
        """
        match n, b with
        | _ => n
        """
    )
    with pytest.raises(SurfaceError, match="Match must be desugared to one scrutinee"):
        elab_infer(term, env, state)


def test_let_tuple_pattern_requires_desugaring() -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term_raw(
        """
        let p := ctor Sigma.Pair(Nat, fun (x: Nat) => Nat, Nat.Zero, Nat.Zero);
        let (x, y) := p;
        x
        """
    )
    with pytest.raises(SurfaceError, match="Tuple patterns must be desugared"):
        elab_infer(term, env, state)
