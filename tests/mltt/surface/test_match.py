from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.prelude import prelude_env
from mltt.surface.sast import SurfaceError


def elab_ok(src: str) -> None:
    env = prelude_env()
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = term.elab_infer(env, state)
    state.solve(env)
    term_k = state.zonk(term_k)
    ty_k = state.zonk(ty_k)
    state.ensure_solved()
    _ = (term_k, ty_k)


def elab_fails(src: str) -> None:
    env = prelude_env()
    state = ElabState()
    term = parse_term(src)
    try:
        term.elab_infer(env, state)
    except SurfaceError:
        return
    raise AssertionError("Expected surface error")


def test_match_pred() -> None:
    src = """
    let pred : Nat -> Nat :=
      fun (n : Nat) =>
        match n with
        | Zero => ctor Nat.Zero
        | Succ k => k;
    pred
    """
    elab_ok(src)


def test_match_pred_dependent() -> None:
    src = """
    let pred : Nat -> Nat :=
      fun (n : Nat) =>
        match n as z return Nat with
        | Zero => ctor Nat.Zero
        | Succ k => k;
    pred
    """
    elab_ok(src)


def test_match_drop_succ() -> None:
    src = """
    let drop : Nat -> Nat :=
      fun (n : Nat) =>
        match n with
        | Zero => ctor Nat.Zero
        | Succ _ => ctor Nat.Zero;
    drop
    """
    elab_ok(src)


def test_match_missing_branch() -> None:
    src = """
    let drop : Nat -> Nat :=
      fun (n : Nat) =>
        match n with
        | Zero => ctor Nat.Zero;
    drop
    """
    elab_fails(src)
