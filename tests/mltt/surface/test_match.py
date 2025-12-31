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


def test_let_destruct_sigma() -> None:
    src = """
    let p : Sigma Nat (fun (x : Nat) => Nat) := ctor Sigma.Pair Nat (fun (x : Nat) => Nat) Nat.Zero Nat.Zero;
    let (a, b) := p;
    a
    """
    elab_ok(src)


def test_match_nested_pattern() -> None:
    src = """
    let xs : const List Nat := ctor List.Cons Nat Nat.Zero (ctor List.Cons Nat Nat.Zero (ctor List.Nil Nat));
    let y : Nat :=
      match xs with
      | Cons x (Cons y ys) => y
      | _ => Nat.Zero;
    y
    """
    elab_ok(src)


def test_match_multi_scrutinee() -> None:
    src = """
    let n : Nat := Nat.Zero;
    let b : Bool := ctor Bool.True;
    let m : Nat :=
      match n, b with
      | (Zero, True) => Nat.Zero
      | _ => Nat.Succ Nat.Zero;
    m
    """
    elab_ok(src)


def test_match_duplicate_binder_error() -> None:
    src = """
    let xs : const List Nat := ctor List.Nil Nat;
    let y : Nat :=
      match xs with
      | Cons x x => Nat.Zero
      | _ => Nat.Zero;
    y
    """
    elab_fails(src)


def test_let_destruct_refutable_error() -> None:
    src = """
    let xs : const List Nat := ctor List.Nil Nat;
    let Cons x ys := xs;
    x
    """
    elab_fails(src)
