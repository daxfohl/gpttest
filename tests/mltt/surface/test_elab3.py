from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.prelude import prelude_env


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


def test_implicit_id_explicit() -> None:
    src = """
    let id : {A : Type 0} -> A -> A :=
      fun {A : Type 0} (x : A) => x;
    id {Nat} Nat.Zero
    """
    elab_ok(src)


def test_implicit_id_omitted() -> None:
    src = """
    let id : {A : Type 0} -> A -> A :=
      fun {A : Type 0} (x : A) => x;
    id Nat.Zero
    """
    elab_ok(src)


def test_multiple_implicits() -> None:
    src = """
    let k : {A : Type 0} {B : Type 0} -> A -> B -> A :=
      fun {A : Type 0} {B : Type 0} (a : A) (b : B) => a;
    k Nat.Zero (Nat.Succ Nat.Zero)
    """
    elab_ok(src)
