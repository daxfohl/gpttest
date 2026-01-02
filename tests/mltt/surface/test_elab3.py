from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.etype import ElabEnv
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


def test_implicit_id_explicit() -> None:
    src = """
    let id(impl A: Type 0, x: A): A := x;
    id(impl Nat, Nat.Zero)
    """
    elab_ok(src)


def test_implicit_id_omitted() -> None:
    src = """
    let id(impl A: Type 0, x: A): A := x;
    id(Nat.Zero)
    """
    elab_ok(src)


def test_multiple_implicits() -> None:
    src = """
    let k(impl A: Type 0, impl B: Type 0, a: A, b: B): A := a;
    k(Nat.Zero, Nat.Succ(Nat.Zero))
    """
    elab_ok(src)


def test_named_args() -> None:
    src = """
    let k<A, B, C>(a: A, b: B, c: C): A := a;
    k<Nat, Nat>(Nat.Zero, c := Nat.Zero, b := Nat.Zero)
    """
    elab_ok(src)
