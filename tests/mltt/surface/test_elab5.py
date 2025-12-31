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


def test_type_without_numeral() -> None:
    src = """
    let id : {A : Type} -> A -> A :=
      fun {A} (x : A) => x;
    id Nat.Zero
    """
    elab_ok(src)


def test_local_universe_binders_with_maybe() -> None:
    src = """
    let mk : {A : Type} -> const Maybe_U A :=
      fun {A} => ctor Maybe.Nothing_U A;
    let m : const Maybe_U Nat := mk {Nat};
    mk {Type}
    """
    elab_ok(src)
