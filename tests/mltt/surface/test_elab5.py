from types import MappingProxyType

from mltt.kernel.environment import Env
from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.prelude import prelude_env, prelude_globals


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


def elab_ok_in_env(src: str, env: Env) -> None:
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


def test_local_universe_binders_with_id() -> None:
    src = """
    let id : {A : Type} -> A -> A :=
      fun {A} (x : A) => x;
    let x : Nat := id {Nat} Nat.Zero;
    id {Type} Type
    """
    elab_ok(src)


def test_local_universe_binders_with_id_implicit() -> None:
    src = """
    let id : {A : Type} -> A -> A :=
      fun {A} (x : A) => x;
    let x : Nat := id Nat.Zero;
    x
    """
    elab_ok(src)


def test_surface_inductive_maybe() -> None:
    g = prelude_globals()
    for name in (
        "Maybe",
        "Maybe.Nothing",
        "Maybe.Just",
        "Maybe_U",
        "Maybe.Nothing_U",
        "Maybe.Just_U",
    ):
        g.pop(name, None)
    env = Env(globals=MappingProxyType(g))
    src = """
    inductive Maybe (A : Type 0) : Type 0 :=
    | Nothing
    | Just (x : A);
    let mk : (A : Type 0) -> const Maybe A :=
      fun (A : Type 0) => ctor Maybe.Nothing A;
    let m : const Maybe Nat := mk Nat;
    m
    """
    elab_ok_in_env(src, env)
