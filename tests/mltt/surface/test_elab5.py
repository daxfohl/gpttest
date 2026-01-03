from types import MappingProxyType

from mltt.kernel.env import Env
from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.etype import ElabEnv
from mltt.surface.prelude import prelude_env, prelude_globals


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


def elab_ok_in_env(src: str, env: Env) -> None:
    elab_env = ElabEnv.from_env(env)
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = term.elab_infer(elab_env, state)
    state.solve(elab_env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def test_type_without_numeral() -> None:
    src = """
    let id<A>(x: A) := x;
    id(Nat.Zero)
    """
    elab_ok(src)


def test_local_universe_binders_with_maybe() -> None:
    src = """
    let mk<A> := ctor Maybe.Nothing_U(A);
    let m := mk;
    mk<Type>
    """
    elab_ok(src)


def test_local_universe_binders_with_id() -> None:
    src = """
    let id<A>(x: A) := x;
    let x := id(Nat.Zero);
    id(Type)
    """
    elab_ok(src)


def test_local_universe_binders_with_id_implicit() -> None:
    src = """
    let id<A>(x: A) := x;
    let x := id(Nat.Zero);
    x
    """
    elab_ok(src)


def test_surface_let_universe_binders() -> None:
    env = prelude_env()
    src = """
    let id{u}(impl A: Type(u), x: A) := x;
    let x := id(Nat.Zero);
    id<Type>(Nat)
    """
    elab_ok_in_env(src, env)


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
    inductive Maybe(A: Type 0): Type 0 := 
    | Nothing
    | Just(x: A);
    let mk(A: Type 0) := ctor Maybe.Nothing(A);
    let m := mk(Nat);
    m
    """
    elab_ok_in_env(src, env)


def test_surface_inductive_ctor_implicit_fields() -> None:
    env = prelude_env()
    src = """
    inductive Wrap(A: Type 0): Type 0 := 
    | Mk(impl x: A);
    let mk(A: Type 0, x: A) := ctor Wrap.Mk(A, x := x);
    mk(Nat, Nat.Zero)
    """
    elab_ok_in_env(src, env)


def test_surface_inductive_maybe_universe_poly() -> None:
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
    inductive Maybe {u} (A: Type(u)): Type(u) := 
    | Nothing
    | Just(x: A);
    let mk(A: Type 0) := ctor Maybe.Nothing@{0}(A);
    mk(Nat)
    """
    elab_ok_in_env(src, env)


def test_surface_inductive_maybe_universe_poly_infer() -> None:
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
    inductive Maybe {u} (A: Type(u)): Type(u) := 
    | Nothing
    | Just(x: A);
    let mk(A: Type 0) := ctor Maybe.Nothing(A);
    mk(Nat)
    """
    elab_ok_in_env(src, env)
