from types import MappingProxyType

from mltt.kernel.env import Env
from mltt.kernel.prelude import prelude_env, prelude_globals
from mltt.elab.elab_helpers import elab_ok_in_env


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
