from mltt.kernel.prelude import prelude_env
from elab_helpers import elab_ok, elab_ok_in_env


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
