import pytest

from mltt.kernel.ast import Let, Term
from mltt.kernel.tel import mk_app
from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError
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


def elab_eval(src: str) -> Term:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_term(src)
    term_k, _ty_k = term.elab_infer(env, state)
    state.solve(env.kenv)
    term_k = state.zonk(term_k)
    state.ensure_solved()
    while isinstance(term_k, Let):
        term_k = term_k.body.subst(term_k.value)
    return term_k.normalize()


def _get_ctor(name: str) -> Term:
    env = prelude_env()
    decl = env.lookup_global(name)
    assert decl is not None
    assert decl.value is not None
    return decl.value


def test_implicit_id_explicit() -> None:
    src = """
    let id(impl A: Type 0, x: A): A := x;
    id<Nat>(Nat.Zero)
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
    let k<A, B>(a: A, b: B): A := a;
    k<Nat, Nat>(Nat.Zero, b := Nat.Zero)
    """
    elab_ok(src)


def test_named_args_evaluates() -> None:
    src = """
    let k(a: Nat, b: Nat): Nat := a;
    k(b := Nat.Succ(Nat.Zero), a := Nat.Zero)
    """
    zero = _get_ctor("Nat.Zero")
    succ = _get_ctor("Nat.Succ")
    assert elab_eval(src) == zero


def test_named_args_before_positional_rejected() -> None:
    src = """
    let k<A, B>(a: A, b: B): A := a;
    k<Nat, Nat>(a := Nat.Zero, Nat.Zero)
    """
    with pytest.raises(
        SurfaceError, match="Positional arguments must come before named"
    ):
        elab_eval(src)


def test_duplicate_named_args_rejected() -> None:
    src = """
    let k<A, B>(a: A, b: B): A := a;
    k<Nat, Nat>(a := Nat.Zero, a := Nat.Succ(Nat.Zero))
    """
    with pytest.raises(SurfaceError, match="Duplicate named argument a"):
        elab_eval(src)


def test_too_few_args_rejected() -> None:
    src = """
    let k<A, B>(a: A, b: B): A := a;
    k<Nat, Nat>(Nat.Zero)
    """
    with pytest.raises(SurfaceError, match="Missing explicit argument"):
        elab_eval(src)


def test_too_many_args_rejected() -> None:
    src = """
    let k<A, B>(a: A, b: B): A := a;
    k<Nat, Nat>(Nat.Zero, Nat.Zero, Nat.Zero)
    """
    with pytest.raises(SurfaceError, match="Application of non-function"):
        elab_eval(src)


def test_named_args_with_implicit() -> None:
    src = """
    let k(impl x: Nat, y: Nat): Nat := y;
    k(x := Nat.Zero, y := Nat.Succ(Nat.Zero))
    """
    zero = _get_ctor("Nat.Zero")
    succ = _get_ctor("Nat.Succ")
    assert elab_eval(src) == mk_app(succ, zero).normalize()


def test_named_args_dependent_type() -> None:
    src = """
    let dep(impl A: Type 0, x: A, P: (y: A) -> Type 0, p: P(x)): P(x) := p;
    dep<Nat>(x := Nat.Zero, P := fun (y: Nat) => Nat, p := x)
    """
    zero = _get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_named_args_dependent_all_named() -> None:
    src = """
    let dep(impl A: Type 0, x: A, P: (y: A) -> Type 0, p: P(x)): P(x) := p;
    dep<Nat>(P := fun (y: Nat) => Nat, p := x, x := Nat.Zero)
    """
    zero = _get_ctor("Nat.Zero")
    assert elab_eval(src) == zero
