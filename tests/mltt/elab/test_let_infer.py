import pytest

from mltt.kernel.tel import mk_app
from mltt.elab.sast import SurfaceError
from elab_helpers import elab_eval, elab_ok, get_ctor


def test_let_infer_partial() -> None:
    src = """
    let k(a: Nat, b: Nat): Nat := a;
    let f := partial k(Nat.Zero);
    f(Nat.Succ(Nat.Zero))
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_let_infer_simple() -> None:
    src = """
    let id := fun (x: Nat) => x;
    id(Nat.Succ(Nat.Zero))
    """
    succ = get_ctor("Nat.Succ")
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == mk_app(succ, zero).normalize()


def test_let_infer_value() -> None:
    src = """
    let x := Nat.Zero;
    x
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_let_infer_generic() -> None:
    src = """
    let id<A>(x: A) := x;
    id(Nat.Zero)
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_let_infer_requires_check_mode() -> None:
    src = """
    let x := match Nat.Zero with
      | Zero => Nat.Zero
      | Succ _ => Nat.Zero;
    x
    """
    with pytest.raises(SurfaceError, match="Cannot infer match result type"):
        elab_eval(src)


def test_let_infer_uparams() -> None:
    src = """
    let id<A>(x: A) := x;
    id(Type)
    """
    elab_ok(src)
