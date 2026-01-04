from mltt.elab.elab_helpers import elab_eval, get_ctor
from mltt.kernel.ast import UApp
from mltt.kernel.tel import mk_app


def test_partial_positional() -> None:
    src = """
    let k(a: Nat, b: Nat): Nat := a;
    let f := partial k(Nat.Zero);
    f(Nat.Succ(Nat.Zero))
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_partial_named_gap() -> None:
    src = """
    let k(a: Nat, b: Nat): Nat := a;
    let f := partial k(b := Nat.Zero);
    f(Nat.Succ(Nat.Zero))
    """
    zero = get_ctor("Nat.Zero")
    succ = get_ctor("Nat.Succ")
    assert elab_eval(src) == mk_app(succ, zero).normalize()


def test_partial_named_gap_named() -> None:
    src = """
    let k(a: Nat, b: Nat): Nat := a;
    let f := partial k(b := Nat.Zero);
    f(a := Nat.Succ(Nat.Zero))
    """
    zero = get_ctor("Nat.Zero")
    succ = get_ctor("Nat.Succ")
    assert elab_eval(src) == mk_app(succ, zero).normalize()


def test_partial_dependent_named_gap() -> None:
    src = """
    let dep<A>(x: A, P: (y: A) -> Type, p: P(x)): P(x) := p;
    let f := partial dep(P := fun (y: Nat) => Nat, p := x);
    f(Nat.Zero)
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_partial_dependent_vec() -> None:
    src = """
    let dep<A>(x: A, P: (y: A) -> Type, p: P(x)): P(x) := p;
    let f := partial dep(
        P := fun (y: Nat) => Vec(Nat, y),
        p :=
          match x with
          | Zero => ctor Vec.Nil(Nat)
          | Succ k => ctor Vec.Cons(Nat, k, Nat.Zero, ctor Vec.Nil(Nat))
    );
    f(Nat.Zero)
    """
    nat = get_ctor("Nat")
    nil = get_ctor("Vec.Nil")
    nil_head = nil.head if isinstance(nil, UApp) else nil
    assert elab_eval(src) == mk_app(nil_head, nat)


def test_partial_generic_inferred() -> None:
    src = """
    let k<A>(a: A, b: A): A := a;
    let f := partial k(Nat.Zero);
    f(Nat.Zero)
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_partial_generic_applied_as_param() -> None:
    src = """
    let k<A>(a: A, b: A): A := a;
    let f := partial k(A := Nat);
    f(Nat.Zero, Nat.Zero)
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_partial_generic_applied_as_generic() -> None:
    src = """
    let k<A>(a: A, b: A): A := a;
    let f := partial k<Nat>();
    f(Nat.Zero, Nat.Zero)
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero


def test_partial_fully_applied() -> None:
    src = """
    let k(a: Nat): Nat := a;
    let f := partial k(Nat.Zero);
    f()
    """
    zero = get_ctor("Nat.Zero")
    assert elab_eval(src) == zero
