import mltt.elab.elab_helpers as elab_helpers
import mltt.kernel.tel as tel


def test_surface_kitchen_sink_rec_match_partial() -> None:
    src = """
    let add(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k => Nat.Succ(add(k, n));
    let inc(n: Nat): Nat := add(n, Nat.Succ(Nat.Zero));
    let f := partial add(Nat.Succ(Nat.Zero));
    f(Nat.Zero)
    """
    zero = elab_helpers.get_ctor("Nat.Zero")
    succ = elab_helpers.get_ctor("Nat.Succ")
    assert elab_helpers.elab_eval(src) == tel.mk_app(succ, zero).normalize()


def test_surface_kitchen_sink_named_implicit() -> None:
    src = """
    let pick(impl A: Type, x: A, y: A): A := x;
    let g := pick(Nat.Zero, y := Nat.Succ(Nat.Zero));
    g
    """
    zero = elab_helpers.get_ctor("Nat.Zero")
    assert elab_helpers.elab_eval(src) == zero
