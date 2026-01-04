import pytest

from mltt.elab.east import EMatch
from mltt.surface.desugar import desugar
from mltt.surface.parse import parse_term_raw
from mltt.surface.sast import SurfaceError
from mltt.surface.to_elab import surface_to_elab


def test_to_elab_accepts_desugared_match() -> None:
    term = desugar(
        parse_term_raw(
            """
            match n with
            | _ => Nat.Zero
            """
        )
    )
    elab_term = surface_to_elab(term)
    assert isinstance(elab_term, EMatch)


def test_to_elab_rejects_multi_scrutinee_match() -> None:
    term = parse_term_raw(
        """
        match n, b with
        | _ => Nat.Zero
        """
    )
    with pytest.raises(SurfaceError, match="Match must be desugared to one scrutinee"):
        surface_to_elab(term)


def test_to_elab_rejects_tuple_pattern() -> None:
    term = parse_term_raw(
        """
        match p with
        | (x, y) => x
        | _ => Nat.Zero
        """
    )
    with pytest.raises(SurfaceError, match="Tuple patterns must be desugared"):
        surface_to_elab(term)


def test_to_elab_rejects_nested_pattern() -> None:
    term = parse_term_raw(
        """
        match xs with
        | Cons x (Cons y ys) => y
        | _ => Nat.Zero
        """
    )
    with pytest.raises(SurfaceError, match="Nested patterns must be desugared"):
        surface_to_elab(term)


def test_to_elab_rejects_let_tuple_pattern() -> None:
    term = parse_term_raw(
        """
        let p := ctor Sigma.Pair(Nat, fun (x: Nat) => Nat, Nat.Zero, Nat.Zero);
        let (x, y) := p;
        x
        """
    )
    with pytest.raises(SurfaceError, match="Let patterns must be desugared"):
        surface_to_elab(term)


def test_to_elab_rejects_match_as_name() -> None:
    term = parse_term_raw(
        """
        match n as m return Nat with
        | _ => Nat.Zero
        """
    )
    with pytest.raises(SurfaceError, match="Match as-name must be desugared"):
        surface_to_elab(term)


def test_to_elab_rejects_ctor_result_omitted() -> None:
    term = parse_term_raw(
        """
        inductive Maybe(A: Type 0): Type 0 :=
        | Nothing
        | Just(x: A);
        Maybe
        """
    )
    with pytest.raises(SurfaceError, match="Constructor result must be desugared"):
        surface_to_elab(term)


def test_to_elab_rejects_positional_after_named() -> None:
    term = parse_term_raw(
        """
        k(a := Nat.Zero, Nat.Zero)
        """
    )
    with pytest.raises(
        SurfaceError, match="Positional arguments must come before named"
    ):
        surface_to_elab(term)


def test_to_elab_rejects_duplicate_named_args() -> None:
    term = parse_term_raw(
        """
        k(a := Nat.Zero, a := Nat.Succ(Nat.Zero))
        """
    )
    with pytest.raises(SurfaceError, match="Duplicate named argument a"):
        surface_to_elab(term)
