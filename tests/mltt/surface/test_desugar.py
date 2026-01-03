from dataclasses import fields, is_dataclass

import pytest

from mltt.surface.desugar import desugar
from mltt.surface.parse import parse_term_raw
from mltt.surface.sast import Span, SurfaceError


def _strip_spans(node: object) -> object:
    if isinstance(node, Span):
        return None
    if is_dataclass(node):
        data: dict[str, object] = {}
        for field in fields(node):
            value = getattr(node, field.name)
            if field.name == "span":
                data[field.name] = None
            else:
                data[field.name] = _strip_spans(value)
        return (type(node).__name__, data)
    if isinstance(node, tuple):
        return tuple(_strip_spans(item) for item in node)
    if isinstance(node, list):
        return [_strip_spans(item) for item in node]
    return node


def _desugar_src(src: str) -> object:
    return _strip_spans(desugar(parse_term_raw(src)))


def _parse_src(src: str) -> object:
    return _strip_spans(parse_term_raw(src))


def _assert_desugars(sugared: str, desugared_src: str) -> None:
    assert _desugar_src(sugared) == _parse_src(desugared_src)


def test_desugar_equation_rec_simple() -> None:
    sugared = """
    let add(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k => succ(add(k, n));
    add
    """
    desugared = """
    let add(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k ih => succ(ih);
    add
    """
    _assert_desugars(sugared, desugared)


def test_desugar_equation_rec_non_binder_scrutinee_no_change() -> None:
    sugared = """
    let f(m: Nat, n: Nat): Nat :=
      match Nat.Zero with
      | Zero => f(m, n)
      | Succ k => f(k, n);
    f
    """
    _assert_desugars(sugared, sugared)


def test_desugar_equation_rec_multi_scrutinee_no_change() -> None:
    sugared = """
    let f(m: Nat, n: Nat): Nat :=
      match m, n with
      | (Zero, Zero) => n
      | (Succ k, Zero) => f(k, n)
      | (Zero, Succ k) => n
      | (Succ k, Succ j) => f(k, n);
    f
    """
    _assert_desugars(sugared, sugared)


def test_desugar_equation_rec_in_inductive_body() -> None:
    sugared = """
    inductive Nat: Type 0 :=
    | Zero
    | Succ(k: Nat);
    let add(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k => add(k, n);
    add
    """
    desugared = """
    inductive Nat: Type 0 :=
    | Zero
    | Succ(k: Nat);
    let add(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k ih => ih;
    add
    """
    _assert_desugars(sugared, desugared)


def test_desugar_equation_rec_multiple_scrutinee_vars_error() -> None:
    sugared = """
    let f(m: Nat, n: Nat): Nat :=
      match m with
      | Zero => n
      | Succ k =>
        match k with
        | Zero => f(k, n)
        | Succ j => f(m, n);
    f
    """
    with pytest.raises(SurfaceError, match="multiple scrutinee vars"):
        desugar(parse_term_raw(sugared))
