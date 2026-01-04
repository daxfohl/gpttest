from dataclasses import fields, is_dataclass

import pytest

from mltt.common.span import Span
from mltt.surface.desugar import desugar
from mltt.surface.parse import parse_term_raw
from mltt.surface.sast import SurfaceError


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
    let add: (m: Nat, n: Nat) -> Nat :=
      fun (m: Nat, n: Nat) =>
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
    desugared = """
    let f: (m: Nat, n: Nat) -> Nat :=
      fun (m: Nat, n: Nat) =>
        match Nat.Zero with
        | Zero => f(m, n)
        | Succ k => f(k, n);
    f
    """
    _assert_desugars(sugared, desugared)


def test_desugar_equation_rec_multi_scrutinee_no_change() -> None:
    sugared = """
    let f(m: Nat, n: Nat): Nat :=
      match m, n with
      | (Zero, Zero) => n
      | _ => n;
    f
    """
    desugared = """
    let f: (m: Nat, n: Nat) -> Nat :=
      fun (m: Nat, n: Nat) =>
        match m with
        | Zero =>
          (match n with
          | Zero => n
          | _ => n)
        | _ => n;
    f
    """
    _assert_desugars(sugared, desugared)


def test_desugar_match_multi_as_names() -> None:
    sugared = """
    match a, b as x, y return Nat with
    | (Zero, Zero) => x
    | _ => y
    """
    desugared = """
    let x := a;
    let y := b;
    match x return Nat with
    | Zero =>
      (match y with
      | Zero => x
      | _ => y)
    | _ => y
    """
    _assert_desugars(sugared, desugared)


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
    | Zero: Nat
    | Succ(k: Nat): Nat;
    let add: (m: Nat, n: Nat) -> Nat :=
      fun (m: Nat, n: Nat) =>
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


def test_desugar_match_nested_pattern() -> None:
    sugared = """
    match xs with
    | Cons x (Cons y ys) => y
    | _ => Nat.Zero
    """
    desugared = """
    match xs with
    | Cons x _pat0 =>
      (match _pat0 with
      | Cons y ys => y
      | _ => Nat.Zero)
    | _ => Nat.Zero
    """
    _assert_desugars(sugared, desugared)


def test_desugar_dependent_nested_pattern() -> None:
    sugared = """
    match xs return Nat with
    | Cons x (Cons y ys) => y
    | _ => Nat.Zero
    """
    desugared = """
    match xs return Nat with
    | Cons x _pat0 =>
      (match _pat0 with
      | Cons y ys => y
      | _ => Nat.Zero)
    | _ => Nat.Zero
    """
    _assert_desugars(sugared, desugared)


def test_desugar_match_multi_scrutinee() -> None:
    sugared = """
    match n, b with
    | (Zero, True) => Nat.Zero
    | _ => Nat.Succ(Nat.Zero)
    """
    desugared = """
    match n with
    | Zero =>
      (match b with
      | True => Nat.Zero
      | _ => Nat.Succ(Nat.Zero))
    | _ => Nat.Succ(Nat.Zero)
    """
    _assert_desugars(sugared, desugared)


def test_desugar_dependent_multi_scrutinee_branches() -> None:
    sugared = """
    match n, b return Nat with
    | (Zero, True) => Nat.Zero
    | _ => Nat.Succ(Nat.Zero)
    """
    desugared = """
    match n return Nat with
    | Zero =>
      (match b with
      | True => Nat.Zero
      | _ => Nat.Succ(Nat.Zero))
    | _ => Nat.Succ(Nat.Zero)
    """
    _assert_desugars(sugared, desugared)


def test_desugar_dependent_multi_scrutinee() -> None:
    sugared = """
    match n, b return Nat with
    | _ => Nat.Zero
    """
    desugared = "Nat.Zero"
    _assert_desugars(sugared, desugared)


def test_desugar_tuple_pattern_in_match() -> None:
    sugared = """
    match p with
    | (x, y) => x
    | _ => Nat.Zero
    """
    desugared = """
    match p with
    | Pair x y => x
    | _ => Nat.Zero
    """
    _assert_desugars(sugared, desugared)


def test_desugar_tuple_pattern_in_let() -> None:
    sugared = """
    let (a, b) := p;
    a
    """
    desugared = """
    match p with
    | Pair a b => a
    """
    _assert_desugars(sugared, desugared)


def test_desugar_match_as_name() -> None:
    sugared = """
    match n as m return Nat with
    | _ => m
    """
    desugared = """
    let m := n;
    match m return Nat with
    | _ => m
    """
    _assert_desugars(sugared, desugared)


def test_desugar_ctor_result_default() -> None:
    sugared = """
    inductive Maybe(A: Type 0): Type 0 :=
    | Nothing
    | Just(x: A);
    Maybe
    """
    desugared = """
    inductive Maybe(A: Type 0): Type 0 :=
    | Nothing: Maybe(A)
    | Just(x: A): Maybe(A);
    Maybe
    """
    _assert_desugars(sugared, desugared)
