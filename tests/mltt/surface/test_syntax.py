import pytest

from mltt.surface.parse import parse_term
from mltt.surface.sast import SurfaceError


def test_reject_infer_mode_unannotated_lambda() -> None:
    src = "fun x => x"
    with pytest.raises(SurfaceError, match="Unexpected token"):
        parse_term(src)


def test_const_syntax() -> None:
    parse_term("const Nat")


def test_ind_ctor_syntax() -> None:
    parse_term("ind Nat")
    parse_term("ctor Nat.Zero")


def test_uapp_syntax() -> None:
    parse_term("Vec_U@{0}")
