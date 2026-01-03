import pytest

from mltt.elab.sast import SurfaceError
from mltt.surface.parse import parse_term
from elab_helpers import elab_ok


def test_reject_infer_mode_unannotated_lambda() -> None:
    src = "fun x => x"
    with pytest.raises(SurfaceError, match="Unexpected token"):
        parse_term(src)


def test_const_syntax() -> None:
    elab_ok("const Nat")


def test_ind_ctor_syntax() -> None:
    elab_ok("ind Nat")
    elab_ok("ctor Nat.Zero")


def test_uapp_syntax() -> None:
    elab_ok("Vec_U@{0}")
