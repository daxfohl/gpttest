import pytest

from mltt.elab.arg_matcher import ArgMatcher
from mltt.elab.east import EArg, EVar
from mltt.elab.etype import ElabBinderInfo
from mltt.surface.sast import Span, SurfaceError


def _arg(name: str, *, implicit: bool = False, arg_name: str | None = None) -> EArg:
    span = Span(0, 0)
    return EArg(term=EVar(span=span, name=name), implicit=implicit, name=arg_name)


def test_matcher_named_requires_binders() -> None:
    binders = (ElabBinderInfo(name=None, implicit=False),)
    args = (_arg("x", arg_name="x"),)
    with pytest.raises(SurfaceError, match="Named arguments require binder names"):
        ArgMatcher(binders, args, Span(0, 0))


def test_matcher_positional_after_named_rejected() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    args = (_arg("x", arg_name="x"), _arg("y"))
    with pytest.raises(SurfaceError, match="Positional arguments must come before"):
        ArgMatcher(binders, args, Span(0, 0))


def test_matcher_duplicate_named_rejected() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    args = (_arg("x", arg_name="x"), _arg("y", arg_name="x"))
    with pytest.raises(SurfaceError, match="Duplicate named argument x"):
        ArgMatcher(binders, args, Span(0, 0))


def test_matcher_implicit_skips_explicit_positional() -> None:
    binders = (ElabBinderInfo(name="x", implicit=True), ElabBinderInfo())
    matcher = ArgMatcher(binders, (_arg("a"),), Span(0, 0))
    decision = matcher.match_for_binder(binders[0], allow_partial=False)
    assert decision.kind == "implicit"
    assert matcher.has_positional()


def test_matcher_named_gap_with_partial() -> None:
    binders = (
        ElabBinderInfo(name="x", implicit=False),
        ElabBinderInfo(name="y", implicit=False),
    )
    matcher = ArgMatcher(binders, (_arg("y", arg_name="y"),), Span(0, 0))
    first = matcher.match_for_binder(binders[0], allow_partial=True)
    second = matcher.match_for_binder(binders[1], allow_partial=True)
    assert first.kind == "missing"
    assert second.kind == "explicit"


def test_matcher_partial_stop() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    matcher = ArgMatcher(binders, (), Span(0, 0))
    decision = matcher.match_for_binder(binders[0], allow_partial=True)
    assert decision.kind == "stop"


def test_matcher_unknown_named_leftover() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    matcher = ArgMatcher(binders, (_arg("y", arg_name="y"),), Span(0, 0))
    matcher.match_for_binder(binders[0], allow_partial=True)
    assert matcher.unknown_named() == "y"
