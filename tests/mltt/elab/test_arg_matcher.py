import pytest

from mltt.elab.arg_matcher import ArgMatcher
from mltt.elab.east import EArg, ENamedArg, EVar
from mltt.elab.errors import ElabError
from mltt.elab.etype import ElabBinderInfo
from mltt.surface.sast import Span


def _arg(name: str, *, implicit: bool = False) -> EArg:
    span = Span(0, 0)
    return EArg(term=EVar(span=span, name=name), implicit=implicit)


def _named(name: str, term_name: str) -> ENamedArg:
    span = Span(0, 0)
    return ENamedArg(name=name, term=EVar(span=span, name=term_name))


def test_matcher_named_requires_binders() -> None:
    binders = (ElabBinderInfo(name=None, implicit=False),)
    args = ()
    named = (_named("x", "x"),)
    with pytest.raises(ElabError, match="Named arguments require binder names"):
        ArgMatcher(binders, args, named, Span(0, 0))


def test_matcher_implicit_skips_explicit_positional() -> None:
    binders = (ElabBinderInfo(name="x", implicit=True), ElabBinderInfo())
    matcher = ArgMatcher(binders, (_arg("a"),), (), Span(0, 0))
    decision = matcher.match_for_binder(binders[0], allow_partial=False)
    assert decision.kind == "implicit"
    assert matcher.has_positional()


def test_matcher_named_gap_with_partial() -> None:
    binders = (
        ElabBinderInfo(name="x", implicit=False),
        ElabBinderInfo(name="y", implicit=False),
    )
    matcher = ArgMatcher(binders, (), (_named("y", "y"),), Span(0, 0))
    first = matcher.match_for_binder(binders[0], allow_partial=True)
    second = matcher.match_for_binder(binders[1], allow_partial=True)
    assert first.kind == "missing"
    assert second.kind == "explicit"


def test_matcher_partial_stop() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    matcher = ArgMatcher(binders, (), (), Span(0, 0))
    decision = matcher.match_for_binder(binders[0], allow_partial=True)
    assert decision.kind == "stop"


def test_matcher_unknown_named_leftover() -> None:
    binders = (ElabBinderInfo(name="x", implicit=False),)
    matcher = ArgMatcher(binders, (), (_named("y", "y"),), Span(0, 0))
    matcher.match_for_binder(binders[0], allow_partial=True)
    assert matcher.unknown_named() == "y"
