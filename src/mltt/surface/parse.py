"""Parser for the surface language."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.surface.sast import (
    Span,
    SurfaceError,
    SurfaceTerm,
    SBinder,
    SVar,
    SConst,
    SUniv,
    SAnn,
    SLam,
    SPi,
    SApp,
    SUApp,
    SLet,
)
from mltt.surface.sind import SInd, SCtor


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    span: Span


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        start = i
        if source.startswith("->", i):
            tokens.append(Token("SYM", "->", Span(i, i + 2)))
            i += 2
            continue
        if source.startswith("=>", i):
            tokens.append(Token("SYM", "=>", Span(i, i + 2)))
            i += 2
            continue
        if source.startswith(":=", i):
            tokens.append(Token("SYM", ":=", Span(i, i + 2)))
            i += 2
            continue
        if ch in "():;@{},":
            tokens.append(Token("SYM", ch, Span(i, i + 1)))
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and source[j].isdigit():
                j += 1
            tokens.append(Token("INT", source[i:j], Span(i, j)))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in "._"):
                j += 1
            ident = source[i:j]
            kind = (
                "KW"
                if ident in {"fun", "let", "Type", "const", "ind", "ctor"}
                else "IDENT"
            )
            tokens.append(Token(kind, ident, Span(i, j)))
            i = j
            continue
        raise SurfaceError(f"Unexpected character {ch!r}", Span(i, i + 1), source)
    return tokens


class Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.pos = 0

    def _peek(self) -> Token | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _match(self, kind: str, value: str | None = None) -> Token | None:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        self.pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._match(kind, value)
        if tok is not None:
            return tok
        expected = f"{kind} {value!r}" if value is not None else kind
        next_tok = self._peek()
        span = (
            next_tok.span
            if next_tok is not None
            else Span(len(self.source), len(self.source))
        )
        raise SurfaceError(f"Expected {expected}", span, self.source)

    def parse_term(self) -> SurfaceTerm:
        let_tok = self._match("KW", "let")
        if let_tok:
            name_tok = self._expect("IDENT")
            self._expect("SYM", ":")
            ty = self.parse_term()
            self._expect("SYM", ":=")
            val = self.parse_term()
            self._expect("SYM", ";")
            body = self.parse_term()
            span = Span(let_tok.span.start, body.span.end)
            return SLet(span=span, name=name_tok.value, ty=ty, val=val, body=body)
        fun_tok = self._match("KW", "fun")
        if fun_tok:
            binders = self._parse_lambda_binders()
            arrow = self._expect("SYM", "=>")
            body = self.parse_term()
            span = Span(fun_tok.span.start, body.span.end)
            return SLam(span=span, binders=binders, body=body)
        return self._parse_pi()

    def _parse_lambda_binders(self) -> tuple[SBinder, ...]:
        binders: list[SBinder] = []
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "SYM" and tok.value == "(":
                binders.append(self._parse_annotated_binder())
                continue
            if tok.kind == "IDENT":
                self.pos += 1
                binders.append(SBinder(tok.value, None, tok.span))
                continue
            break
        if not binders:
            next_tok = self._peek()
            span = (
                next_tok.span
                if next_tok is not None
                else Span(len(self.source), len(self.source))
            )
            raise SurfaceError("Expected lambda binder", span, self.source)
        return tuple(binders)

    def _parse_annotated_binder(self) -> SBinder:
        lpar = self._expect("SYM", "(")
        name_tok = self._expect("IDENT")
        self._expect("SYM", ":")
        ty = self.parse_term()
        rpar = self._expect("SYM", ")")
        span = Span(lpar.span.start, rpar.span.end)
        return SBinder(name_tok.value, ty, span)

    def _parse_pi(self) -> SurfaceTerm:
        tok = self._peek()
        if tok is not None and tok.kind == "SYM" and tok.value == "(":
            save = self.pos
            try:
                binders = []
                while True:
                    tok = self._peek()
                    if tok is None or tok.kind != "SYM" or tok.value != "(":
                        break
                    if not self._looks_like_binder():
                        break
                    binders.append(self._parse_annotated_binder())
                if binders and self._match("SYM", "->"):
                    body = self._parse_pi()
                    span = Span(binders[0].span.start, body.span.end)
                    return SPi(span=span, binders=tuple(binders), body=body)
            except SurfaceError:
                self.pos = save
            self.pos = save
        left = self._parse_app()
        if self._match("SYM", "->"):
            right = self._parse_pi()
            binder = SBinder("_", left, Span(left.span.start, left.span.end))
            span = Span(left.span.start, right.span.end)
            return SPi(span=span, binders=(binder,), body=right)
        return left

    def _looks_like_binder(self) -> bool:
        if self.pos + 2 >= len(self.tokens):
            return False
        if self.tokens[self.pos].value != "(":
            return False
        return (
            self.tokens[self.pos + 1].kind == "IDENT"
            and self.tokens[self.pos + 2].kind == "SYM"
            and self.tokens[self.pos + 2].value == ":"
        )

    def _parse_app(self) -> SurfaceTerm:
        term = self._parse_atom()
        args: list[SurfaceTerm] = []
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "IDENT":
                args.append(self._parse_atom())
                continue
            if tok.kind == "KW" and tok.value in {"Type", "const", "ind", "ctor"}:
                args.append(self._parse_atom())
                continue
            if tok.kind == "SYM" and tok.value == "(":
                args.append(self._parse_atom())
                continue
            break
        if not args:
            return term
        span = Span(term.span.start, args[-1].span.end)
        return SApp(span=span, fn=term, args=tuple(args))

    def _parse_atom(self) -> SurfaceTerm:
        tok = self._peek()
        if tok is None:
            raise SurfaceError(
                "Unexpected end of input",
                Span(len(self.source), len(self.source)),
                self.source,
            )
        if tok.kind == "IDENT":
            self.pos += 1
            term: SurfaceTerm = SVar(span=tok.span, name=tok.value)
            return self._parse_uapp_suffix(term)
        if tok.kind == "KW" and tok.value == "Type":
            type_tok = self._expect("KW", "Type")
            level_tok: Token | None
            if self._match("SYM", "("):
                level_tok = self._expect("INT")
                self._expect("SYM", ")")
            else:
                level_tok = self._expect("INT")
            span = Span(type_tok.span.start, level_tok.span.end)
            term = SUniv(span=span, level=int(level_tok.value))
            return self._parse_uapp_suffix(term)
        if tok.kind == "KW" and tok.value == "const":
            kw = self._expect("KW", "const")
            name_tok = self._expect("IDENT")
            term = SConst(
                span=Span(kw.span.start, name_tok.span.end), name=name_tok.value
            )
            return self._parse_uapp_suffix(term)
        if tok.kind == "KW" and tok.value == "ind":
            kw = self._expect("KW", "ind")
            name_tok = self._expect("IDENT")
            term = SInd(
                span=Span(kw.span.start, name_tok.span.end), name=name_tok.value
            )
            return self._parse_uapp_suffix(term)
        if tok.kind == "KW" and tok.value == "ctor":
            kw = self._expect("KW", "ctor")
            name_tok = self._expect("IDENT")
            term = SCtor(
                span=Span(kw.span.start, name_tok.span.end), name=name_tok.value
            )
            return self._parse_uapp_suffix(term)
        if tok.kind == "SYM" and tok.value == "(":
            lpar = self._expect("SYM", "(")
            inner = self.parse_term()
            if self._match("SYM", ":"):
                ty = self.parse_term()
                rpar = self._expect("SYM", ")")
                span = Span(lpar.span.start, rpar.span.end)
                return self._parse_uapp_suffix(SAnn(span=span, term=inner, ty=ty))
            self._expect("SYM", ")")
            return self._parse_uapp_suffix(inner)
        raise SurfaceError("Expected term", tok.span, self.source)

    def _parse_uapp_suffix(self, term: SurfaceTerm) -> SurfaceTerm:
        tok = self._peek()
        if tok is None or tok.kind != "SYM" or tok.value != "@":
            return term
        at_tok = self._expect("SYM", "@")
        self._expect("SYM", "{")
        levels: list[int] = []
        while True:
            level_tok = self._expect("INT")
            levels.append(int(level_tok.value))
            if not self._match("SYM", ","):
                break
        rbrace = self._expect("SYM", "}")
        span = Span(term.span.start, rbrace.span.end)
        uapp = SUApp(span=span, head=term, levels=tuple(levels))
        if at_tok:
            return uapp
        return uapp


def parse_term(source: str) -> SurfaceTerm:
    parser = Parser(source)
    term = parser.parse_term()
    tok = parser._peek()
    if tok is not None:
        raise SurfaceError("Unexpected token", tok.span, source)
    return term
