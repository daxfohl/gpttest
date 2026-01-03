"""Parser for the surface language."""

from __future__ import annotations

from typing import cast

import ply.lex as lex  # type: ignore[import-untyped]
import ply.yacc as yacc  # type: ignore[import-untyped]

from mltt.surface.sast import (
    Span,
    SurfaceError,
    SurfaceTerm,
    SBinder,
    SArg,
    SVar,
    SConst,
    SUniv,
    SAnn,
    SLam,
    SPi,
    SApp,
    SUApp,
    SHole,
    SLet,
    SPartial,
)
from mltt.surface.match import (
    Pat,
    PatCtor,
    PatTuple,
    PatVar,
    PatWild,
    SBranch,
    SLetPat,
    SMatch,
)
from mltt.surface.sind import SConstructorDecl, SCtor, SInd, SInductiveDef

_SOURCE: str = ""

reserved = {
    "fun": "FUN",
    "let": "LET",
    "match": "MATCH",
    "with": "WITH",
    "as": "AS",
    "return": "RETURN",
    "Type": "TYPE",
    "const": "CONST",
    "ind": "IND",
    "ctor": "CTOR",
    "inductive": "INDUCTIVE",
    "impl": "IMPL",
    "partial": "PARTIAL",
}

tokens = (
    "IDENT",
    "INT",
    "HOLE",
    "ARROW",
    "DARROW",
    "DEFINE",
    "COLON",
    "PIPE",
    "LPAREN",
    "RPAREN",
    "SEMI",
    "AT",
    "LBRACE",
    "RBRACE",
    "LANGLE",
    "RANGLE",
    "COMMA",
    *tuple(reserved.values()),
)

t_ARROW = r"->"
t_DARROW = r"=>"
t_DEFINE = r":="
t_COLON = r":"
t_PIPE = r"\|"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_SEMI = r";"
t_AT = r"@"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_LANGLE = r"<"
t_RANGLE = r">"
t_COMMA = r","

t_ignore = " \t"


def t_newline(t: lex.LexToken) -> None:
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_INT(t: lex.LexToken) -> lex.LexToken:
    r"\d+"
    t.end = t.lexpos + len(t.value)
    t.value = int(t.value)
    return t


def t_IDENT(t: lex.LexToken) -> lex.LexToken:
    r"[A-Za-z_][A-Za-z0-9_\.]*"
    if t.value == "_":
        t.type = "HOLE"
    else:
        t.type = reserved.get(t.value, "IDENT")
    t.end = t.lexpos + len(t.value)
    return t


def t_error(t: lex.LexToken) -> None:
    span = Span(t.lexpos, t.lexpos + 1)
    raise SurfaceError(f"Unexpected character {t.value[0]!r}", span, _SOURCE)


precedence = (("right", "ARROW"),)


def _tok_span(tok: lex.LexToken) -> Span:
    end = getattr(tok, "end", tok.lexpos + len(str(tok.value)))
    return Span(tok.lexpos, end)


def _item_span(p: yacc.YaccProduction, index: int) -> Span:
    value = p[index]
    if isinstance(value, SurfaceTerm):
        return value.span
    tok = cast(lex.LexToken, p.slice[index])
    return _tok_span(tok)


def _span(p: yacc.YaccProduction, start: int, end: int) -> Span:
    start_span = _item_span(p, start)
    end_span = _item_span(p, end)
    return Span(start_span.start, end_span.end)


def p_term_let(p: yacc.YaccProduction) -> None:
    "term : LET IDENT COLON term DEFINE term SEMI term"
    span = _span(p, 1, 8)
    p[0] = SLet(
        span=span,
        uparams=(),
        name=p[2],
        ty=p[4],
        val=p[6],
        body=p[8],
    )


def p_term_let_suffix_uparams(p: yacc.YaccProduction) -> None:
    "term : LET IDENT u_binder COLON term DEFINE term SEMI term"
    span = _span(p, 1, 9)
    p[0] = SLet(
        span=span,
        uparams=p[3],
        name=p[2],
        ty=p[5],
        val=p[7],
        body=p[9],
    )


def p_term_let_binders(p: yacc.YaccProduction) -> None:
    "term : LET IDENT let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 9)
    ty_span = Span(p[3][0].span.start, p[5].span.end)
    val_span = Span(p[3][0].span.start, p[7].span.end)
    ty = SPi(span=ty_span, binders=p[3], body=p[5])
    val = SLam(span=val_span, binders=p[3], body=p[7])
    p[0] = SLet(
        span=span,
        uparams=(),
        name=p[2],
        ty=ty,
        val=val,
        body=p[9],
    )


def p_term_let_binders_suffix_uparams(p: yacc.YaccProduction) -> None:
    "term : LET IDENT u_binder let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 10)
    ty_span = Span(p[4][0].span.start, p[6].span.end)
    val_span = Span(p[4][0].span.start, p[8].span.end)
    ty = SPi(span=ty_span, binders=p[4], body=p[6])
    val = SLam(span=val_span, binders=p[4], body=p[8])
    p[0] = SLet(
        span=span,
        uparams=p[3],
        name=p[2],
        ty=ty,
        val=val,
        body=p[10],
    )


def p_term_let_type_params(p: yacc.YaccProduction) -> None:
    "term : LET IDENT type_params COLON term DEFINE term SEMI term"
    span = _span(p, 1, 9)
    binders = p[3]
    ty_span = Span(binders[0].span.start, p[5].span.end)
    val_span = Span(binders[0].span.start, p[7].span.end)
    ty = SPi(span=ty_span, binders=binders, body=p[5])
    val = SLam(span=val_span, binders=binders, body=p[7])
    p[0] = SLet(
        span=span,
        uparams=(),
        name=p[2],
        ty=ty,
        val=val,
        body=p[9],
    )


def p_term_let_type_params_binders(p: yacc.YaccProduction) -> None:
    "term : LET IDENT type_params let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 10)
    binders = p[3] + p[4]
    ty_span = Span(binders[0].span.start, p[6].span.end)
    val_span = Span(binders[0].span.start, p[8].span.end)
    ty = SPi(span=ty_span, binders=binders, body=p[6])
    val = SLam(span=val_span, binders=binders, body=p[8])
    p[0] = SLet(
        span=span,
        uparams=(),
        name=p[2],
        ty=ty,
        val=val,
        body=p[10],
    )


def p_term_let_type_params_uparams(p: yacc.YaccProduction) -> None:
    "term : LET IDENT u_binder type_params let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 11)
    binders = p[4] + p[5]
    ty_span = Span(binders[0].span.start, p[7].span.end)
    val_span = Span(binders[0].span.start, p[9].span.end)
    ty = SPi(span=ty_span, binders=binders, body=p[7])
    val = SLam(span=val_span, binders=binders, body=p[9])
    p[0] = SLet(
        span=span,
        uparams=p[3],
        name=p[2],
        ty=ty,
        val=val,
        body=p[11],
    )


def p_term_let_uparams_type_params(p: yacc.YaccProduction) -> None:
    "term : LET u_binders_nonempty IDENT type_params let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 11)
    binders = p[4] + p[5]
    ty_span = Span(binders[0].span.start, p[7].span.end)
    val_span = Span(binders[0].span.start, p[9].span.end)
    ty = SPi(span=ty_span, binders=binders, body=p[7])
    val = SLam(span=val_span, binders=binders, body=p[9])
    p[0] = SLet(
        span=span,
        uparams=p[2],
        name=p[3],
        ty=ty,
        val=val,
        body=p[11],
    )


def p_term_let_uparams(p: yacc.YaccProduction) -> None:
    "term : LET u_binders_nonempty IDENT COLON term DEFINE term SEMI term"
    span = _span(p, 1, 9)
    p[0] = SLet(
        span=span,
        uparams=p[2],
        name=p[3],
        ty=p[5],
        val=p[7],
        body=p[9],
    )


def p_term_let_uparams_binders(p: yacc.YaccProduction) -> None:
    "term : LET u_binders_nonempty IDENT let_binders COLON term DEFINE term SEMI term"
    span = _span(p, 1, 10)
    ty_span = Span(p[4][0].span.start, p[6].span.end)
    val_span = Span(p[4][0].span.start, p[8].span.end)
    ty = SPi(span=ty_span, binders=p[4], body=p[6])
    val = SLam(span=val_span, binders=p[4], body=p[8])
    p[0] = SLet(
        span=span,
        uparams=p[2],
        name=p[3],
        ty=ty,
        val=val,
        body=p[10],
    )


def p_term_let_pat(p: yacc.YaccProduction) -> None:
    "term : LET let_pat DEFINE term SEMI term"
    span = _span(p, 1, 6)
    p[0] = SLetPat(span=span, pat=p[2], value=p[4], body=p[6])


def p_term_match(p: yacc.YaccProduction) -> None:
    "term : MATCH match_scrutinees match_tail"
    as_name, motive, branches = p[3]
    span = Span(_item_span(p, 1).start, branches[-1].span.end)
    scrutinees = p[2]
    as_names = (as_name,) if as_name is not None else tuple(None for _ in scrutinees)
    p[0] = SMatch(
        span=span,
        scrutinees=scrutinees,
        as_names=as_names,
        motive=motive,
        branches=branches,
    )


def p_term_inductive(p: yacc.YaccProduction) -> None:
    "term : INDUCTIVE IDENT ind_binders COLON term DEFINE ctor_decls SEMI term"
    span = _span(p, 1, 9)
    p[0] = SInductiveDef(
        span=span,
        name=p[2],
        uparams=(),
        params=p[3],
        level=p[5],
        ctors=p[7],
        body=p[9],
    )


def p_term_inductive_uparams(p: yacc.YaccProduction) -> None:
    "term : INDUCTIVE IDENT LBRACE u_list RBRACE ind_binders COLON term DEFINE ctor_decls SEMI term"
    span = _span(p, 1, 12)
    p[0] = SInductiveDef(
        span=span,
        name=p[2],
        uparams=p[4],
        params=p[6],
        level=p[8],
        ctors=p[10],
        body=p[12],
    )


def p_term_inductive_type_params(p: yacc.YaccProduction) -> None:
    "term : INDUCTIVE IDENT type_params ind_binders COLON term DEFINE ctor_decls SEMI term"
    span = _span(p, 1, 10)
    p[0] = SInductiveDef(
        span=span,
        name=p[2],
        uparams=(),
        params=p[3] + p[4],
        level=p[6],
        ctors=p[8],
        body=p[10],
    )


def p_term_inductive_uparams_type_params(p: yacc.YaccProduction) -> None:
    "term : INDUCTIVE IDENT LBRACE u_list RBRACE type_params ind_binders COLON term DEFINE ctor_decls SEMI term"
    span = _span(p, 1, 14)
    p[0] = SInductiveDef(
        span=span,
        name=p[2],
        uparams=p[4],
        params=p[6] + p[7],
        level=p[9],
        ctors=p[11],
        body=p[13],
    )


def p_match_tail_with(p: yacc.YaccProduction) -> None:
    "match_tail : WITH match_branches"
    p[0] = (None, None, p[2])


def p_match_tail_return(p: yacc.YaccProduction) -> None:
    "match_tail : RETURN term WITH match_branches"
    p[0] = (None, p[2], p[4])


def p_match_tail_as_return(p: yacc.YaccProduction) -> None:
    "match_tail : AS IDENT RETURN term WITH match_branches"
    p[0] = (p[2], p[4], p[6])


def p_match_scrutinees_multi(p: yacc.YaccProduction) -> None:
    "match_scrutinees : match_scrutinees COMMA term"
    p[0] = p[1] + (p[3],)


def p_match_scrutinees_single(p: yacc.YaccProduction) -> None:
    "match_scrutinees : term"
    p[0] = (p[1],)


def p_term_fun(p: yacc.YaccProduction) -> None:
    "term : FUN lam_binders DARROW term"
    span = _span(p, 1, 4)
    p[0] = SLam(span=span, binders=p[2], body=p[4])


def p_term_fun_type_params(p: yacc.YaccProduction) -> None:
    "term : FUN type_params lam_binders DARROW term"
    span = _span(p, 1, 5)
    p[0] = SLam(span=span, binders=p[2] + p[3], body=p[5])


def p_term_partial(p: yacc.YaccProduction) -> None:
    "term : PARTIAL app"
    span = _span(p, 1, 2)
    p[0] = SPartial(span=span, term=p[2])


def p_term_pi(p: yacc.YaccProduction) -> None:
    "term : pi"
    p[0] = p[1]


def p_term_paren(p: yacc.YaccProduction) -> None:
    "term : LPAREN term RPAREN"
    p[0] = p[2]


def p_term_ann(p: yacc.YaccProduction) -> None:
    "term : LPAREN term COLON term RPAREN"
    span = _span(p, 1, 5)
    p[0] = SAnn(span=span, term=p[2], ty=p[4])


def p_term_hole_ann(p: yacc.YaccProduction) -> None:
    "term : HOLE COLON term"
    span = _span(p, 1, 3)
    p[0] = SAnn(span=span, term=SHole(span=_item_span(p, 1)), ty=p[3])


def p_pi_tower(p: yacc.YaccProduction) -> None:
    "pi : pi_binders ARROW term"
    span = Span(p[1][0].span.start, p[3].span.end)
    p[0] = SPi(span=span, binders=p[1], body=p[3])


def p_pi_arrow(p: yacc.YaccProduction) -> None:
    "pi : app ARROW term"
    left = p[1]
    right = p[3]
    binder = SBinder("_", left, Span(left.span.start, left.span.end), implicit=False)
    span = Span(left.span.start, right.span.end)
    p[0] = SPi(span=span, binders=(binder,), body=right)


def p_pi_app(p: yacc.YaccProduction) -> None:
    "pi : app"
    p[0] = p[1]


def p_pi_binders(p: yacc.YaccProduction) -> None:
    "pi_binders : param_group"
    p[0] = p[1]


def p_ind_binders(p: yacc.YaccProduction) -> None:
    "ind_binders : param_group"
    p[0] = p[1]


def p_ind_binders_empty(p: yacc.YaccProduction) -> None:
    "ind_binders : empty"
    p[0] = ()


def p_u_binders_nonempty_single(p: yacc.YaccProduction) -> None:
    "u_binders_nonempty : u_binder"
    p[0] = p[1]


def p_type_params(p: yacc.YaccProduction) -> None:
    "type_params : LANGLE type_param_list RANGLE"
    binders: tuple[SBinder, ...] = tuple(
        SBinder(name, SUniv(span=span, level=None), span, implicit=True)
        for name, span in p[2]
    )
    p[0] = binders


def p_type_param_list_multi(p: yacc.YaccProduction) -> None:
    "type_param_list : type_param_list COMMA IDENT"
    span = _span(p, 3, 3)
    p[0] = p[1] + ((p[3], span),)


def p_type_param_list_single(p: yacc.YaccProduction) -> None:
    "type_param_list : IDENT"
    span = _span(p, 1, 1)
    p[0] = ((p[1], span),)


def p_u_binder(p: yacc.YaccProduction) -> None:
    "u_binder : LBRACE u_list RBRACE"
    p[0] = p[2]


def p_u_list_multi(p: yacc.YaccProduction) -> None:
    "u_list : u_list COMMA IDENT"
    p[0] = p[1] + (p[3],)


def p_u_list_single(p: yacc.YaccProduction) -> None:
    "u_list : IDENT"
    p[0] = (p[1],)


def p_let_binders(p: yacc.YaccProduction) -> None:
    "let_binders : param_group"
    p[0] = p[1]


def p_lam_binders(p: yacc.YaccProduction) -> None:
    "lam_binders : param_group"
    p[0] = p[1]


def p_param_group(p: yacc.YaccProduction) -> None:
    "param_group : LPAREN param_list RPAREN"
    p[0] = tuple(
        SBinder(name, ty, span, implicit=implicit) for name, ty, span, implicit in p[2]
    )


def p_param_list_multi(p: yacc.YaccProduction) -> None:
    "param_list : param_list COMMA param_entry"
    p[0] = p[1] + (p[3],)


def p_param_list_single(p: yacc.YaccProduction) -> None:
    "param_list : param_entry"
    p[0] = (p[1],)


def p_param_entry(p: yacc.YaccProduction) -> None:
    "param_entry : IDENT COLON term"
    span = _span(p, 1, 3)
    p[0] = (p[1], p[3], span, False)


def p_param_entry_impl(p: yacc.YaccProduction) -> None:
    "param_entry : IMPL IDENT COLON term"
    span = _span(p, 1, 4)
    p[0] = (p[2], p[4], span, True)


def p_param_entry_hole(p: yacc.YaccProduction) -> None:
    "param_entry : HOLE COLON term"
    span = _span(p, 1, 3)
    p[0] = ("_", p[3], span, False)


def p_param_entry_hole_impl(p: yacc.YaccProduction) -> None:
    "param_entry : IMPL HOLE COLON term"
    span = _span(p, 1, 4)
    p[0] = ("_", p[4], span, True)


def p_ctor_decls_multi(p: yacc.YaccProduction) -> None:
    "ctor_decls : ctor_decls ctor_decl"
    p[0] = p[1] + (p[2],)


def p_ctor_decls_single(p: yacc.YaccProduction) -> None:
    "ctor_decls : ctor_decl"
    p[0] = (p[1],)


def p_match_branches_multi(p: yacc.YaccProduction) -> None:
    "match_branches : match_branches match_branch"
    p[0] = p[1] + (p[2],)


def p_match_branches_single(p: yacc.YaccProduction) -> None:
    "match_branches : match_branch"
    p[0] = (p[1],)


def p_match_branch(p: yacc.YaccProduction) -> None:
    "match_branch : PIPE pat DARROW term"
    pipe_tok = cast(lex.LexToken, p.slice[1])
    span = Span(pipe_tok.lexpos, p[4].span.end)
    p[0] = SBranch(pat=p[2], rhs=p[4], span=span)


def p_pat_ctor_args(p: yacc.YaccProduction) -> None:
    "pat : IDENT pat_args"
    ident_tok = cast(lex.LexToken, p.slice[1])
    end = p[2][-1].span.end
    p[0] = PatCtor(ctor=p[1], args=p[2], span=Span(ident_tok.lexpos, end))


def p_pat_ctor_ident(p: yacc.YaccProduction) -> None:
    "pat : IDENT"
    tok = cast(lex.LexToken, p.slice[1])
    p[0] = PatVar(name=p[1], span=Span(tok.lexpos, tok.lexpos + len(p[1])))


def p_pat_wild(p: yacc.YaccProduction) -> None:
    "pat : HOLE"
    tok = cast(lex.LexToken, p.slice[1])
    p[0] = PatWild(span=Span(tok.lexpos, tok.lexpos + 1))


def p_pat_paren(p: yacc.YaccProduction) -> None:
    "pat : LPAREN pat RPAREN"
    p[0] = p[2]


def p_pat_tuple(p: yacc.YaccProduction) -> None:
    "pat : LPAREN pat_tuple RPAREN"
    lparen = cast(lex.LexToken, p.slice[1])
    end = p[2][-1].span.end
    p[0] = PatTuple(elts=p[2], span=Span(lparen.lexpos, end))


def p_pat_args_multi(p: yacc.YaccProduction) -> None:
    "pat_args : pat_args pat_atom"
    p[0] = p[1] + (p[2],)


def p_pat_args_single(p: yacc.YaccProduction) -> None:
    "pat_args : pat_atom"
    p[0] = (p[1],)


def p_pat_atom_ident(p: yacc.YaccProduction) -> None:
    "pat_atom : IDENT"
    tok = cast(lex.LexToken, p.slice[1])
    p[0] = PatVar(name=p[1], span=Span(tok.lexpos, tok.lexpos + len(p[1])))


def p_pat_atom_wild(p: yacc.YaccProduction) -> None:
    "pat_atom : HOLE"
    tok = cast(lex.LexToken, p.slice[1])
    p[0] = PatWild(span=Span(tok.lexpos, tok.lexpos + 1))


def p_pat_atom_paren(p: yacc.YaccProduction) -> None:
    "pat_atom : LPAREN pat RPAREN"
    p[0] = p[2]


def p_pat_atom_tuple(p: yacc.YaccProduction) -> None:
    "pat_atom : LPAREN pat_tuple RPAREN"
    lparen = cast(lex.LexToken, p.slice[1])
    end = p[2][-1].span.end
    p[0] = PatTuple(elts=p[2], span=Span(lparen.lexpos, end))


def p_pat_tuple_multi(p: yacc.YaccProduction) -> None:
    "pat_tuple : pat_tuple COMMA pat"
    p[0] = p[1] + (p[3],)


def p_pat_tuple_start(p: yacc.YaccProduction) -> None:
    "pat_tuple : pat COMMA pat"
    p[0] = (p[1], p[3])


def p_let_pat_ctor(p: yacc.YaccProduction) -> None:
    "let_pat : IDENT pat_args"
    ident_tok = cast(lex.LexToken, p.slice[1])
    end = p[2][-1].span.end
    p[0] = PatCtor(ctor=p[1], args=p[2], span=Span(ident_tok.lexpos, end))


def p_let_pat_wild(p: yacc.YaccProduction) -> None:
    "let_pat : HOLE"
    tok = cast(lex.LexToken, p.slice[1])
    p[0] = PatWild(span=Span(tok.lexpos, tok.lexpos + 1))


def p_let_pat_paren(p: yacc.YaccProduction) -> None:
    "let_pat : LPAREN pat RPAREN"
    p[0] = p[2]


def p_let_pat_tuple(p: yacc.YaccProduction) -> None:
    "let_pat : LPAREN pat_tuple RPAREN"
    lparen = cast(lex.LexToken, p.slice[1])
    end = p[2][-1].span.end
    p[0] = PatTuple(elts=p[2], span=Span(lparen.lexpos, end))


def p_ctor_decl(p: yacc.YaccProduction) -> None:
    "ctor_decl : PIPE IDENT"
    span = _span(p, 1, 2)
    p[0] = SConstructorDecl(name=p[2], fields=(), result=None, span=span)


def p_ctor_decl_fields(p: yacc.YaccProduction) -> None:
    "ctor_decl : PIPE IDENT ctor_binders"
    pipe_tok = cast(lex.LexToken, p.slice[1])
    end = p[3][-1].span.end
    span = Span(pipe_tok.lexpos, end)
    p[0] = SConstructorDecl(name=p[2], fields=p[3], result=None, span=span)


def p_ctor_decl_result(p: yacc.YaccProduction) -> None:
    "ctor_decl : PIPE IDENT COLON term"
    span = _span(p, 1, 4)
    p[0] = SConstructorDecl(name=p[2], fields=(), result=p[4], span=span)


def p_ctor_decl_fields_result(p: yacc.YaccProduction) -> None:
    "ctor_decl : PIPE IDENT ctor_binders COLON term"
    pipe_tok = cast(lex.LexToken, p.slice[1])
    span = Span(pipe_tok.lexpos, p[5].span.end)
    p[0] = SConstructorDecl(name=p[2], fields=p[3], result=p[5], span=span)


def p_ctor_binders(p: yacc.YaccProduction) -> None:
    "ctor_binders : param_group"
    p[0] = p[1]


def _append_app(left: SurfaceTerm, args: tuple[SArg, ...]) -> SApp:
    if isinstance(left, SApp):
        fn = left.fn
        merged = left.args + args
        span = Span(left.span.start, args[-1].term.span.end)
        return SApp(span=span, fn=fn, args=merged)
    span = Span(left.span.start, args[-1].term.span.end)
    return SApp(span=span, fn=left, args=args)


def p_app(p: yacc.YaccProduction) -> None:
    "app : atom call_groups"
    if p[2]:
        p[0] = _append_app(p[1], p[2])
    else:
        p[0] = p[1]


def p_call_groups_multi(p: yacc.YaccProduction) -> None:
    "call_groups : call_groups call_group"
    p[0] = p[1] + p[2]


def p_call_groups_empty(p: yacc.YaccProduction) -> None:
    "call_groups : empty"
    p[0] = ()


def _apply_type_args(head: SurfaceTerm, args: tuple[SurfaceTerm, ...]) -> SurfaceTerm:
    if not args:
        return head
    return SApp(
        span=Span(head.span.start, args[-1].span.end),
        fn=head,
        args=tuple(SArg(arg, implicit=True) for arg in args),
    )


def p_atom_uapp(p: yacc.YaccProduction) -> None:
    "atom : atom_base AT LBRACE level_list RBRACE"
    span = _span(p, 1, 5)
    p[0] = SUApp(span=span, head=p[1], levels=tuple(p[4]))


def p_atom_type_args(p: yacc.YaccProduction) -> None:
    "atom : atom_base type_args"
    p[0] = _apply_type_args(p[1], p[2])


def p_atom_uapp_type_args(p: yacc.YaccProduction) -> None:
    "atom : atom_base AT LBRACE level_list RBRACE type_args"
    head = SUApp(span=_span(p, 1, 5), head=p[1], levels=tuple(p[4]))
    p[0] = _apply_type_args(head, p[6])


def p_atom_base(p: yacc.YaccProduction) -> None:
    "atom : atom_base"
    p[0] = p[1]


def p_type_args(p: yacc.YaccProduction) -> None:
    "type_args : LANGLE type_arg_list RANGLE"
    p[0] = p[2]


def p_type_arg_list_multi(p: yacc.YaccProduction) -> None:
    "type_arg_list : type_arg_list COMMA term"
    p[0] = p[1] + (p[3],)


def p_type_arg_list_single(p: yacc.YaccProduction) -> None:
    "type_arg_list : term"
    p[0] = (p[1],)


def p_call_group_explicit(p: yacc.YaccProduction) -> None:
    "call_group : LPAREN call_args RPAREN"
    p[0] = p[2]


def p_call_args_multi(p: yacc.YaccProduction) -> None:
    "call_args : call_args COMMA call_arg"
    p[0] = p[1] + (p[3],)


def p_call_args_single(p: yacc.YaccProduction) -> None:
    "call_args : call_arg"
    p[0] = (p[1],)


def p_call_arg(p: yacc.YaccProduction) -> None:
    "call_arg : term"
    p[0] = SArg(p[1], implicit=False)


def p_call_arg_named(p: yacc.YaccProduction) -> None:
    "call_arg : IDENT DEFINE term"
    p[0] = SArg(p[3], implicit=False, name=p[1])


def p_level_list_single(p: yacc.YaccProduction) -> None:
    "level_list : level_atom"
    p[0] = [p[1]]


def p_level_list_multi(p: yacc.YaccProduction) -> None:
    "level_list : level_list COMMA level_atom"
    p[0] = p[1] + [p[3]]


def p_level_atom_int(p: yacc.YaccProduction) -> None:
    "level_atom : INT"
    p[0] = p[1]


def p_level_atom_ident(p: yacc.YaccProduction) -> None:
    "level_atom : IDENT"
    p[0] = p[1]


def p_atom_base_ident(p: yacc.YaccProduction) -> None:
    "atom_base : IDENT"
    span = _span(p, 1, 1)
    p[0] = SVar(span=span, name=p[1])


def p_atom_base_hole(p: yacc.YaccProduction) -> None:
    "atom_base : HOLE"
    span = _span(p, 1, 1)
    p[0] = SHole(span=span)


def p_atom_base_univ(p: yacc.YaccProduction) -> None:
    "atom_base : TYPE INT"
    span = _span(p, 1, 2)
    p[0] = SUniv(span=span, level=p[2])


def p_atom_base_univ_plain(p: yacc.YaccProduction) -> None:
    "atom_base : TYPE"
    span = _span(p, 1, 1)
    p[0] = SUniv(span=span, level=None)


def p_atom_base_univ_paren(p: yacc.YaccProduction) -> None:
    "atom_base : TYPE LPAREN INT RPAREN"
    span = _span(p, 1, 4)
    p[0] = SUniv(span=span, level=p[3])


def p_atom_base_univ_paren_var(p: yacc.YaccProduction) -> None:
    "atom_base : TYPE LPAREN IDENT RPAREN"
    span = _span(p, 1, 4)
    p[0] = SUniv(span=span, level=p[3])


def p_atom_base_const(p: yacc.YaccProduction) -> None:
    "atom_base : CONST IDENT"
    span = _span(p, 1, 2)
    p[0] = SConst(span=span, name=p[2])


def p_atom_base_ind(p: yacc.YaccProduction) -> None:
    "atom_base : IND IDENT"
    span = _span(p, 1, 2)
    p[0] = SInd(span=span, name=p[2])


def p_atom_base_ctor(p: yacc.YaccProduction) -> None:
    "atom_base : CTOR IDENT"
    span = _span(p, 1, 2)
    p[0] = SCtor(span=span, name=p[2])


def p_empty(p: yacc.YaccProduction) -> None:
    "empty :"
    p[0] = ()


def p_error(p: lex.LexToken | None) -> None:
    if p is None:
        span = Span(len(_SOURCE), len(_SOURCE))
        raise SurfaceError("Unexpected end of input", span, _SOURCE)
    span = _tok_span(cast(lex.LexToken, p))
    raise SurfaceError("Unexpected token", span, _SOURCE)


_PARSER = None


def parse_term(source: str) -> SurfaceTerm:
    global _SOURCE, _PARSER
    _SOURCE = source
    lexer = lex.lex()
    if _PARSER is None:
        _PARSER = yacc.yacc(start="term", debug=False, write_tables=False)
    term = cast(SurfaceTerm, _PARSER.parse(source, lexer=lexer))
    if term is None:
        span = Span(len(source), len(source))
        raise SurfaceError("Unexpected end of input", span, source)
    return term
