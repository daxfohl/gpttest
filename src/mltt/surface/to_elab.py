"""Convert desugared surface AST into elaborator AST."""

from __future__ import annotations

from mltt.common.span import Span
from mltt.elab.ast import (
    EAnn,
    EApp,
    EArg,
    EBinder,
    EBranch,
    EConst,
    ECtor,
    EConstructorDecl,
    EHole,
    EInd,
    EInductiveDef,
    ELevel,
    ELam,
    ELet,
    EMatch,
    ENamedArg,
    EPartial,
    EPat,
    EPatCtor,
    EPatVar,
    EPatWild,
    EPi,
    EUApp,
    EUniv,
    EVar,
    ETerm,
)
from mltt.surface.sast import (
    Pat,
    PatCtor,
    PatTuple,
    PatVar,
    PatWild,
    SAnn,
    SApp,
    SArg,
    SBinder,
    SBranch,
    SConst,
    SCtor,
    SConstructorDecl,
    SHole,
    SInd,
    SInductiveDef,
    SLet,
    SLetPat,
    SLam,
    SMatch,
    SPi,
    SPartial,
    SUApp,
    SUniv,
    SVar,
    SurfaceError,
    SurfaceTerm,
)


def surface_to_elab(term: SurfaceTerm) -> ETerm:
    return _convert_term(term, [])


def _convert_term(term: SurfaceTerm, level_env: list[str]) -> ETerm:
    match term:
        case SVar():
            return EVar(span=term.span, name=term.name)
        case SConst():
            return EConst(span=term.span, name=term.name)
        case SUniv():
            if term.level is None:
                level = None
            else:
                level = _convert_level(term.level, level_env, term.span)
            return EUniv(span=term.span, level=level)
        case SAnn():
            return EAnn(
                span=term.span,
                term=_convert_term(term.term, level_env),
                ty=_convert_term(term.ty, level_env),
            )
        case SHole():
            return EHole(span=term.span)
        case SLam():
            return ELam(
                span=term.span,
                binders=tuple(_convert_binder(b, level_env) for b in term.binders),
                body=_convert_term(term.body, level_env),
            )
        case SPi():
            return EPi(
                span=term.span,
                binders=tuple(_convert_binder(b, level_env) for b in term.binders),
                body=_convert_term(term.body, level_env),
            )
        case SApp():
            positional: list[EArg] = []
            named: dict[str, ENamedArg] = {}
            named_order: list[ENamedArg] = []
            named_seen = False
            for arg in term.args:
                if arg.name is not None:
                    named_seen = True
                    if arg.name in named:
                        raise SurfaceError(
                            f"Duplicate named argument {arg.name}", arg.term.span
                        )
                    named_arg = ENamedArg(
                        name=arg.name,
                        term=_convert_term(arg.term, level_env),
                    )
                    named[arg.name] = named_arg
                    named_order.append(named_arg)
                    continue
                if named_seen:
                    raise SurfaceError(
                        "Positional arguments must come before named arguments",
                        arg.term.span,
                    )
                positional.append(_convert_arg(arg, level_env))
            return EApp(
                span=term.span,
                fn=_convert_term(term.fn, level_env),
                args=tuple(positional),
                named_args=tuple(named_order),
            )
        case SUApp():
            return EUApp(
                span=term.span,
                head=_convert_term(term.head, level_env),
                levels=tuple(
                    _convert_level(level, level_env, term.span) for level in term.levels
                ),
            )
        case SPartial():
            return EPartial(span=term.span, term=_convert_term(term.term, level_env))
        case SLet():
            if term.params:
                raise SurfaceError("Let parameters must be desugared", term.span)
            extended_env = _push_uparams(level_env, term.uparams)
            return ELet(
                span=term.span,
                uparams=term.uparams,
                name=term.name,
                ty=(
                    _convert_term(term.ty, extended_env)
                    if term.ty is not None
                    else None
                ),
                val=_convert_term(term.val, extended_env),
                body=_convert_term(term.body, level_env),
            )
        case SMatch():
            if len(term.scrutinees) != 1:
                raise SurfaceError(
                    "Match must be desugared to one scrutinee", term.span
                )
            if term.as_names:
                raise SurfaceError("Match as-name must be desugared", term.span)
            return EMatch(
                span=term.span,
                scrutinee=_convert_term(term.scrutinees[0], level_env),
                motive=(
                    _convert_term(term.motive, level_env)
                    if term.motive is not None
                    else None
                ),
                branches=tuple(
                    _convert_branch(branch, level_env) for branch in term.branches
                ),
            )
        case SLetPat():
            raise SurfaceError("Let patterns must be desugared", term.span)
        case SInd():
            return EInd(span=term.span, name=term.name)
        case SCtor():
            return ECtor(span=term.span, name=term.name)
        case SInductiveDef():
            extended_env = _push_uparams(level_env, term.uparams)
            return EInductiveDef(
                span=term.span,
                name=term.name,
                uparams=term.uparams,
                params=tuple(_convert_binder(b, extended_env) for b in term.params),
                level=_convert_term(term.level, extended_env),
                ctors=tuple(_convert_ctor_decl(c, extended_env) for c in term.ctors),
                body=_convert_term(term.body, level_env),
            )
        case _:
            raise SurfaceError("Unsupported surface term for elaboration", term.span)


def _convert_binder(binder: SBinder, level_env: list[str]) -> EBinder:
    ty = _convert_term(binder.ty, level_env) if binder.ty is not None else None
    return EBinder(
        name=binder.name,
        ty=ty,
        span=binder.span,
        implicit=binder.implicit,
    )


def _convert_arg(arg: SArg, level_env: list[str]) -> EArg:
    return EArg(
        term=_convert_term(arg.term, level_env),
        implicit=arg.implicit,
    )


def _convert_branch(branch: SBranch, level_env: list[str]) -> EBranch:
    return EBranch(
        pat=_convert_pat(branch.pat),
        rhs=_convert_term(branch.rhs, level_env),
        span=branch.span,
    )


def _convert_pat(pat: Pat) -> EPat:
    match pat:
        case PatVar():
            return EPatVar(span=pat.span, name=pat.name)
        case PatWild():
            return EPatWild(span=pat.span)
        case PatCtor():
            args = tuple(_convert_pat(arg) for arg in pat.args)
            if any(isinstance(arg, EPatCtor) for arg in args):
                raise SurfaceError("Nested patterns must be desugared", pat.span)
            return EPatCtor(span=pat.span, ctor=pat.ctor, args=args)
        case PatTuple():
            raise SurfaceError("Tuple patterns must be desugared", pat.span)
        case _:
            raise SurfaceError("Unsupported pattern for elaboration", pat.span)


def _convert_ctor_decl(
    ctor: SConstructorDecl, level_env: list[str]
) -> EConstructorDecl:
    if ctor.result is None:
        raise SurfaceError("Constructor result must be desugared", ctor.span)
    return EConstructorDecl(
        name=ctor.name,
        fields=tuple(_convert_binder(b, level_env) for b in ctor.fields),
        result=_convert_term(ctor.result, level_env),
        span=ctor.span,
    )


def _convert_level(level: int | str, level_env: list[str], span: Span) -> ELevel:
    if isinstance(level, int):
        return ELevel(kind="const", value=level)
    if level in level_env:
        return ELevel(kind="bound", value=level_env.index(level))
    raise SurfaceError(f"Unknown universe level {level}", span)


def _push_uparams(level_env: list[str], uparams: tuple[str, ...]) -> list[str]:
    if not uparams:
        return list(level_env)
    return list(reversed(uparams)) + list(level_env)
