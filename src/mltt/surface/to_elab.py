"""Convert desugared surface AST into elaborator AST."""

from __future__ import annotations

from mltt.elab.east import (
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
    return _convert_term(term)


def _convert_term(term: SurfaceTerm) -> ETerm:
    match term:
        case SVar():
            return EVar(span=term.span, name=term.name)
        case SConst():
            return EConst(span=term.span, name=term.name)
        case SUniv():
            return EUniv(span=term.span, level=term.level)
        case SAnn():
            return EAnn(
                span=term.span,
                term=_convert_term(term.term),
                ty=_convert_term(term.ty),
            )
        case SHole():
            return EHole(span=term.span)
        case SLam():
            return ELam(
                span=term.span,
                binders=tuple(_convert_binder(b) for b in term.binders),
                body=_convert_term(term.body),
            )
        case SPi():
            return EPi(
                span=term.span,
                binders=tuple(_convert_binder(b) for b in term.binders),
                body=_convert_term(term.body),
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
                        term=_convert_term(arg.term),
                    )
                    named[arg.name] = named_arg
                    named_order.append(named_arg)
                    continue
                if named_seen:
                    raise SurfaceError(
                        "Positional arguments must come before named arguments",
                        arg.term.span,
                    )
                positional.append(_convert_arg(arg))
            return EApp(
                span=term.span,
                fn=_convert_term(term.fn),
                args=tuple(positional),
                named_args=tuple(named_order),
            )
        case SUApp():
            return EUApp(
                span=term.span,
                head=_convert_term(term.head),
                levels=term.levels,
            )
        case SPartial():
            return EPartial(span=term.span, term=_convert_term(term.term))
        case SLet():
            if term.params:
                raise SurfaceError("Let parameters must be desugared", term.span)
            return ELet(
                span=term.span,
                uparams=term.uparams,
                name=term.name,
                ty=_convert_term(term.ty) if term.ty is not None else None,
                val=_convert_term(term.val),
                body=_convert_term(term.body),
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
                scrutinee=_convert_term(term.scrutinees[0]),
                motive=_convert_term(term.motive) if term.motive is not None else None,
                branches=tuple(_convert_branch(branch) for branch in term.branches),
            )
        case SLetPat():
            raise SurfaceError("Let patterns must be desugared", term.span)
        case SInd():
            return EInd(span=term.span, name=term.name)
        case SCtor():
            return ECtor(span=term.span, name=term.name)
        case SInductiveDef():
            return EInductiveDef(
                span=term.span,
                name=term.name,
                uparams=term.uparams,
                params=tuple(_convert_binder(b) for b in term.params),
                level=_convert_term(term.level),
                ctors=tuple(_convert_ctor_decl(c) for c in term.ctors),
                body=_convert_term(term.body),
            )
        case _:
            raise SurfaceError("Unsupported surface term for elaboration", term.span)


def _convert_binder(binder: SBinder) -> EBinder:
    ty = _convert_term(binder.ty) if binder.ty is not None else None
    return EBinder(
        name=binder.name,
        ty=ty,
        span=binder.span,
        implicit=binder.implicit,
    )


def _convert_arg(arg: SArg) -> EArg:
    return EArg(
        term=_convert_term(arg.term),
        implicit=arg.implicit,
    )


def _convert_branch(branch: SBranch) -> EBranch:
    return EBranch(
        pat=_convert_pat(branch.pat),
        rhs=_convert_term(branch.rhs),
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


def _convert_ctor_decl(ctor: SConstructorDecl) -> EConstructorDecl:
    if ctor.result is None:
        raise SurfaceError("Constructor result must be desugared", ctor.span)
    return EConstructorDecl(
        name=ctor.name,
        fields=tuple(_convert_binder(b) for b in ctor.fields),
        result=_convert_term(ctor.result),
        span=ctor.span,
    )
