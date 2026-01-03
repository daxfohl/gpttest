"""Surface syntax desugaring."""

from __future__ import annotations

from dataclasses import replace

from mltt.surface.sast import (
    SAnn,
    SApp,
    SArg,
    SLet,
    SLam,
    SPi,
    SPartial,
    SUApp,
    SVar,
    SurfaceError,
    SurfaceTerm,
)

from mltt.surface.sast import (
    PatCtor,
    PatVar,
    PatWild,
    SBranch,
    SInductiveDef,
    SMatch,
)


def desugar(term: SurfaceTerm) -> SurfaceTerm:
    """Normalize surface syntax into a simpler core surface form."""
    return _desugar_term(term)


def _desugar_term(term: SurfaceTerm) -> SurfaceTerm:
    if isinstance(term, SLet):
        new_val = _desugar_term(term.val)
        new_val = _desugar_equation_rec_in_lambda(term.name, new_val)
        new_body = _desugar_term(term.body)
        if new_val is not term.val or new_body is not term.body:
            return replace(term, val=new_val, body=new_body)
        return term
    if isinstance(term, SLam):
        new_body = _desugar_term(term.body)
        if new_body is not term.body:
            return SLam(span=term.span, binders=term.binders, body=new_body)
        return term
    if isinstance(term, SPi):
        new_body = _desugar_term(term.body)
        if new_body is not term.body:
            return SPi(span=term.span, binders=term.binders, body=new_body)
        return term
    if isinstance(term, SApp):
        new_fn = _desugar_term(term.fn)
        new_args: list[SArg] = []
        changed = new_fn is not term.fn
        for arg in term.args:
            new_term = _desugar_term(arg.term)
            changed = changed or new_term is not arg.term
            new_args.append(SArg(new_term, implicit=arg.implicit, name=arg.name))
        if changed:
            return SApp(span=term.span, fn=new_fn, args=tuple(new_args))
        return term
    if isinstance(term, SPartial):
        new_term = _desugar_term(term.term)
        if new_term is not term.term:
            return SPartial(span=term.span, term=new_term)
        return term
    if isinstance(term, SAnn):
        new_term = _desugar_term(term.term)
        if new_term is not term.term:
            return SAnn(span=term.span, term=new_term, ty=term.ty)
        return term
    if isinstance(term, SUApp):
        new_head = _desugar_term(term.head)
        if new_head is not term.head:
            return SUApp(span=term.span, head=new_head, levels=term.levels)
        return term
    if isinstance(term, SMatch):
        new_scrutinees = tuple(_desugar_term(s) for s in term.scrutinees)
        new_branches = []
        changed = new_scrutinees != term.scrutinees
        for br in term.branches:
            new_rhs = _desugar_term(br.rhs)
            changed = changed or new_rhs is not br.rhs
            new_branches.append(SBranch(br.pat, new_rhs, br.span))
        new_motive = _desugar_term(term.motive) if term.motive is not None else None
        changed = changed or new_motive is not term.motive
        if changed:
            return SMatch(
                span=term.span,
                scrutinees=new_scrutinees,
                as_names=term.as_names,
                motive=new_motive,
                branches=tuple(new_branches),
            )
        return term
    if isinstance(term, SInductiveDef):
        new_body = _desugar_term(term.body)
        if new_body is not term.body:
            return replace(term, body=new_body)
        return term
    return term


def _desugar_equation_rec_in_lambda(name: str, term: SurfaceTerm) -> SurfaceTerm:
    if not isinstance(term, SLam):
        return term
    binder_names = [binder.name for binder in term.binders]
    if not binder_names:
        return term
    match_term = None
    match_args: list[SArg] = []
    if isinstance(term.body, SMatch):
        match_term = term.body
    else:
        head, args = _decompose_sapp(term.body)
        if isinstance(head, SMatch):
            match_term = head
            match_args = args
    if match_term is None:
        return term
    if len(match_term.scrutinees) != 1:
        return term
    scrutinee = match_term.scrutinees[0]
    if not isinstance(scrutinee, SVar):
        return term
    if scrutinee.name not in binder_names:
        return term
    scrut_index = binder_names.index(scrutinee.name)
    used_any = False
    new_branches: list[SBranch] = []
    for branch in match_term.branches:
        pat = branch.pat
        if not isinstance(pat, PatCtor):
            new_branches.append(branch)
            continue
        ih_name = _fresh_ih_name(binder_names, pat)
        new_rhs, used_var = _replace_recursive_call(
            name,
            branch.rhs,
            binder_names,
            scrut_index,
            ih_name,
            match_args,
        )
        if used_var is None:
            new_branches.append(branch)
            continue
        used_any = True
        new_args = pat.args + (PatVar(name=ih_name, span=pat.span),)
        new_pat = PatCtor(span=pat.span, ctor=pat.ctor, args=new_args)
        new_branches.append(SBranch(pat=new_pat, rhs=new_rhs, span=branch.span))
    if not used_any:
        return term
    new_match = SMatch(
        span=match_term.span,
        scrutinees=match_term.scrutinees,
        as_names=match_term.as_names,
        motive=match_term.motive,
        branches=tuple(new_branches),
    )
    if match_args:
        rebuilt = SApp(span=term.body.span, fn=new_match, args=tuple(match_args))
        return SLam(span=term.span, binders=term.binders, body=rebuilt)
    return SLam(span=term.span, binders=term.binders, body=new_match)


def _fresh_ih_name(binder_names: list[str], pat: PatCtor) -> str:
    existing = {name for name in binder_names if name != "_"}
    for arg in pat.args:
        if isinstance(arg, PatVar):
            existing.add(arg.name)
    if "ih" not in existing:
        return "ih"
    index = 1
    while f"ih{index}" in existing:
        index += 1
    return f"ih{index}"


def _replace_recursive_call(
    name: str,
    term: SurfaceTerm,
    binder_names: list[str],
    scrut_index: int,
    ih_name: str,
    ih_args: list[SArg],
) -> tuple[SurfaceTerm, str | None]:
    used_var: str | None = None

    def decompose_app(app: SurfaceTerm) -> tuple[SurfaceTerm, list[SArg]]:
        if isinstance(app, SApp):
            head, args = decompose_app(app.fn)
            return head, args + list(app.args)
        return app, []

    def replace_inductive(t: SInductiveDef) -> SInductiveDef:
        new_body = replace_term(t.body)
        if new_body is not t.body:
            return replace(t, body=new_body)
        return t

    def is_recursive_call(t: SurfaceTerm) -> str | None:
        head, args = decompose_app(t)
        if not isinstance(head, SVar) or head.name != name or not args:
            return None
        if any(arg.implicit for arg in args):
            return None
        if len(args) != len(binder_names):
            return None
        candidate: str | None = None
        for idx, (arg, binder) in enumerate(zip(args, binder_names, strict=True)):
            if idx == scrut_index:
                if not isinstance(arg.term, SVar):
                    return None
                candidate = arg.term.name
            else:
                if not isinstance(arg.term, SVar) or arg.term.name != binder:
                    return None
        return candidate

    def replace_term(t: SurfaceTerm) -> SurfaceTerm:
        nonlocal used_var
        candidate = is_recursive_call(t)
        if candidate is not None:
            if used_var is None:
                used_var = candidate
            elif used_var != candidate:
                raise SurfaceError(
                    "Recursive call uses multiple scrutinee vars", t.span
                )
            if ih_args:
                return SApp(
                    span=t.span,
                    fn=SVar(span=t.span, name=ih_name),
                    args=tuple(
                        SArg(term=arg.term, implicit=arg.implicit, name=arg.name)
                        for arg in ih_args
                    ),
                )
            return SVar(span=t.span, name=ih_name)
        if isinstance(t, SApp):
            new_fn = replace_term(t.fn)
            new_args: list[SArg] = []
            changed = new_fn is not t.fn
            for arg in t.args:
                new_term = replace_term(arg.term)
                changed = changed or new_term is not arg.term
                new_args.append(SArg(new_term, implicit=arg.implicit, name=arg.name))
            if changed:
                return SApp(span=t.span, fn=new_fn, args=tuple(new_args))
            return t
        if isinstance(t, SPartial):
            new_term = replace_term(t.term)
            if new_term is not t.term:
                return SPartial(span=t.span, term=new_term)
            return t
        if isinstance(t, SLam):
            new_body = replace_term(t.body)
            if new_body is not t.body:
                return SLam(span=t.span, binders=t.binders, body=new_body)
            return t
        if isinstance(t, SPi):
            new_body = replace_term(t.body)
            if new_body is not t.body:
                return SPi(span=t.span, binders=t.binders, body=new_body)
            return t
        if isinstance(t, SLet):
            new_val = replace_term(t.val)
            new_body = replace_term(t.body)
            if new_val is not t.val or new_body is not t.body:
                return replace(t, val=new_val, body=new_body)
            return t
        if isinstance(t, SAnn):
            new_term = replace_term(t.term)
            if new_term is not t.term:
                return SAnn(span=t.span, term=new_term, ty=t.ty)
            return t
        if isinstance(t, SUApp):
            new_head = replace_term(t.head)
            if new_head is not t.head:
                return SUApp(span=t.span, head=new_head, levels=t.levels)
            return t
        if isinstance(t, SMatch):
            new_scrutinees = tuple(replace_term(s) for s in t.scrutinees)
            new_branches = []
            changed = new_scrutinees != t.scrutinees
            for br in t.branches:
                new_rhs = replace_term(br.rhs)
                changed = changed or new_rhs is not br.rhs
                new_branches.append(SBranch(br.pat, new_rhs, br.span))
            new_motive = replace_term(t.motive) if t.motive is not None else None
            changed = changed or new_motive is not t.motive
            if changed:
                return SMatch(
                    span=t.span,
                    scrutinees=new_scrutinees,
                    as_names=t.as_names,
                    motive=new_motive,
                    branches=tuple(new_branches),
                )
            return t
        if isinstance(t, SInductiveDef):
            return replace_inductive(t)
        return t

    replaced = replace_term(term)
    return replaced, used_var


def _decompose_sapp(term: SurfaceTerm) -> tuple[SurfaceTerm, list[SArg]]:
    if isinstance(term, SApp):
        head, args = _decompose_sapp(term.fn)
        return head, args + list(term.args)
    return term, []
