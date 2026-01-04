"""Surface syntax desugaring."""

from __future__ import annotations

import dataclasses

from mltt.surface.sast import (
    Pat,
    SAnn,
    SApp,
    SArg,
    SLet,
    SLetPat,
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
    PatTuple,
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
    match term:
        case SLet():
            new_val = _desugar_equation_rec_in_lambda(term.name, term.val)
            new_val = _desugar_term(new_val)
            new_body = _desugar_term(term.body)
            if new_val is not term.val or new_body is not term.body:
                return dataclasses.replace(term, val=new_val, body=new_body)
            return term
        case SLam():
            new_body = _desugar_term(term.body)
            if new_body is not term.body:
                return SLam(span=term.span, binders=term.binders, body=new_body)
            return term
        case SPi():
            new_body = _desugar_term(term.body)
            if new_body is not term.body:
                return SPi(span=term.span, binders=term.binders, body=new_body)
            return term
        case SApp():
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
        case SPartial():
            new_term = _desugar_term(term.term)
            if new_term is not term.term:
                return SPartial(span=term.span, term=new_term)
            return term
        case SAnn():
            new_term = _desugar_term(term.term)
            if new_term is not term.term:
                return SAnn(span=term.span, term=new_term, ty=term.ty)
            return term
        case SUApp():
            new_head = _desugar_term(term.head)
            if new_head is not term.head:
                return SUApp(span=term.span, head=new_head, levels=term.levels)
            return term
        case SMatch():
            return _desugar_match(term)
        case SLetPat():
            _check_duplicate_binders((SBranch(term.pat, term.body, term.span),))
            new_value = _desugar_term(term.value)
            new_body = _desugar_term(term.body)
            new_pat = _expand_tuple_pat(term.pat)
            if (
                new_value is not term.value
                or new_body is not term.body
                or new_pat is not term.pat
            ):
                return dataclasses.replace(
                    term, value=new_value, body=new_body, pat=new_pat
                )
            return term
        case SInductiveDef():
            new_body = _desugar_term(term.body)
            if new_body is not term.body:
                return dataclasses.replace(term, body=new_body)
            return term
        case _:
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
            return dataclasses.replace(t, body=new_body)
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
                return dataclasses.replace(t, val=new_val, body=new_body)
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


def _expand_tuple_pat(pat: Pat) -> Pat:
    match pat:
        case PatTuple():
            if len(pat.elts) < 2:
                return _expand_tuple_pat(pat.elts[0])
            left = _expand_tuple_pat(pat.elts[0])
            right = _expand_tuple_pat(
                PatTuple(span=pat.span, elts=pat.elts[1:])
                if len(pat.elts) > 2
                else pat.elts[1]
            )
            return PatCtor(span=pat.span, ctor="Pair", args=(left, right))
        case PatCtor():
            return PatCtor(
                span=pat.span,
                ctor=pat.ctor,
                args=tuple(_expand_tuple_pat(arg) for arg in pat.args),
            )
        case _:
            return pat


def _looks_like_ctor(name: str) -> bool:
    return bool(name) and (name[0].isupper() or "." in name)


def _check_duplicate_binders(branches: tuple[SBranch, ...]) -> None:
    for branch in branches:
        names: list[str] = []
        for name in _pat_bindings(branch.pat):
            if name in names:
                raise SurfaceError(f"Duplicate binder {name}", branch.span)
            names.append(name)


def _pat_bindings(pat: Pat) -> list[str]:
    match pat:
        case PatVar():
            if _looks_like_ctor(pat.name):
                return []
            return [pat.name]
        case PatWild():
            return []
        case PatCtor():
            names: list[str] = []
            for arg in pat.args:
                names.extend(_pat_bindings(arg))
            return names
        case PatTuple():
            tuple_names: list[str] = []
            for elt in pat.elts:
                tuple_names.extend(_pat_bindings(elt))
            return tuple_names
        case _:
            return []


def _needs_nested(branches: tuple[SBranch, ...]) -> bool:
    return any(_pat_nested(branch.pat) for branch in branches)


def _pat_nested(pat: Pat) -> bool:
    match pat:
        case PatCtor():
            for arg in pat.args:
                if _pat_nested(arg):
                    return True
                if isinstance(arg, PatCtor):
                    return True
            return False
        case _:
            return False


def _extract_default(branches: tuple[SBranch, ...]) -> SBranch | None:
    if branches and isinstance(branches[-1].pat, PatWild):
        return branches[-1]
    return None


def _expand_tuple_branches(branches: tuple[SBranch, ...]) -> tuple[SBranch, ...]:
    expanded: list[SBranch] = []
    for branch in branches:
        pat = _expand_tuple_pat(branch.pat)
        expanded.append(SBranch(pat, branch.rhs, branch.span))
    return tuple(expanded)


def _compile_branches(
    scrutinee: SurfaceTerm,
    branches: tuple[SBranch, ...],
    fallback: SurfaceTerm,
    counter: list[int],
) -> SurfaceTerm:
    def compile_branch_list(
        branches: tuple[SBranch, ...], default_term: SurfaceTerm
    ) -> SurfaceTerm:
        if not branches:
            return default_term
        head, *rest = branches
        match head.pat:
            case PatWild():
                return head.rhs
            case PatVar():
                if not _looks_like_ctor(head.pat.name):
                    return SLet(
                        span=head.span,
                        uparams=(),
                        name=head.pat.name,
                        ty=None,
                        val=scrutinee,
                        body=head.rhs,
                    )
        next_fallback = _compile_branches(scrutinee, tuple(rest), fallback, counter)
        return _compile_pat(scrutinee, head.pat, head.rhs, next_fallback, counter)

    return compile_branch_list(branches, fallback)


def _compile_pat(
    scrutinee: SurfaceTerm,
    pat: Pat,
    success: SurfaceTerm,
    fallback: SurfaceTerm,
    counter: list[int],
) -> SurfaceTerm:
    pat = _expand_tuple_pat(pat)
    match pat:
        case PatCtor():
            pat_args: list[Pat] = []
            for arg in pat.args:
                if isinstance(arg, PatWild):
                    pat_args.append(PatWild(arg.span))
                elif isinstance(arg, PatVar):
                    pat_args.append(PatVar(arg.span, arg.name))
                elif isinstance(arg, PatCtor):
                    fresh = _fresh_name(counter)
                    pat_args.append(PatVar(arg.span, fresh))
                else:
                    pat_args.append(arg)
            nested_success = success
            for arg, orig in zip(pat_args, pat.args, strict=True):
                if isinstance(orig, PatCtor):
                    if not isinstance(arg, PatVar):
                        raise SurfaceError(
                            "Nested constructor binder missing", orig.span
                        )
                    nested_success = _compile_pat(
                        scrutinee=SVar(span=orig.span, name=arg.name),
                        pat=orig,
                        success=nested_success,
                        fallback=fallback,
                        counter=counter,
                    )
            branch_pat = PatCtor(span=pat.span, ctor=pat.ctor, args=tuple(pat_args))
            branch = SBranch(pat=branch_pat, rhs=nested_success, span=pat.span)
            branches = (branch, SBranch(PatWild(pat.span), fallback, pat.span))
            return SMatch(
                span=pat.span,
                scrutinees=(scrutinee,),
                as_names=(None,),
                motive=None,
                branches=branches,
            )
        case PatVar():
            if _looks_like_ctor(pat.name):
                branch = SBranch(pat, success, pat.span)
                return SMatch(
                    span=pat.span,
                    scrutinees=(scrutinee,),
                    as_names=(None,),
                    motive=None,
                    branches=(branch, SBranch(PatWild(pat.span), fallback, pat.span)),
                )
            return SLet(
                span=pat.span,
                uparams=(),
                name=pat.name,
                ty=None,
                val=scrutinee,
                body=success,
            )
        case PatWild():
            return success
        case _:
            return fallback


def _desugar_match(match: SMatch) -> SurfaceTerm:
    _check_duplicate_binders(match.branches)
    new_scrutinees = tuple(_desugar_term(s) for s in match.scrutinees)
    new_branches = []
    for br in match.branches:
        new_rhs = _desugar_term(br.rhs)
        new_branches.append(SBranch(br.pat, new_rhs, br.span))
    new_motive = _desugar_term(match.motive) if match.motive is not None else None
    rebuilt = SMatch(
        span=match.span,
        scrutinees=new_scrutinees,
        as_names=match.as_names,
        motive=new_motive,
        branches=tuple(new_branches),
    )
    if len(rebuilt.scrutinees) != 1:
        return _desugar_multi(rebuilt)
    expanded = _expand_tuple_branches(rebuilt.branches)
    if _needs_nested(expanded):
        return _compile_nested(rebuilt, rebuilt.scrutinees[0], expanded)
    if expanded != rebuilt.branches:
        return dataclasses.replace(rebuilt, branches=expanded)
    return rebuilt


def _compile_nested(
    match: SMatch, scrutinee: SurfaceTerm, branches: tuple[SBranch, ...]
) -> SurfaceTerm:
    default = _extract_default(branches)
    if default is None:
        raise SurfaceError("Nested patterns require a final '_' branch", match.span)
    counter = [0]
    return _compile_branches(scrutinee, branches[:-1], default.rhs, counter)


def _expand_tuple_in_multi(branches: tuple[SBranch, ...]) -> tuple[SBranch, ...]:
    return tuple(
        SBranch(_expand_tuple_multi_pat(branch.pat), branch.rhs, branch.span)
        for branch in branches
    )


def _expand_tuple_multi_pat(pat: Pat) -> Pat:
    match pat:
        case PatTuple():
            return PatTuple(
                span=pat.span,
                elts=tuple(_expand_tuple_multi_pat(p) for p in pat.elts),
            )
        case PatCtor():
            return PatCtor(
                span=pat.span,
                ctor=pat.ctor,
                args=tuple(_expand_tuple_multi_pat(p) for p in pat.args),
            )
        case _:
            return pat


def _desugar_multi(match: SMatch) -> SurfaceTerm:
    if match.motive is not None or any(n is not None for n in match.as_names):
        raise SurfaceError("Dependent match needs one scrutinee", match.span)
    scrutinees = match.scrutinees
    branches = _expand_tuple_in_multi(match.branches)
    if len(scrutinees) == 1:
        return dataclasses.replace(
            match,
            scrutinees=scrutinees,
            as_names=match.as_names,
            motive=match.motive,
            branches=branches,
        )
    default = _extract_default(branches)
    if default is None:
        raise SurfaceError(
            "Multi-scrutinee match requires a final '_' branch", match.span
        )
    return _compile_multi(scrutinees, branches[:-1], default.rhs)


def _compile_multi(
    scrutinees: tuple[SurfaceTerm, ...],
    branches: tuple[SBranch, ...],
    fallback: SurfaceTerm,
) -> SurfaceTerm:
    if len(scrutinees) == 1:
        counter = [0]
        return _compile_branches(scrutinees[0], branches, fallback, counter)
    scrutinee = scrutinees[0]

    def compile_branch_list(
        branches: tuple[SBranch, ...], default_term: SurfaceTerm
    ) -> SurfaceTerm:
        if not branches:
            return default_term
        head, *rest = branches
        pat = head.pat
        match pat:
            case PatTuple():
                if len(pat.elts) != len(scrutinees):
                    raise SurfaceError("Tuple pattern arity mismatch", head.span)
                first_pat = pat.elts[0]
                rest_pat = (
                    pat.elts[1]
                    if len(pat.elts) == 2
                    else PatTuple(span=pat.span, elts=pat.elts[1:])
                )
            case PatWild():
                first_pat = PatWild(pat.span)
                rest_pat = PatWild(pat.span)
            case _:
                raise SurfaceError(
                    "Multi-scrutinee patterns must be tuple or _", head.span
                )
        inner = _compile_multi(
            scrutinees[1:], (SBranch(rest_pat, head.rhs, head.span),), default_term
        )
        next_default = compile_branch_list(tuple(rest), default_term)
        counter = [0]
        return _compile_pat(scrutinee, first_pat, inner, next_default, counter)

    return compile_branch_list(branches, fallback)


def _fresh_name(counter: list[int]) -> str:
    name = f"_pat{counter[0]}"
    counter[0] += 1
    return name
