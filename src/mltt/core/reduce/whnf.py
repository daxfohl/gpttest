"""Weak head normal form reduction leveraging beta/iota head steps."""

from __future__ import annotations

from ..ast import Term, App, Lam, InductiveElim, InductiveConstructor, InductiveType
from .beta import beta_head_step
from .iota import iota_head_step
from ..debruijn import subst
from ..inductive_utils import decompose_ctor_app, ctor_index, decompose_app, apply_term


def whnf_step(term: Term) -> Term:
    """One small-step using beta or iota head reduction."""
    t1 = beta_head_step(term)
    if t1 != term:
        return t1
    t2 = iota_head_step(term)
    if t2 != term:
        return t2
    return term


def whnf(term: Term) -> Term:
    while True:
        t1 = whnf_step(term)
        if t1 == term:
            return term
        term = t1



def iota_reduce(
    inductive: InductiveType,
    motive: Term,
    cases: list[Term],
    ctor: InductiveConstructor,
    args: tuple[Term, ...],
) -> Term:
    """Compute the iota-reduction of an eliminator on a fully-applied ctor."""
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    index = ctor_index(ctor)
    branch = cases[index]

    ctor_args = args[param_count + index_count :]

    ihs: list[Term] = []
    for arg_term, arg_ty in zip(ctor_args, ctor.arg_types):
        head, head_args = decompose_app(arg_ty)
        if head is ctor.inductive:
            # only works if after substituting param_args and index_args into ctor_arg_types.
            # assert head_args[:param_count] == param_args, f"{arg_ty}: {head_args[:param_count]!r} == {param_args}"
            ih = InductiveElim(
                inductive=ctor.inductive,
                motive=motive,
                cases=cases,
                scrutinee=arg_term,
            )
            ihs.append(ih)


    return apply_term(branch, (*args, *ihs))


def whnf1(term: Term) -> Term:
    match term:
        case App(f, a):
            f_whnf = whnf(f)
            match f_whnf:
                case Lam(body):
                    return whnf(subst(body, a))
                case _:
                    return App(f_whnf, a)

        case InductiveElim(params, motive, cases, scrutinee):
            scrutinee_whnf = whnf(scrutinee)
            match decompose_ctor_app(scrutinee_whnf):
                case (ctor, args):
                    index = ctor_index(ctor)
                    case = cases[index]
                    return whnf(iota_reduce(ctor, case, args))
                case _:
                    # Decomposition terminates in Var or Axiom, etc.
                    return InductiveElim(params, motive, cases, scrutinee_whnf)

        case _:
            return term


__all__ = ["whnf_step", "whnf"]
