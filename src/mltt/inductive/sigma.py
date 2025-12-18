"""Helpers for dependent pairs (Sigma type)."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Lam, Pi, Term, Univ, Var
from ..core.debruijn import shift
from ..core.util import apply_term, nested_pi, nested_lam

Sigma = I(
    name="Sigma",
    param_types=(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # B : A -> Type
    ),
    level=0,
)
PairCtor = Ctor(
    name="Pair",
    inductive=Sigma,
    arg_types=(
        Var(1),  # a : A
        App(Var(1), Var(0)),  # b : B a
    ),
)
object.__setattr__(Sigma, "constructors", (PairCtor,))


def SigmaType(A: Term, B: Term) -> Term:
    return apply_term(Sigma, A, B)


def Pair(A: Term, B: Term, a: Term, b: Term) -> Term:
    return apply_term(PairCtor, A, B, a, b)


def SigmaRec(P: Term, pair_case: Term, pair: Term) -> Elim:
    """Recursor for ``Sigma A B`` using the generic eliminator."""

    return Elim(
        inductive=Sigma,
        motive=P,
        cases=(pair_case,),
        scrutinee=pair,
    )


def fst(A: Term, B: Term, p: Term) -> Term:
    return SigmaRec(
        # P : Sigma A B -> Type, here constant A
        P=Lam(SigmaType(A, B), shift(A, 1)),
        # pair_case : Π a:A. Π b:B a. A  returning a
        pair_case=nested_lam(A, App(shift(B, 1), Var(0)), body=Var(1)),
        pair=p,
    )


def fst_term() -> Term:
    # fst : Π A:Type. Π B:(A->Type). Sigma A B -> A
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type
        SigmaType(Var(1), Var(0)),  # p : Sigma A B
        body=fst(Var(2), Var(1), Var(0)),
    )


def snd(A: Term, B: Term, p: Term) -> Term:
    A1 = shift(A, 1)
    B1 = shift(B, 1)

    return SigmaRec(
        # P p := B (fst p)
        P=Lam(SigmaType(A, B), App(B1, fst(A1, B1, Var(0)))),
        # pair_case : Π a:A. Π b:B a. B (fst (Pair a b))  (and fst (Pair a b) ≡ a)
        pair_case=nested_lam(A, App(shift(B, 1), Var(0)), body=Var(0)),
        pair=p,
    )


def snd_term() -> Term:
    # snd : Π A:Type. Π B:(A->Type). Π p:Sigma A B. B (fst A B p)
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        SigmaType(Var(1), Var(0)),  # p
        body=snd(Var(2), Var(1), Var(0)),
    )


def let_pair_dep_fn(A: Term, B: Term, C: Term, p: Term, f: Term) -> Term:
    return apply_term(f, fst(A, B, p), snd(A, B, p))


def let_pair_dep(A: Term, B: Term, C: Term, p: Term, f_body: Term) -> Term:
    # f_body is in context extended by a then b (so Var(0)=b, Var(1)=a)
    f_fn = nested_lam(A, App(B, Var(0)), body=f_body)  # λ a. λ b. f_body
    return let_pair_dep_fn(A, B, C, p, f_fn)


def let_pair_dep_term() -> Term:
    # let_pair_dep :
    #   Π A:Type. Π B:(A->Type).
    #   Π C:(Π a:A. Π b:B a. Type).
    #   Π p:Sigma A B.
    #   (Π a:A. Π b:B a. C a b) -> C (fst p) (snd p)
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        nested_pi(Var(1), App(Var(1), Var(0)), return_ty=Univ(0)),  # C
        SigmaType(Var(2), Var(1)),  # p : Sigma A B   (ctx [C,B,A])
        nested_pi(
            Var(3), App(Var(3), Var(0)), return_ty=apply_term(Var(3), Var(1), Var(0))
        ),  # f
        body=let_pair_dep_fn(Var(4), Var(3), Var(2), Var(1), Var(0)),
    )


def let_pair_fn(A: Term, B: Term, C: Term, p: Term, f: Term) -> Term:
    return SigmaRec(
        P=Lam(SigmaType(A, B), shift(C, 1)),
        pair_case=nested_lam(
            A,
            App(shift(B, 1), Var(0)),
            body=apply_term(shift(f, 2), Var(1), Var(0)),
        ),
        pair=p,
    )


def let_pair(A: Term, B: Term, C: Term, p: Term, f: Term) -> Term:
    # Treat ``f`` as the body under binders ``a`` and ``b``, then pass the
    # resulting function to ``let_pair_fn``.
    f_fn = nested_lam(A, App(B, Var(0)), body=f)
    return let_pair_fn(A, B, C, p, f_fn)


def let_pair_term() -> Term:
    # let_pair :
    #   Π A:Type0. Π B:(A->Type0). Π C:Type0.
    #   Sigma A B -> (Π a:A. Π b:B a. C) -> C
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type0
        Univ(0),  # C
        SigmaType(Var(2), Var(1)),  # p : Sigma A B
        nested_pi(Var(3), App(Var(3), Var(0)), return_ty=Var(3)),  # f
        body=let_pair_fn(Var(4), Var(3), Var(2), Var(1), Var(0)),
    )
