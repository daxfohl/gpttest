"""Helpers for dependent pairs (Sigma type)."""

from __future__ import annotations

from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Pi,
    Term,
    Univ,
    Var,
    Lam,
)
from ..core.inductive_utils import apply_term, nested_lam

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


def fst_term() -> Term:
    # fst : Π A:Type. Π B:(A->Type). Sigma A B -> A
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type
        SigmaType(Var(1), Var(0)),  # p : Sigma A B
        body=SigmaRec(
            # P : Sigma A B -> Type, here constant A
            P=Lam(
                SigmaType(
                    Var(2), Var(1)
                ),  # _ : Sigma A B   (A=Var(2), B=Var(1) in ctx [p,B,A])
                Var(3),  # A (under the motive binder)
            ),
            # pair_case : Π a:A. Π b:B a. A  returning a
            pair_case=Lam(
                Var(2),  # a : A
                Lam(
                    App(Var(2), Var(0)),  # b : B a   (B is Var(2) after binding a)
                    Var(1),  # return a   (in ctx [b,a,p,B,A], a is Var(1))
                ),
            ),
            pair=Var(0),  # p
        ),
    )


def fst(A: Term, B: Term, p: Term) -> Term:
    return apply_term(fst_term(), A, B, p)


def snd_term() -> Term:
    # snd : Π A:Type. Π B:(A->Type). Π p:Sigma A B. B (fst A B p)
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        SigmaType(Var(1), Var(0)),  # p
        body=SigmaRec(
            # P p := B (fst p)
            P=Lam(
                SigmaType(Var(2), Var(1)),  # p' : Sigma A B
                App(
                    Var(2),  # B (in ctx [p',p,B,A], B is Var(2))
                    apply_term(
                        fst_term(),  # closed constant, no shift needed
                        Var(3),  # A
                        Var(2),  # B
                        Var(0),  # p'
                    ),
                ),
            ),
            # pair_case : Π a:A. Π b:B a. B (fst (Pair a b))  (and fst (Pair a b) ≡ a)
            pair_case=Lam(
                Var(2),  # a : A
                Lam(
                    App(Var(2), Var(0)),  # b : B a
                    Var(0),  # return b
                ),
            ),
            pair=Var(0),  # p
        ),
    )


def snd(A: Term, B: Term, p: Term) -> Term:
    return apply_term(snd_term(), A, B, p)


def let_pair_dep_term() -> Term:
    # let_pair_dep :
    #   Π A:Type. Π B:(A->Type).
    #   Π C:(Π a:A. Π b:B a. Type).
    #   Π p:Sigma A B.
    #   (Π a:A. Π b:B a. C a b) -> C (fst p) (snd p)
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        Pi(  # C : Π a:A. Π b:B a. Type
            Var(1),  # a : A    (ctx [B,A])
            Pi(
                App(Var(1), Var(0)),  # b : B a  (ctx [a,B,A], B=Var(1))
                Univ(0),
            ),
        ),
        SigmaType(Var(2), Var(1)),  # p : Sigma A B   (ctx [C,B,A])
        Pi(  # f : Π a:A. Π b:B a. C a b
            Var(3),  # a : A           (ctx [p,C,B,A], A=Var(3))
            Pi(
                App(Var(3), Var(0)),  # b : B a     (ctx [a,p,C,B,A], B=Var(3))
                App(
                    App(Var(3), Var(1)), Var(0)
                ),  # C a b       (ctx [b,a,p,C,B,A], C=Var(3))
            ),
        ),
        body=apply_term(
            Var(0),  # f    (ctx [f,p,C,B,A])
            apply_term(fst_term(), Var(4), Var(3), Var(1)),  # fst A B p
            apply_term(snd_term(), Var(4), Var(3), Var(1)),  # snd A B p
        ),
    )


def let_pair_dep(A: Term, B: Term, C: Term, p: Term, f_body: Term) -> Term:
    # f_body is in context extended by a then b (so Var(0)=b, Var(1)=a)
    f_fn = nested_lam(A, App(B, Var(0)), body=f_body)  # λ a. λ b. f_body
    return apply_term(let_pair_dep_term(), A, B, C, p, f_fn)


def let_pair_term() -> Term:
    # let_pair :
    #   Π A:Type0. Π B:(A->Type0). Π C:Type0.
    #   Sigma A B -> (Π a:A. Π b:B a. C) -> C
    return nested_lam(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type0
        Univ(0),  # C
        SigmaType(Var(2), Var(1)),  # p : Sigma A B
        Pi(Var(3), Pi(App(Var(3), Var(0)), Var(3))),  # f : Π a. Π b. C
        body=SigmaRec(
            P=Lam(SigmaType(Var(4), Var(3)), Var(3)),  # λ _ : Sigma A B. C
            pair_case=nested_lam(
                Var(4),  # a : A
                App(Var(4), Var(0)),  # b : B a
                body=apply_term(Var(2), Var(1), Var(0)),  # f a b
            ),
            pair=Var(1),  # p
        ),
    )


def let_pair(A: Term, B: Term, C: Term, p: Term, f: Term) -> Term:
    # Treat ``f`` as the body under binders ``a`` and ``b``, then pass the
    # resulting function to ``let_pair_term``.
    f_fn = nested_lam(A, App(B, Var(0)), body=f)
    return apply_term(let_pair_term(), A, B, C, p, f_fn)
