"""Helpers for dependent pairs (Sigma type)."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Pi, Term, Univ, Var
from mltt.kernel.debruijn import mk_app, mk_pis, mk_lams, Telescope
from mltt.kernel.ind import Elim, Ctor, Ind

Sigma = Ind(
    name="Sigma",
    param_types=Telescope.of(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # B : A -> Type
    ),
    level=0,
)
PairCtor = Ctor(
    name="Pair",
    inductive=Sigma,
    field_schemas=Telescope.of(
        Var(1),  # a : A
        App(Var(1), Var(0)),  # b : B a
    ),
)
object.__setattr__(Sigma, "constructors", (PairCtor,))


def SigmaType(A: Term, B: Term) -> Term:
    return mk_app(Sigma, A, B)


def Pair(A: Term, B: Term, a: Term, b: Term) -> Term:
    return mk_app(PairCtor, A, B, a, b)


def SigmaElim(P: Term, pair_case: Term, pair: Term) -> Elim:
    return Elim(
        inductive=Sigma,
        motive=P,
        cases=(pair_case,),
        scrutinee=pair,
    )


def SigmaRec(A: Term, B: Term, C: Term, pair_case: Term, pair: Term) -> Term:
    # C is constant result type (may have free vars), so shift by 1 under the motive binder
    return SigmaElim(
        P=Lam(SigmaType(A, B), C.shift(1)),
        pair_case=pair_case,
        pair=pair,
    )


def fst(A: Term, B: Term, p: Term) -> Term:
    return SigmaRec(A, B, A, mk_lams(A, App(B.shift(1), Var(0)), body=Var(1)), p)


def fst_term() -> Term:
    # fst : Π A:Type. Π B:(A->Type). Sigma A B -> A
    return mk_lams(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type
        SigmaType(Var(1), Var(0)),  # p : Sigma A B
        body=fst(Var(2), Var(1), Var(0)),
    )


def snd(A: Term, B: Term, p: Term) -> Term:
    A1 = A.shift(1)
    B1 = B.shift(1)

    return SigmaElim(
        # P p := B (fst p)
        P=Lam(SigmaType(A, B), App(B1, fst(A1, B1, Var(0)))),
        # pair_case : Π a:A. Π b:B a. B (fst (Pair a b))  (and fst (Pair a b) ≡ a)
        pair_case=mk_lams(A, App(B.shift(1), Var(0)), body=Var(0)),
        pair=p,
    )


def snd_term() -> Term:
    # snd : Π A:Type. Π B:(A->Type). Π p:Sigma A B. B (fst A B p)
    return mk_lams(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        SigmaType(Var(1), Var(0)),  # p
        body=snd(Var(2), Var(1), Var(0)),
    )


def let_pair_dep_fn(A: Term, B: Term, p: Term, f: Term) -> Term:
    return mk_app(f, fst(A, B, p), snd(A, B, p))


def let_pair_dep(A: Term, B: Term, p: Term, f_body: Term) -> Term:
    # f_body is in context extended by a then b (so Var(0)=b, Var(1)=a)
    f_fn = mk_lams(A, App(B, Var(0)), body=f_body)  # λ a. λ b. f_body
    return let_pair_dep_fn(A, B, p, f_fn)


def let_pair_dep_term() -> Term:
    # let_pair_dep :
    #   Π A:Type. Π B:(A->Type).
    #   Π C:(Π a:A. Π b:B a. Type).
    #   Π p:Sigma A B.
    #   (Π a:A. Π b:B a. C a b) -> C (fst p) (snd p)
    return mk_lams(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        mk_pis(Var(1), App(Var(1), Var(0)), return_ty=Univ(0)),  # C
        SigmaType(Var(2), Var(1)),  # p : Sigma A B   (env [C,B,A])
        mk_pis(
            Var(3), App(Var(3), Var(0)), return_ty=mk_app(Var(3), Var(1), Var(0))
        ),  # f
        body=let_pair_dep_fn(Var(4), Var(3), Var(1), Var(0)),
    )


def let_pair(A: Term, B: Term, p: Term, f: Term) -> Term:
    # Treat ``f`` as the body under binders ``a`` and ``b``, then pass the
    # resulting function to ``let_pair_fn``.
    f_fn = mk_lams(A, App(B, Var(0)), body=f)
    return let_pair_dep_fn(A, B, p, f_fn)


def let_pair_term() -> Term:
    # let_pair :
    #   Π A:Type0. Π B:(A->Type0). Π C:Type0.
    #   Sigma A B -> (Π a:A. Π b:B a. C) -> C
    return mk_lams(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B : A -> Type0
        Univ(0),  # C
        SigmaType(Var(2), Var(1)),  # p : Sigma A B
        mk_pis(Var(3), App(Var(3), Var(0)), return_ty=Var(3)),  # f
        body=let_pair_dep_fn(Var(4), Var(3), Var(1), Var(0)),
    )
