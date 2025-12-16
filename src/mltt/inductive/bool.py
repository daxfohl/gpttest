"""Boolean type and logical operators."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Lam, Term, Univ, Var
from ..core.inductive_utils import apply_term, nested_lam

Bool = I(name="Bool", level=0)
FalseCtor = Ctor(name="False", inductive=Bool)
TrueCtor = Ctor(name="True", inductive=Bool)
object.__setattr__(Bool, "constructors", (FalseCtor, TrueCtor))


def BoolType() -> I:
    return Bool


def False_() -> Term:
    return FalseCtor


def True_() -> Term:
    return TrueCtor


def BoolRec(motive: Term, false_case: Term, true_case: Term, scrutinee: Term) -> Elim:
    """Eliminate Bool by providing branches for ``False`` and ``True``."""

    return Elim(
        inductive=Bool,
        motive=motive,
        cases=(false_case, true_case),
        scrutinee=scrutinee,
    )


def if_() -> Term:
    """
    If : Π A : Type0. Bool -> A -> A -> A
    If A b t f := BoolRec (λ _ : Bool. A) f t b
    """

    return Lam(
        Univ(0),  # A
        Lam(
            BoolType(),  # b
            Lam(
                Var(1),  # t : A   (A is Var(1) here)
                Lam(
                    Var(2),  # f : A (A is Var(2) here)
                    body=BoolRec(
                        motive=Lam(BoolType(), Var(4)),  # ctx = [Bool, f, t, b, A]
                        false_case=Var(0),  # f
                        true_case=Var(1),  # t
                        scrutinee=Var(2),  # b
                    ),
                ),
            ),
        ),
    )


def if_terms(A: Term, b: Term, t: Term, f: Term) -> Term:
    return apply_term(if_(), A, b, t, f)


def not_() -> Term:
    """Boolean negation."""

    return nested_lam(
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), True_(), False_(), Var(0)),
    )


def not_term(b: Term) -> Term:
    return App(not_(), b)


def and_() -> Term:
    """Boolean conjunction."""

    return nested_lam(
        BoolType(),
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), False_(), Var(0), scrutinee=Var(1)),
    )


def and_terms(lhs: Term, rhs: Term) -> Term:
    return apply_term(and_(), lhs, rhs)


def or_() -> Term:
    """Boolean disjunction."""

    return nested_lam(
        BoolType(),
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), Var(0), True_(), scrutinee=Var(1)),
    )


def or_terms(lhs: Term, rhs: Term) -> Term:
    return apply_term(or_(), lhs, rhs)


__all__ = [
    "Bool",
    "BoolType",
    "FalseCtor",
    "TrueCtor",
    "False_",
    "True_",
    "BoolRec",
    "if_",
    "if_terms",
    "not_",
    "not_term",
    "and_",
    "and_terms",
    "or_",
    "or_terms",
]
