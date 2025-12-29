"""Boolean type and logical operators."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Term, Univ, Var
from mltt.kernel.telescope import mk_app, mk_lams
from mltt.kernel.ind import Elim, Ctor, Ind

Bool = Ind(name="Bool", level=0)
FalseCtor = Ctor(name="False", inductive=Bool)
TrueCtor = Ctor(name="True", inductive=Bool)
object.__setattr__(Bool, "constructors", (FalseCtor, TrueCtor))


def BoolType() -> Ind:
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


def if_term() -> Term:
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
                        motive=Lam(BoolType(), Var(4)),  # env = [Bool, f, t, b, A]
                        false_case=Var(0),  # f
                        true_case=Var(1),  # t
                        scrutinee=Var(2),  # b
                    ),
                ),
            ),
        ),
    )


def if_(A: Term, b: Term, t: Term, f: Term) -> Term:
    return mk_app(if_term(), A, b, t, f)


def not_term() -> Term:
    """Boolean negation."""

    return mk_lams(
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), True_(), False_(), Var(0)),
    )


def not_(b: Term) -> Term:
    return App(not_term(), b)


def and_term() -> Term:
    """Boolean conjunction."""

    return mk_lams(
        BoolType(),
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), False_(), Var(0), scrutinee=Var(1)),
    )


def and_(lhs: Term, rhs: Term) -> Term:
    return mk_app(and_term(), lhs, rhs)


def or_term() -> Term:
    """Boolean disjunction."""

    return mk_lams(
        BoolType(),
        BoolType(),
        body=BoolRec(Lam(BoolType(), BoolType()), Var(0), True_(), scrutinee=Var(1)),
    )


def or_(lhs: Term, rhs: Term) -> Term:
    return mk_app(or_term(), lhs, rhs)
