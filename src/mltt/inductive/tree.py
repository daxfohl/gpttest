"""Helpers for binary trees with separate leaf and node payload types."""

from __future__ import annotations

from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Term,
    Univ,
    Var,
)

Tree = I(
    name="Tree",
    param_types=(
        Univ(0),  # A : Type
        Univ(0),  # B : Type
    ),
    level=0,
)
LeafCtor = Ctor(
    "Leaf",
    Tree,
    (Var(1),),  # payload : A
)
b = Var(2)
b1 = Var(3)
a = App(Tree, b)
b2 = Var(1)
a1 = App(Tree, b1)
b3 = Var(2)
NodeCtor = Ctor(
    "Node",
    Tree,
    (
        Var(0),  # label : B
        App(a, b2),  # left : Tree A B
        App(a1, b3),  # right : Tree A B
    ),
)
object.__setattr__(Tree, "constructors", (LeafCtor, NodeCtor))


def TreeType(leaf_ty: Term, node_ty: Term) -> App:
    a = App(Tree, leaf_ty)
    return App(a, node_ty)


def Leaf(leaf_ty: Term, node_ty: Term, payload: Term) -> Term:
    a = App(LeafCtor, leaf_ty)
    a1 = App(a, node_ty)
    return App(a1, payload)


def Node(leaf_ty: Term, node_ty: Term, label: Term, left: Term, right: Term) -> Term:
    a = App(NodeCtor, leaf_ty)
    a1 = App(a, node_ty)
    a2 = App(a1, label)
    a3 = App(a2, left)
    return App(a3, right)


def TreeRec(
    P: Term,
    leaf_case: Term,
    node_case: Term,
    tree: Term,
) -> Elim:
    """Recursor for ``Tree leaf_ty node_ty`` using the generic eliminator."""

    return Elim(
        inductive=Tree,
        motive=P,
        cases=(
            leaf_case,
            node_case,
        ),
        scrutinee=tree,
    )
