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
a = App(b, Tree)
b2 = Var(1)
a1 = App(b1, Tree)
b3 = Var(2)
NodeCtor = Ctor(
    "Node",
    Tree,
    (
        Var(0),  # label : B
        App(b2, a),  # left : Tree A B
        App(b3, a1),  # right : Tree A B
    ),
)
object.__setattr__(Tree, "constructors", (LeafCtor, NodeCtor))


def TreeType(leaf_ty: Term, node_ty: Term) -> App:
    a = App(leaf_ty, Tree)
    return App(node_ty, a)


def Leaf(leaf_ty: Term, node_ty: Term, payload: Term) -> Term:
    a = App(leaf_ty, LeafCtor)
    a1 = App(node_ty, a)
    return App(payload, a1)


def Node(leaf_ty: Term, node_ty: Term, label: Term, left: Term, right: Term) -> Term:
    a = App(leaf_ty, NodeCtor)
    a1 = App(node_ty, a)
    a2 = App(label, a1)
    a3 = App(left, a2)
    return App(right, a3)


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
