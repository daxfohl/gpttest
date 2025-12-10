"""Helpers for binary trees with separate leaf and node payload types."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Term, Univ, Var
from ..core.inductive_utils import apply_term

Tree = I(
    name="Tree",
    param_types=(
        Univ(0),  # A : Type
        Univ(0),  # B : Type
    ),
    level=0,
)
LeafCtor = Ctor(
    name="Leaf",
    inductive=Tree,
    arg_types=(Var(1),),  # payload : A
)
NodeCtor = Ctor(
    name="Node",
    inductive=Tree,
    arg_types=(
        Var(0),  # label : B
        apply_term(Tree, Var(2), Var(1)),  # left : Tree A B
        apply_term(Tree, Var(3), Var(2)),  # right : Tree A B
    ),
)
object.__setattr__(Tree, "constructors", (LeafCtor, NodeCtor))


def TreeType(leaf_ty: Term, node_ty: Term) -> Term:
    return apply_term(Tree, leaf_ty, node_ty)


def Leaf(leaf_ty: Term, node_ty: Term, payload: Term) -> Term:
    return apply_term(LeafCtor, leaf_ty, node_ty, payload)


def Node(leaf_ty: Term, node_ty: Term, label: Term, left: Term, right: Term) -> Term:
    return apply_term(NodeCtor, leaf_ty, node_ty, label, left, right)


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
