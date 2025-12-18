import mltt.inductive.tree as treem
from mltt.core.ast import Lam, Univ, Var
from mltt.core.inductive_utils import nested_lam, nested_pi, apply_term
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.nat import NatType, Succ, Zero, add_terms


def test_infer_tree_type_constructor() -> None:
    expected = nested_pi(Univ(0), Univ(0), return_ty=Univ(0))

    assert infer_type(treem.Tree) == expected


def test_leaf_and_node_type_check() -> None:
    leaf_ty = NatType()
    node_ty = NatType()
    tree_ty = treem.TreeType(leaf_ty, node_ty)

    leaf = treem.Leaf(leaf_ty, node_ty, Zero())
    type_check(leaf, tree_ty)

    left = treem.Leaf(leaf_ty, node_ty, Zero())
    right = treem.Leaf(leaf_ty, node_ty, Succ(Zero()))
    node = treem.Node(leaf_ty, node_ty, Zero(), left, right)
    type_check(node, tree_ty)


def test_treerec_counts_leaves() -> None:
    leaf_ty = NatType()
    node_ty = NatType()
    tree_ty = treem.TreeType(leaf_ty, node_ty)

    P = Lam(tree_ty, NatType())
    leaf_case = Lam(leaf_ty, Succ(Zero()))
    node_case = nested_lam(
        node_ty,
        tree_ty,
        tree_ty,
        apply_term(P, Var(1)),
        apply_term(P, Var(1)),
        body=add_terms(Var(1), Var(0)),
    )
    left = treem.Leaf(leaf_ty, node_ty, Zero())
    right = treem.Leaf(leaf_ty, node_ty, Succ(Zero()))
    tree = treem.Node(
        leaf_ty=leaf_ty, node_ty=node_ty, label=Zero(), left=left, right=right
    )

    count = treem.TreeRec(P, leaf_case, node_case, tree)

    assert normalize(count) == Succ(Succ(Zero()))
    type_check(count, NatType())
