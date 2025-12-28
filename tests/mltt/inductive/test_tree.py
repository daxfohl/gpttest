import mltt.inductive.tree as treem
from mltt.kernel.ast import Lam, Univ, Var
from mltt.kernel.debruijn import mk_app, mk_pis, mk_lams
from mltt.inductive.nat import NatType, Succ, Zero, add


def test_infer_tree_type_constructor() -> None:
    expected = mk_pis(Univ(0), Univ(0), return_ty=Univ(0))

    assert treem.Tree.infer_type() == expected


def test_leaf_and_node_type_check() -> None:
    leaf_ty = NatType()
    node_ty = NatType()
    tree_ty = treem.TreeType(leaf_ty, node_ty)

    leaf = treem.Leaf(leaf_ty, node_ty, Zero())
    leaf.type_check(tree_ty)

    left = treem.Leaf(leaf_ty, node_ty, Zero())
    right = treem.Leaf(leaf_ty, node_ty, Succ(Zero()))
    node = treem.Node(leaf_ty, node_ty, Zero(), left, right)
    node.type_check(tree_ty)


def test_treerec_counts_leaves() -> None:
    leaf_ty = NatType()
    node_ty = NatType()
    tree_ty = treem.TreeType(leaf_ty, node_ty)

    P = Lam(tree_ty, NatType())
    leaf_case = Lam(leaf_ty, Succ(Zero()))
    node_case = mk_lams(
        node_ty,
        tree_ty,
        tree_ty,
        mk_app(P, Var(1)),
        mk_app(P, Var(1)),
        body=add(Var(1), Var(0)),
    )
    left = treem.Leaf(leaf_ty, node_ty, Zero())
    right = treem.Leaf(leaf_ty, node_ty, Succ(Zero()))
    tree = treem.Node(
        leaf_ty=leaf_ty, node_ty=node_ty, label=Zero(), left=left, right=right
    )

    count = treem.TreeRec(P, leaf_case, node_case, tree)

    assert count.normalize() == Succ(Succ(Zero()))
    count.type_check(NatType())
