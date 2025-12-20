from mltt.core.ast import Univ, Var, Pi, App, Term
from mltt.core.debruijn import mk_lams, Ctx, mk_pis, mk_app
from mltt.core.typing import infer_type
from mltt.inductive.allvec import AllVecType, AllVecElim, AllNil, AllCons
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.vec import VecType, Nil


def test_elim_allvec_allcons_requires_param_shift_under_fields_for_result_indices() -> None:
    """
    This is an *actual eliminator* test that detects the bug in the 2nd approach
    to instantiating ctor.result_indices.

    Key idea:
      - Choose A_actual and P_actual from Γ (as Vars), so they are *open*.
      - In the AllCons constructor, result_indices[1] includes 'Cons A n x xs'.
      - If you instantiate result_indices without shifting params under the ctor fields,
        then A_actual (a Γ Var) gets captured by a field binder, breaking the definitional
        equality needed when type-checking the all_cons branch.
    """

    # ------------------------------------------------------------
    # Close over Γ = (Z : Type0, F : Π z:Z. Type0, z : Z, px0 : F z)
    # so infer_type can run on a closed term.
    # ------------------------------------------------------------
    Z_ty = Univ(0)

    closed = mk_lams(
        Z_ty,                      # Z : Type0            (in body: Z = Var(3))
        Pi(Var(0), Univ(0)),       # F : Z -> Type0       (in body: F = Var(2))
        Var(1),                    # z : Z                (in body: z = Var(1))
        App(Var(1), Var(0)),       # px0 : F z            (in body: px0 = Var(0))
        body=mk_allvec_elim_body(),  # we’ll define this below using the Γ vars
    )

    ty = infer_type(closed)
    expected = mk_pis(
        Univ(0),              # Z : Type0
        Pi(Var(0), Univ(0)),  # F : Z -> Type0   (Z is Var(0) here)
        Var(1),               # z : Z            (Z is Var(1) here)
        App(Var(1), Var(0)),  # px0 : F z        (F is Var(1), z is Var(0) here)
        return_ty=NatType(),
    )
    assert ty.type_equal(expected, Ctx())



def mk_allvec_elim_body() -> Term:
    """
    Build the eliminator application under Γ = (Z,F,z,px0).

    Under this context:
      px0 = Var(0)
      z   = Var(1)
      F   = Var(2)
      Z   = Var(3)
    """
    # Γ-level “names”
    Z = Var(3)
    F = Var(2)
    z0 = Var(1)
    px0 = Var(0)

    # We choose AllVec parameters from Γ (open):
    A = Z
    P = F

    # ------------------------------------------------------------
    # Build a small scrutinee:
    #   scrutinee = AllCons A P 0 z0 (Nil A) px0 (AllNil A P)
    # ------------------------------------------------------------
    n0 = Zero()
    xs0 = Nil(A)          # Vec.Nil A : Vec A 0
    ih0 = AllNil(A, P)    # AllNil A P : AllVec A P 0 (Nil A)

    scrutinee = AllCons(A, P, n0, z0, xs0, px0, ih0)

    # ------------------------------------------------------------
    # Motive (constant Nat):
    #   motive : Π n:Nat. Π xs:Vec A n. Π all:AllVec A P n xs. Type0
    # ------------------------------------------------------------
    A1 = A.shift(1)  # under binder n
    A2 = A.shift(2)  # under binders n,xs
    P2 = P.shift(2)

    motive = mk_lams(
        NatType(),                         # n : Nat          (n = Var(0))
        VecType(A1, Var(0)),               # xs : Vec A n      (xs = Var(0), n = Var(1) after xs binder, but here n is still Var(0) in this arg_ty position)
        AllVecType(A2, P2, Var(1), Var(0)),# all : AllVec A P n xs
        body=NatType(),
    )

    # ------------------------------------------------------------
    # Cases:
    #   all_nil  : motive 0 (Nil A) (AllNil A P)  == Nat
    #
    #   all_cons : Π n x xs px ih. Π ih_ih : motive n xs ih.
    #              motive (Succ n) (Cons A n x xs) (AllCons ...)
    #
    # Since motive is constant Nat, bodies can just be Zero().
    # ------------------------------------------------------------
    all_nil = Zero()

    # In all_cons, we build binder types using A/P shifted as binders are introduced.

    # After binding n: shift Γ by 1
    A1 = A.shift(1)
    P1 = P.shift(1)

    # After binding n,x: shift Γ by 2
    A2 = A.shift(2)
    P2 = P.shift(2)

    # After binding n,x,xs: shift Γ by 3
    A3 = A.shift(3)
    P3 = P.shift(3)

    # After binding n,x,xs,px: shift Γ by 4
    A4 = A.shift(4)
    P4 = P.shift(4)

    # After binding n,x,xs,px,ih: shift Γ by 5
    # We’ll also need motive under those binders.
    # At that point, the local Vars are:
    #   ih   = Var(0)
    #   px   = Var(1)
    #   xs   = Var(2)
    #   x    = Var(3)
    #   n    = Var(4)
    ih_ih_ty = mk_app(motive.shift(5), Var(4), Var(2), Var(0))  # motive n xs ih

    all_cons = mk_lams(
        NatType(),                     # n : Nat
        A1,                            # x : A
        VecType(A2, Var(1)),           # xs : Vec A n   (n is Var(1) after x binder)
        App(P3, Var(1)),               # px : P x       (x is Var(1) after xs binder)
        AllVecType(A4, P4, Var(3), Var(1)),  # ih : AllVec A P n xs  (n=Var(3), xs=Var(1) after px binder)
        ih_ih_ty,                      # ih_ih : motive n xs ih
        body=Zero(),
    )

    return AllVecElim(motive, all_nil, all_cons, scrutinee)

