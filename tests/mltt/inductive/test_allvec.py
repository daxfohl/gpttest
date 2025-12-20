from mltt.core.ast import Lam, Pi, Univ, Var
from mltt.core.debruijn import mk_lams
from mltt.core.typing import infer_type
from mltt.inductive.all import AllCons, AllNil
from mltt.inductive.allvec import *
from mltt.inductive.list import Nil
from mltt.inductive.nat import NatType, Zero


def test_elim_allvec_allcons_requires_param_shift_under_fields_for_result_indices() -> (
    None
):
    """
    This is an *actual eliminator* test that detects the bug in the 2nd approach
    to instantiating ctor.result_indices.

    Key idea:
      - Choose A_actual and P_actual from Γ (as Vars), so they are *open*.
      - In the AllCons constructor, result_indices[1] includes 'Cons A n x xs'.
      - If you instantiate result_indices without shifting params under the 5 ctor fields,
        then A_actual = Var(0) is interpreted as the *innermost ctor field* (capture),
        not Γ's Z. That breaks the definitional equality needed when type-checking the
        all_cons branch of the eliminator.

    With the correct approach (shift params by m fields, depth_above = m + i),
    the eliminator type-checks.
    With the wrong approach (depth_above=i, no shift), it fails.
    """

    # ---------------------------------------
    # Build a context Γ with:
    #   Z : Type0
    #   F : Z -> Type0
    #
    # We'll write the whole term closed by lambda-abstracting over Γ,
    # so the test is self-contained and infer_type can run on a closed term.
    # ---------------------------------------

    # Under binder Z: Type0, Z is Var(0).
    # Under binder F: Z -> Type0 (after Z), Z is Var(1) and F is Var(0).
    Z_ty = Univ(0)
    F_ty = Lam(
        Var(0), Univ(0)
    )  # Π z:Z. Type0   (as a term, use Pi if you have it; Lam ok for term-building here)

    # We'll construct a body under context (Z, F, ...) below.

    # ---------------------------------------
    # Choose actual parameters for AllVec:
    #   A := Z
    #   P := F
    #
    # In the body (under Z,F), these are:
    #   A = Var(1)  (Z)
    #   P = Var(0)  (F)
    # ---------------------------------------

    A = Var(1)  # Z
    P = Var(0)  # F

    # ---------------------------------------
    # Now build an AllCons scrutinee:
    #   AllCons A P n x xs px ih : AllVec A P (Succ n) (Cons A n x xs)
    #
    # We will keep it very small:
    #   n  := 0
    #   x  := some z : Z (we'll bind it)
    #   xs := Nil A
    #   px := some proof of P x (we'll bind it)
    #   ih := AllNil A P   (works because xs = Nil A, n = 0)
    # ---------------------------------------

    # We'll build this under extra binders:
    #   z : Z
    #   px : P z
    #
    # Under (Z, F, z, px):
    #   Var(0)=px
    #   Var(1)=z
    #   Var(2)=F
    #   Var(3)=Z

    n = Zero()
    z = Var(1)  # z
    xs = Nil(Var(3))  # Nil A, where A = Z = Var(3) here
    px = Var(0)  # px : P z
    ih = AllNil(Var(3), Var(2))  # AllNil A P

    scrutinee = AllCons(Var(3), Var(2), n, z, xs, px, ih)

    # ---------------------------------------
    # Define a motive that returns a constant type, say Nat.
    # For AllVecElim, motive should be:
    #   motive : Π n:Nat. Π xs:Vec A n. AllVec A P n xs -> Type0
    #
    # We'll use constant motive = NatType(), ignoring args.
    # ---------------------------------------

    motive = mk_lams(
        NatType(),  # n
        VecType(
            Var(4), Var(0)
        ),  # xs : Vec A n    (A is Var(4) under Z,F,z,px,n,xs ???)
        AllVecType(
            Var(5), Var(4), Var(1), Var(0)
        ),  # all : AllVec A P n xs  (adjust if your AllVecType expects A,P,n,xs)
        body=NatType(),  # constant Type = Nat
    )

    # ---------------------------------------
    # Provide cases:
    #   all_nil : motive 0 (Nil A) (AllNil A P)  == Nat
    #   all_cons : Π n x xs px ih. motive (Succ n) (Cons A n x xs) (AllCons ...) == Nat
    #
    # Since motive is constant Nat, we can use:
    #   all_nil  = 0
    #   all_cons = λ n x xs px ih. 0
    # ---------------------------------------

    all_nil = Zero()
    all_cons = mk_lams(
        NatType(),  # n
        Var(6),  # x : A
        VecType(Var(7), Var(1)),  # xs : Vec A n
        App(Var(8), Var(2)),  # px : P x   (P is Var(8), x is Var(2) here)
        AllVecType(Var(9), Var(8), Var(3), Var(1)),  # ih : AllVec A P n xs
        body=Zero(),
    )

    elim_term = AllVecElim(motive, all_nil, all_cons, scrutinee)

    # Finally close over Z, F, z, px so the whole thing is closed:
    closed = mk_lams(
        Z_ty,  # Z : Type0
        Pi(
            Var(0), Univ(0)
        ),  # F : Z -> Type0  (use your Pi ctor here; placeholder if needed)
        Var(1),  # z : Z
        App(Var(1), Var(0)),  # px : F z
        body=elim_term,
    )

    # ----------------------------
    # EXPECTATION:
    #
    # With correct result_indices instantiation (shift params under fields),
    # infer_type(closed) should succeed and return Nat.
    #
    # With the wrong (2nd) approach, the all_cons branch check should fail
    # because the scrutinee's computed indices are wrong (A is captured).
    # ----------------------------
    ty = infer_type(closed)
    assert ty == NatType()
