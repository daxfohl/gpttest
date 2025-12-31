from mltt.inductive import vec, allvec
from mltt.inductive.allvec import (
    AllCons,
    AllNil,
    AllVecAt,
    AllVecElim,
    AllVecType,
)
from mltt.inductive.nat import NatType, Zero, Succ
from mltt.inductive.vec import VecType, Nil
from mltt.kernel.ast import Univ, Var, Pi, App, Term
from mltt.kernel.tel import mk_lams, mk_pis, mk_app


def test_infer_allvec_type_constructor_at_level() -> None:
    expected = mk_pis(
        Univ(1),
        Pi(Var(0), Univ(1)),
        NatType(),
        VecType(Var(2), Var(0), level=1),
        return_ty=Univ(1),
    )
    assert AllVecAt(1).infer_type().type_equal(expected)


def test_elim_allvec_allcons_requires_param_shift_under_fields_for_result_indices() -> (
    None
):
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
        Z_ty,  # Z : Type0            (in body: Z = Var(3))
        Pi(Var(0), Univ(0)),  # F : Z -> Type0       (in body: F = Var(2))
        Var(1),  # z : Z                (in body: z = Var(1))
        App(Var(1), Var(0)),  # px0 : F z            (in body: px0 = Var(0))
        body=mk_allvec_elim_body(),  # we’ll define this below using the Γ vars
    )

    ty = closed.infer_type()
    expected = mk_pis(
        Univ(0),  # Z : Type0
        Pi(Var(0), Univ(0)),  # F : Z -> Type0   (Z is Var(0) here)
        Var(1),  # z : Z            (Z is Var(1) here)
        App(Var(1), Var(0)),  # px0 : F z        (F is Var(1), z is Var(0) here)
        return_ty=NatType(),
    )
    assert ty.type_equal(expected)


def mk_allvec_elim_body(*, recursive: bool = False, nil: bool = False) -> Term:
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
    xs0 = Nil(A)  # Vec.Nil A : Vec A 0
    ih0 = AllNil(A, P)  # AllNil A P : AllVec A P 0 (Nil A)

    scrutinee = AllNil(A, P) if nil else AllCons(A, P, n0, z0, xs0, px0, ih0)

    # ------------------------------------------------------------
    # Motive (constant Nat):
    #   motive : Π n:Nat. Π xs:Vec A n. Π all:AllVec A P n xs. Type0
    # ------------------------------------------------------------
    motive = mk_lams(
        NatType(),  # n : Nat          (n = Var(0))
        VecType(A.shift(1), Var(0)),
        # xs : Vec A n      (xs = Var(0), n = Var(1) after xs binder, but here n is still Var(0) in this arg_ty position)
        AllVecType(A.shift(2), P.shift(2), Var(1), Var(0)),  # all : AllVec A P n xs
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

    ih_ih_ty = mk_app(motive.shift(5), Var(4), Var(2), Var(0))  # motive n xs ih

    all_cons = mk_lams(
        NatType(),  # n : Nat
        A.shift(1),  # x : A
        VecType(A.shift(2), Var(1)),  # xs : Vec A n   (n is Var(1) after x binder)
        App(P.shift(3), Var(1)),  # px : P x       (x is Var(1) after xs binder)
        AllVecType(
            A.shift(4), P.shift(4), Var(3), Var(1)
        ),  # ih : AllVec A P n xs  (n=Var(3), xs=Var(1) after px binder)
        ih_ih_ty,  # ih_ih : motive n xs ih
        body=Succ(Var(0)) if recursive else Zero(),  # Var(0) is ih_ih (the last binder)
    )

    return AllVecElim(motive, all_nil, all_cons, scrutinee)


def test_allvec_elim_body_iota_nil() -> None:
    """
    Under concrete A,P,n,xs,pf for pf=AllNil, the eliminator should iota-reduce
    to the nil case (here assumed Zero()).
    """
    # Build the Γ-closed version of mk_allvec_elim_body
    closed = mk_lams(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # P
        NatType(),  # n
        vec.VecType(Var(2), Var(0)),  # xs
        allvec.AllVecType(Var(3), Var(2), Var(1), Var(0)),  # pf
        body=mk_allvec_elim_body(nil=True),  # <-- changed
    )

    A = NatType()
    P = mk_lams(NatType(), body=Univ(0))
    pf = allvec.AllNil(A, P)
    xs = vec.Nil(A)

    term = mk_app(closed, A, P, Zero(), xs, pf)
    assert term.normalize() == Zero()


def test_allvec_elim_body_iota_cons_uses_ih() -> None:
    """
    Build pf1 = AllCons ... with IH = AllNil ... and check the eliminator
    uses the IH in the cons branch (so result becomes Succ Zero if body is Succ ih).
    """
    closed = mk_lams(
        Univ(0),
        Pi(Var(0), Univ(0)),
        NatType(),
        vec.VecType(Var(2), Var(0)),
        allvec.AllVecType(Var(3), Var(2), Var(1), Var(0)),
        body=mk_allvec_elim_body(recursive=True),
    )

    A = NatType()
    P = mk_lams(NatType(), body=Univ(0))

    pf0 = allvec.AllNil(A, P)
    xs0 = vec.Nil(A)

    x = Zero()
    px = mk_app(P, x)

    pf1 = allvec.AllCons(A, P, Zero(), x, xs0, px, pf0)
    xs1 = vec.Cons(A, Zero(), x, xs0)

    term = mk_app(closed, A, P, Succ(Zero()), xs1, pf1).normalize()
    assert term == Succ(Zero())


def mk_allvec_elim_closed_over_ZFzpx0() -> Term:
    """
    Close mk_allvec_elim_body over Γ = (Z,F,z,px0) so it becomes a closed term:

      Π Z : Type.
      Π F : Π z:Z. Type.
      Π z : Z.
      Π px0 : F z.
      Nat
    """
    return mk_lams(
        Univ(0),  # Z : Type0
        Pi(Var(0), Univ(0)),  # F : Π z:Z. Type0   (Z is Var(0) here)
        Var(1),  # z : Z              (Z is Var(1) here)
        App(Var(1), Var(0)),  # px0 : F z          (F is Var(1), z is Var(0))
        body=mk_allvec_elim_body(),
    )


def test_allvec_elim_body_typechecks() -> None:
    closed = mk_allvec_elim_closed_over_ZFzpx0()

    expected_ty = mk_pis(
        Univ(0),  # Z
        Pi(Var(0), Univ(0)),  # F : Z -> Type
        Var(1),  # z : Z
        App(Var(1), Var(0)),  # px0 : F z
        return_ty=NatType(),
    )
    closed.type_check(expected_ty)


def test_allvec_elim_body_iota_cons_is_zero() -> None:
    closed = mk_allvec_elim_closed_over_ZFzpx0()

    Z = NatType()
    F = mk_lams(Z, body=NatType())  # F : Z -> Type0, returns Nat
    z0 = Zero()
    px0 = Zero()  # px0 : F z0 = Nat, OK

    term = mk_app(closed, Z, F, z0, px0).normalize()
    assert term == Zero()


def test_allvec_elim_body_param_vars_shift_correctly() -> None:
    closed = mk_allvec_elim_closed_over_ZFzpx0()

    # term : Π Z:Type. Π F:(Z->Type). Π z:Z. Π px0:F z. Nat
    term = mk_lams(
        Univ(0),  # Z : Type0
        Pi(Var(0), Univ(0)),  # F : Z -> Type0
        Var(1),  # z : Z
        App(Var(1), Var(0)),  # px0 : F z
        body=mk_app(
            closed,
            Var(3),  # Z
            Var(2),  # F
            Var(1),  # z
            Var(0),  # px0
        ),
    )

    expected_ty = mk_pis(
        Univ(0),
        Pi(Var(0), Univ(0)),
        Var(1),
        App(Var(1), Var(0)),
        return_ty=NatType(),
    )
    term.type_check(expected_ty)
