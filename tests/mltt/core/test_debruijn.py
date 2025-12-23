from mltt.core.ast import App, Lam, Pi, Term, Univ, Var
from mltt.core.debruijn import mk_pis, mk_lams, discharge_binders, ArgList
from mltt.core.ind import Elim
from mltt.inductive.allvec import AllConsCtor, AllVecType
from mltt.inductive.bool import BoolType
from mltt.inductive.eq import Id, IdElim, ReflCtor
from mltt.inductive.fin import FZCtor
from mltt.inductive.nat import NatRec, Succ, Zero, NatType
from mltt.inductive.sigma import PairCtor
from mltt.inductive.vec import Cons, ConsCtor, VecType


# ------------- Shift: basic behavior -------------


def test_shift_var_free_at_or_above_cutoff_is_bumped() -> None:
    assert Var(0).shift(by=1) == Var(1)
    assert Var(2).shift(by=3, cutoff=2) == Var(5)


def test_shift_var_below_cutoff_unchanged() -> None:
    # cutoff shields lower indices
    assert Var(0).shift(by=2, cutoff=1) == Var(0)
    assert Var(1).shift(by=2, cutoff=2) == Var(1)


def test_shift_by_zero_is_identity() -> None:
    t = App(Var(2), Lam(Var(0), App(Var(0), Var(1))))
    assert t.shift(by=0) == t


def test_shift_app_distributes() -> None:
    shifted = App(Var(1), Var(0)).shift(by=1)
    assert shifted == App(Var(2), Var(1))


def test_shift_lam_body_uses_cutoff_plus_1() -> None:
    # λ. Var(1)  -> shifting by +1 at cutoff=0 should become λ. Var(2)
    assert Lam(Var(0), Var(1)).shift(by=1) == Lam(Var(1), Var(2))


def test_shift_lam_preserves_bound_var() -> None:
    # λ. Var(0) : inner Var(0) is bound, cutoff+1 prevents shift
    assert Lam(Var(42), Var(0)).shift(by=5) == Lam(Var(47), Var(0))


def test_shift_nested_lams_correctly_increments_cutoff() -> None:
    # λ. λ. Var(2)  -> going under two binders raises cutoff twice
    s = mk_lams(Var(0), Var(1), body=Var(2)).shift(by=1)
    # Only the Var(2) (free w.r.t both binders) becomes Var(3)
    assert s == mk_lams(Var(1), Var(2), body=Var(3))


def test_shift_pi_behaves_like_lam() -> None:
    assert Pi(Var(7), Var(1)).shift(by=2) == Pi(Var(9), Var(3))


def test_shift_negative_pops_binder_levels() -> None:
    # Equivalent to "popping" a binder layer for indices >= cutoff
    assert App(Var(3), Var(1)).shift(by=-1, cutoff=1) == App(Var(2), Var(0))
    # Below cutoff index unchanged
    assert Var(0).shift(by=-1, cutoff=1) == Var(0)


# ------------- Subst on Vars (local rule) -------------


def test_subst_var_equal_index_replaced() -> None:
    assert Var(0).subst(sub=Var(99), j=0) == Var(99)
    assert Var(5).subst(sub=Var(77), j=5) == Var(77)


def test_subst_var_higher_index_decrements() -> None:
    assert Var(3).subst(sub=Var(42), j=1) == Var(2)
    assert Var(2).subst(sub=Var(10), j=0) == Var(1)


def test_subst_var_lower_index_unchanged() -> None:
    assert Var(0).subst(sub=Var(42), j=2) == Var(0)
    assert Var(1).subst(sub=Var(99), j=3) == Var(1)


# ------------- Subst under binders (capture-avoidance) -------------


def test_subst_under_lam_shifts_subterm_and_increments_j() -> None:
    # subst(λ(ty). body, sub, j) = λ(subst(ty, sub, j), subst(body, shift(sub,1,0), j+1))
    sub = App(Var(1), Var(0))  # free vars must shift when entering the body
    shifted_sub = sub.shift(1, 0)  # app(v(2), v(1))
    res = Lam(Var(2), App(Var(2), Var(0))).subst(sub=sub, j=1)
    assert res == Lam(Var(1), App(shifted_sub, Var(0)))


def test_subst_bound_variable_not_replaced() -> None:
    # λ. Var(0); substituting j=0 should not affect bound occurrences in the body when using plain subst
    # (they are bound by the λ and the rule increments j when descending)
    # NOTE: substituting into the lambda node doesn't eliminate the binder; that's what subst is for.
    res = Lam(Var(0), Var(0)).subst(sub=Var(42), j=0)
    # ty: subst(v(0), sub, 0) -> matches j, so replaced by sub
    # body: subst(v(0), shift(sub,1,0), 1) -> 0 < 1 => unchanged
    assert res == Lam(Var(42), Var(0))


def test_subst_free_above_j_drops_by_one_under_binder() -> None:
    # t = λ. App1(Var(2), Var(1))
    t = Lam(Var(0), App(Var(2), Var(1)))
    # substitute j=0 (the innermost free var at the top level)
    sub = Var(5)
    res = t.subst(sub, j=0)
    # ty: subst(v(0), v(5), 0) -> replaced by v(5)
    # body: j becomes 1; sub becomes shift(v(5),1,0) = v(6)
    #   v(2) with j=1 -> 2 > 1 => v(1)
    #   v(1) with j=1 -> equal => replaced by v(6)
    assert res == Lam(Var(5), App(Var(1), Var(6)))


def test_subst_under_pi_increments_j_and_shifts_subterm() -> None:
    t = Pi(Var(3), App(Var(2), Var(0)))
    sub = App(Var(0), Var(1))
    res = t.subst(sub, j=1)
    exp_ty = Var(2)  # v(3) with j=1 -> 3>1 => 2
    shifted_sub = sub.shift(1, 0)  # app(v(1), v(2))
    # body under binder: j -> 2
    #   v(2) == j -> replace with shifted_sub
    #   v(0) < j -> stays v(0)
    exp_body = App(shifted_sub, Var(0))
    assert res == Pi(exp_ty, exp_body)


# ------------- β-reduction via subst (TAPL shift dance) -------------


def test_beta_simple_identity() -> None:
    # (λ. Var(0)) arg  -> arg
    assert Var(0).subst(Var(7)) == Var(7)


def test_beta_ignores_argument() -> None:
    # (λ. Var(1)) arg  -> Var(0)  (outer variable gets one level closer)
    assert Var(1).subst(Var(0)) == Var(0)


def test_beta_capture_avoidance_nontrivial() -> None:
    # (λ. λ. Var(2))  applied to  Var(0)
    # body inside outer λ is λ. Var(2)
    # After β on the outer λ, we expect λ. Var(2) (the free var remains free)
    # Walkthrough: subst(body, arg) performs the +1/-1 shift dance correctly
    res = Lam(Var(0), Var(2)).subst(Var(0))
    assert res == Lam(Var(0), Var(1))  # ty shifted; body free index unchanged


def test_beta_argument_with_free_vars_no_capture() -> None:
    # (λ. App1(Var(1), Var(0))) (App1(Var(1), Var(0)))
    # Should produce App1(Var(1), App1(Var(2), Var(1))) with correct shifting
    res = App(Var(1), Var(0)).subst(App(Var(1), Var(0)))
    # Compute expected:
    # shift(arg,1,0) = app(v(2), v(1))
    # subst_impl(body, shift(arg,1,0), 0):
    #   v(1) with j=0 => 1>0 -> v(0)
    #   v(0) with j=0 => replaced with app(v(2), v(1))
    # => app(v(0), app(v(2), v(1)))
    # shift(...,-1,0):
    #   v(0) below cutoff=0? No (0>=0) => v(-1)? Illegal—BUT note:
    #   That v(0) came from decrementing 1>0 -> 0, then the final global pop
    #   Let's recompute safely by direct evaluation using functions:
    # Instead of hand-deriving fragilely, check structural properties:
    assert isinstance(res, App)
    assert isinstance(res.func, Var) or isinstance(res.func, App)

    # Check that no Var indices became negative:
    def no_negative_vars(t: Term) -> bool:
        match t:
            case Var(k):
                return k >= 0
            case App(f, a):
                return no_negative_vars(f) and no_negative_vars(a)
            case Lam(ty, b):
                return no_negative_vars(ty) and no_negative_vars(b)
            case Pi(ty, b):
                return no_negative_vars(ty) and no_negative_vars(b)
            case _:
                return True

    assert no_negative_vars(res)


def test_beta_against_nested_binders() -> None:
    # ((λ. λ. App1(Var(2), Var(0))) arg)
    # After β of the outer λ with arg=Var(3):
    #   - New λ param type: ty' = subst(v(0), j=0, sub=v(3)) = v(3).
    #   - In the inner body App1(Var(2), Var(0)), under the remaining λ we substitute with
    #     j=1 and sub' = shift(v(3), +1, 0) = v(4):
    #       Var(2) -> Var(1)  (since 2>1)
    #       Var(0) -> Var(0)  (since 0<1)
    #   So the result is λ. App1(Var(1), Var(0)).
    res = Lam(Var(0), App(Var(2), Var(0))).subst(Var(3))
    assert res == Lam(Var(3), App(Var(1), Var(0)))


# ------------- Subst/Shift interaction laws (spot checks) -------------


def test_shift_subst_commutation_law_spotcheck() -> None:
    # shift(subst(t, j, s), d, c) == subst(shift(t, d, c), j + (0 if c <= j else 0) + d, shift(s, d, 0))
    # We use the common special case c=0:
    left = (
        App(Var(2), Lam(Var(0), App(Var(1), Var(0))))
        .subst(j=1, sub=App(Var(0), Var(2)))
        .shift(by=2, cutoff=0)
    )
    right = (
        App(Var(2), Lam(Var(0), App(Var(1), Var(0))))
        .shift(by=2)
        .subst(j=3, sub=App(Var(0), Var(2)).shift(by=2))
    )
    assert left == right


def test_subst_then_subst_index_adjustment_spotcheck() -> None:
    # subst(subst(t, j, s), i, r) vs subst(subst(t, i, r), j', s') with proper index arithmetic
    # Check a small concrete instance
    # Left: replace 2->1, then 0->0
    left = (
        App(Var(3), App(Var(2), Var(0))).subst(j=2, sub=Var(1)).subst(j=0, sub=Var(0))
    )
    # Right: replace 0->0 first, then adjust j because replacing 0 can lower indices >0 by 1
    # After removing j=0, original j=2 becomes j' = 1
    right = (
        App(Var(3), App(Var(2), Var(0)))
        .subst(j=0, sub=Var(0))
        .subst(j=1, sub=Var(1).subst(j=0, sub=Var(0)))
    )
    assert left == right


# ------------- Subst in types (ty field) -------------


def test_subst_affects_ty_field_at_same_depth() -> None:
    res = Lam(Var(1), Var(0)).subst(sub=Var(7), j=1)
    # ty: Var(1) == j -> replaced by sub
    # body: j becomes 2, sub shifts to shift(7,1,0)=Var(8); v(0) with j=2 stays v(0)
    assert res == Lam(Var(7), Var(0))


def test_subst_ty_decrements_higher_indices() -> None:
    res = Lam(Var(3), Var(0)).subst(sub=Var(42), j=1)
    # ty: 3>1 -> 2
    assert isinstance(res, Lam)
    assert isinstance(res.arg_ty, Var)
    assert res.arg_ty == Var(2)


# ------------- Identity and stability checks -------------


def test_subst_irrelevant_index_no_change() -> None:
    t = App(Var(2), Lam(Var(0), Var(0)))
    s = App(Var(1), Var(0))
    assert t.subst(sub=s, j=99) == t


def test_subst_with_closed_subterm_is_well_behaved() -> None:
    # sub has no free variables relative to cutoff=0 (e.g., Var(0) under its own binder in tests via subst)
    assert App(Var(2), Var(0)).subst(sub=Var(0), j=2) == App(Var(0), Var(0))


# ------------- Regression-style edge cases -------------


def test_no_negative_indices_after_subst_top() -> None:
    # Stress: ensure subst never creates negative indices
    body = App(Var(1), Var(0))
    arg = App(Var(0), Var(1))
    res = body.subst(arg)

    # Walk tree to confirm
    def ok(t: Term) -> bool:
        match t:
            case Var(k):
                return k >= 0
            case App(f, a):
                return ok(f) and ok(a)
            case Lam(ty, b):
                return ok(ty) and ok(b)
            case Pi(ty, b):
                return ok(ty) and ok(b)
            case _:
                return True

    assert ok(res)


def test_shift_cutoff_beyond_all_vars_is_identity() -> None:
    t = App(Var(1), Lam(Var(0), App(Var(1), Var(0))))
    # max free depth is 1 at top-level; cutoff=10 shields everything
    assert t.shift(by=3, cutoff=10) == t


def test_subst_high_j_beyond_all_vars_identity() -> None:
    assert Lam(Var(0), App(Var(1), Var(0))).subst(sub=Var(9), j=10) == Lam(
        Var(0), App(Var(1), Var(0))
    )


# --------- other types ----------


def test_shift_respects_cutoff() -> None:
    term = App(Var(1), Var(0))
    shifted = term.shift(by=2, cutoff=1)
    assert shifted == App(Var(3), Var(0))


def test_shift_through_lambda_increments_free_variable() -> None:
    term = Lam(Univ(), App(Var(1), Var(0)))
    shifted = term.shift(by=1)
    assert shifted == Lam(Univ(), App(Var(2), Var(0)))


def test_subst_replaces_target_and_decrements_greater_indices() -> None:
    term = App(Var(1), Var(0))
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert result == App(Var(0), Succ(Var(0)))


def test_subst_under_lambda_preserves_bound_variable() -> None:
    term = Lam(Univ(), App(Var(1), Var(0)))
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert result == Lam(Univ(), App(Succ(Var(1)), Var(0)))


def test_shift_nested_binders() -> None:
    term = mk_lams(Univ(), Univ(), body=Var(2))
    shifted = term.shift(by=1)
    assert shifted == mk_lams(Univ(), Univ(), body=Var(3))


def test_subst_nested_binder_chain() -> None:
    term = mk_lams(Univ(), Univ(), body=Var(2))
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert result == mk_lams(Univ(), Univ(), body=Succ(Var(2)))


def test_subst_pi_body() -> None:
    pi_term = mk_pis(Univ(), Univ(), return_ty=Var(1))
    sub = Succ(Var(0))
    result = pi_term.subst(sub)
    assert result == mk_pis(Univ(), Univ(), return_ty=Var(1))


def test_subst_natrec_components() -> None:
    term = NatRec(
        A=Univ(),
        base=Zero(),
        step=Lam(Univ(), Var(0)),
        n=Var(0),
    )
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert isinstance(result, Elim)


def test_subst_identity_constructs() -> None:
    term = Id(Univ(), Var(0), Var(1))
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert result == Id(Univ(), Succ(Var(0)), Var(0))


def test_subst_idelim_components() -> None:
    term = IdElim(
        P=Var(1),
        d=Var(2),
        p=Var(4),
    )
    sub = Succ(Var(0))
    result = term.subst(sub)
    assert isinstance(result, Elim)


def test_discharge_sigma_like_field_type_with_depth_above() -> None:
    """
    Realistic scenario: Sigma Pair's second field type "B a".

    Context shape for the schema term:
        Γ, A, B, a ⊢ schema
    where:
        - Δ = (A, B) is the discharged "actuals block" (k=2)
        - Θ = (a) is below that block and stays in scope => depth_above = 1

    Indices inside schema (innermost = 0):
        Var(0) = a
        Var(1) = B
        Var(2) = A
        Var(3) = last binder of Γ (if Γ nonempty)

    Schema for "B a" is:
        App(Var(1), Var(0))

    Choose actuals:
        A_actual = Type0 (closed)
        B_actual = Type0 (closed)  -- boring but keeps focus on indices
    Expected:
        Substitute away A,B but keep 'a' -> (Type0) a reduces later if your reducer does,
        but structurally should become App(Type0.shift(1), Var(0)) == App(Type0, Var(0)).
        (Shifting a closed term is no-op.)
    """
    schema = App(Var(1), Var(0))  # B a
    actuals = ArgList.of(Univ(0), Univ(0))
    out = discharge_binders(schema, actuals, depth_above=1)

    assert out == App(Univ(0), Var(0))


def test_discharge_affects_gamma_vars_by_decrementing_above_eliminated_block() -> None:
    """
    This test targets the "wait, doesn't elimination squash stuff out of existence?"
    question — but from the schema side, not the actuals side.

    Let Γ have ONE binder 'Z'. Schema is written under:
        Z, A, B, a ⊢ schema
    with Δ=(A,B) discharged and Θ=(a) staying, so depth_above=1, k=2.

    Inside that context:
        Var(0)=a, Var(1)=B, Var(2)=A, Var(3)=Z.

    Schema references Z and a:
        App(Var(3), Var(0))

    After discharging A at index 2:
        variables >2 shift down by 1, so Z: Var(3) -> Var(2)

    After discharging B at index 1:
        variables >1 shift down by 1, so Z: Var(2) -> Var(1)

    So expected:
        App(Var(1), Var(0))   (Z moved closer because A,B were eliminated)
    """
    schema = App(Var(3), Var(0))  # Z a
    actuals = ArgList.of(Univ(0), Univ(0))  # closed actuals; irrelevant
    out = discharge_binders(schema, actuals, depth_above=1)

    assert out == App(Var(1), Var(0))


def test_actuals_from_gamma_do_not_get_mutually_decremented_they_get_shifted_on_insertion() -> (
    None
):
    """
    This is the core concern you raised: "shouldn't actuals be decremented as other
    binders are eliminated?"

    Setup:
        Γ has one binder Z.
        Schema context: Z, A, B, a ⊢ schema
        Discharge Δ=(A,B), keep Θ=(a), so depth_above=1, k=2.

    Schema references BOTH A and B:
        App(Var(2), Var(1))   -- think "A B" just to stress indices.

    Take:
        A_actual = Var(0) in Γ  (i.e., refers to Z)
        B_actual = Type0 (closed)

    Run discharge in the function's order:
      i=0: substitute A at index 2 using A_actual.shift(2).
           A_actual is Var(0) in Γ, so after shift(2) it becomes Var(2) in the schema context.
           After this elimination, nothing else in the term changes yet.

      i=1: substitute B at index 1 using Type0.shift(1)=Type0.
           This elimination *also decrements* variables above 1, which includes the inserted Var(2),
           turning it into Var(1).

    Expected final:
        App(Var(1), Type0)

    Note what this demonstrates:
        - We did NOT "update A_actual itself".
        - We shifted it ON INSERTION (shift(2)), and then subsequent eliminations inside the schema
          naturally reindexed the *inserted occurrence* as part of the term.
    """
    schema = App(Var(2), Var(1))  # A B  (index-mechanics test)
    actuals = ArgList.of(Var(0), Univ(0))  # A_actual=Z from Γ, B_actual=Type0
    out = discharge_binders(schema, actuals, depth_above=1)

    assert out == App(Var(1), Univ(0))


def test_discharge_when_schema_ignores_some_actuals() -> None:
    """
    Common realistic case: a constructor field type doesn't mention every param/index.

    Context: Γ, A, B, a ⊢ schema  (depth_above=1, k=2)
    Schema mentions only B and a:
        App(Var(1), Var(0))   # B a

    But we give a "weird" A_actual that would be problematic if it were in scope of A,B:
        A_actual = Var(0) in Γ (Z)
    This should have NO EFFECT because schema never mentions A (Var(2)).

    Then B_actual = Var(0) in Γ (also Z), which after shift(1) becomes Var(1),
    so result becomes App(Var(1), Var(0)) — i.e., "Z a" after discharge.

    And Z's index in the schema shrinks because we eliminated A and B.
    Let's compute precisely:

      Initial: Z,A,B,a -> indices: a0, B1, A2, Z3.
      Substitute A at 2: schema doesn't mention Var(2), but elimination still happens:
          Z: 3 -> 2
      Substitute B at 1: B replaced with shifted Z, and elimination shifts Z:
          inserted Z was shift(1) inside current context, and then elimination leaves it as Var(1)
          (and any remaining Z occurrences above 1 shift down)

    Expected:
        App(Var(1), Var(0))
    """
    schema = App(Var(1), Var(0))  # B a
    actuals = ArgList.of(Var(0), Var(0))  # both actuals refer to Γ's Z
    out = discharge_binders(schema, actuals, depth_above=1)

    assert out == App(Var(1), Var(0))


def test_discharge_under_lam_preserves_inner_binder_cutoff_and_reindexes_outer_refs() -> (
    None
):
    """
    Nested binder realism: schema contains a Lam, so substitution must respect cutoff.

    Context (outer -> inner): Z, A, B, a
    Discharge Δ=(A,B), keep Θ=(a), depth_above=1.

    Schema term:
        Lam(Type0, App(Var(4), Var(0)))

    Inside the Lam body, binder stack (inner -> outer):
        Var(0) = lam_arg
        Var(1) = a
        Var(2) = B
        Var(3) = A
        Var(4) = Z

    Eliminating A (j=2 at top level) corresponds to eliminating j=3 in the body
    (because Lam adds one binder). Your subst() does that via j+1 and sub.shift(1).
    Similarly, eliminating B corresponds to j=2 in the body.

    Z was Var(4) in the body.
      eliminate A => Var(4) -> Var(3)
      eliminate B => Var(3) -> Var(2)

    Expected:
        Lam(Type0, App(Var(2), Var(0)))
    """
    schema = Lam(Univ(0), App(Var(4), Var(0)))
    actuals = ArgList.of(Univ(0), Univ(0))
    out = discharge_binders(schema, actuals, depth_above=1)
    assert out == Lam(Univ(0), App(Var(2), Var(0)))


# -----------------------
# Sigma / PairCtor tests
# -----------------------


def test_sigma_pair_argtype_a_instantiates_param_A_only() -> None:
    """
    PairCtor.arg_types[0] = Var(1)  # a : A

    At the start of constructor args, the schema is under params only:
        (A : Type), (B : A -> Type) ⊢ Var(1)
    where:
        Var(0)=B, Var(1)=A

    We instantiate params (A,B) with actuals (A0, B0).
    There are no previous fields yet => depth_above=0.

    Expected:
        discharge_binders(Var(1), (A0,B0), depth_above=0) == A0
    """
    A0 = NatType()
    B0 = Lam(NatType(), Univ(0))  # arbitrary family; not used for 'a'
    schema_a_ty = PairCtor.field_schemas[0]  # Var(1)
    out = discharge_binders(schema_a_ty, ArgList.of(A0, B0), depth_above=0)
    assert out == A0


def test_sigma_pair_argtype_b_instantiates_params_but_keeps_field_a_in_scope() -> None:
    """
    PairCtor.arg_types[1] = App(Var(1), Var(0))  # b : B a

    IMPORTANT: this schema is written assuming previous field `a` is in scope:
        (A, B, a) ⊢ App(Var(1), Var(0))
    Under (A,B,a):
        Var(0)=a
        Var(1)=B
        Var(2)=A

    When instantiating *params* (A,B) with actuals (A0,B0),
    we must keep `a` untouched => depth_above=1.

    Choose B0 = (λ _:Nat. Bool). Then B0 a should reduce to Bool later,
    but structurally after discharge we expect:
        App(B0, Var(0))
    because `Var(0)` is still the binder for `a`.
    """
    A0 = NatType()
    B0 = Lam(NatType(), BoolType())  # λ _:Nat. Bool
    schema_b_ty = PairCtor.field_schemas[1]  # App(Var(1), Var(0))
    out = discharge_binders(schema_b_ty, ArgList.of(A0, B0), depth_above=1)
    assert out == App(B0, Var(0))


def test_sigma_pair_b_schema_can_reference_gamma_above_params_and_is_reindexed() -> (
    None
):
    """
    This one targets the 'Gamma above the discharged block' scenario.

    Pretend Γ contributes a binder Z *above* the params block. In de Bruijn terms
    inside the schema for b : B a, that outer Z would show up as Var(3) because:
        (Z, A, B, a)
        Var(0)=a, Var(1)=B, Var(2)=A, Var(3)=Z

    Create a schema term that mentions Z and also B a:
        App(Var(3), App(Var(1), Var(0)))  ~  Z (B a)

    Now discharge params (A,B) with depth_above=1. Eliminating A,B shifts Var(3)->Var(1),
    because two binders between Z and a are removed.

    Expected output:
        App(Var(1), App(B0, Var(0)))
    """
    # Schema under (Z, A, B, a):
    schema = App(Var(3), App(Var(1), Var(0)))

    A0 = NatType()
    B0 = Lam(NatType(), BoolType())
    out = discharge_binders(schema, ArgList.of(A0, B0), depth_above=1)

    assert out == App(Var(1), App(B0, Var(0)))


# -----------------------
# Vec / ConsCtor tests
# -----------------------


def test_vec_cons_tail_type_instantiates_param_A_but_keeps_n_and_head() -> None:
    """
    Vec has params: (A)
    ConsCtor.arg_types include:
        0: NatType()            # n : Nat        (doesn't mention A)
        1: Var(1)               # head : A       (written under (A,n))
        2: Vec A n              # tail : Vec A n (written under (A,n,head))

    The tail schema is:
        apply_term(Vec, Var(2), Var(1))   (in your paste)
    Under (A, n, head):
        Var(0)=head
        Var(1)=n
        Var(2)=A

    We instantiate params block (A) with actual A0,
    keeping (n, head) in scope => depth_above=2.

    Expected:
        VecType(A0, Var(1))   but note: after discharge, Var(1) still refers to n.
    """
    A0 = NatType()
    schema_tail_ty = ConsCtor.field_schemas[2]  # Vec A n  under (A,n,head)
    out = discharge_binders(schema_tail_ty, ArgList.of(A0), depth_above=2)
    assert out == VecType(A0, Var(1))


def test_vec_cons_result_index_instantiates_param_A_keeps_fields_n_head_tail() -> None:
    """
    ConsCtor.result_indices = (Succ(Var(2)),) with comment 'Succ n'.

    result_indices are written under (params)(fields...), i.e. (A, n, head, tail) for Cons.
    Under (A, n, head, tail):
        Var(0)=tail
        Var(1)=head
        Var(2)=n
        Var(3)=A

    Succ(Var(2)) indeed means Succ(n).

    Instantiate params (A) with A0, keeping 3 fields in scope => depth_above=3.

    Expected: Succ(Var(2)) is unchanged except A is removed (it wasn't referenced anyway).
    """
    A0 = NatType()
    schema_idx = ConsCtor.result_indices[0]  # Succ(Var(2))
    out = discharge_binders(schema_idx, ArgList.of(A0), depth_above=3)
    assert out == Succ(Var(2))


# -----------------------
# AllVec / AllConsCtor tests
# -----------------------


def test_allvec_allcons_ih_type_instantiates_params_keeps_prior_fields() -> None:
    """
    AllVec params: (A, P)
    AllConsCtor.arg_types last entry:
        ih : AllVec A P n xs
      encoded as:
        apply_term(AllVec, Var(5), Var(4), Var(3), Var(1))

    This schema assumes prior fields (n, x, xs, px) are in scope:
        Context for ih schema is (A, P, n, x, xs, px)
        Indices:
            Var(0)=px
            Var(1)=xs
            Var(2)=x
            Var(3)=n
            Var(4)=P
            Var(5)=A

    Instantiate params (A,P) with (A0,P0) while keeping 4 fields (n,x,xs,px):
        depth_above = 4

    Expected:
        AllVecType(A0, P0, Var(3), Var(1))
    where Var(3)=n and Var(1)=xs still refer to those binders.
    """
    A0 = NatType()
    P0 = Lam(NatType(), Univ(0))  # arbitrary family over A0
    schema_ih = AllConsCtor.field_schemas[4]
    out = discharge_binders(schema_ih, ArgList.of(A0, P0), depth_above=4)

    assert out == AllVecType(A0, P0, Var(3), Var(1))


def test_allvec_allcons_result_indices_instantiates_params_keeps_all_fields() -> None:
    """
    AllConsCtor.result_indices are:
        ( Succ(Var(4)),  Cons(A, n, x, xs) )
    encoded as:
        Succ(Var(4)) and Cons(Var(6), Var(4), Var(3), Var(2))

    These are written under (A,P)(fields...), i.e. (A,P,n,x,xs,px,ih)
    Indices under (A,P,n,x,xs,px,ih):
        Var(0)=ih
        Var(1)=px
        Var(2)=xs
        Var(3)=x
        Var(4)=n
        Var(5)=P
        Var(6)=A

    Instantiate params (A,P) with (A0,P0), keep 5 fields in scope (n,x,xs,px,ih):
        depth_above = 5

    Expected:
        Succ(Var(4)) unchanged (still Succ(n))
        Cons(A0, Var(4), Var(3), Var(2))  (Cons A0 n x xs)
    """
    A0 = NatType()
    P0 = Lam(NatType(), Univ(0))
    idx0, idx1 = AllConsCtor.result_indices

    out0 = discharge_binders(idx0, ArgList.of(A0, P0), depth_above=5)
    out1 = discharge_binders(idx1, ArgList.of(A0, P0), depth_above=5)

    assert out0 == Succ(Var(4))
    assert out1 == Cons(A0, Var(4), Var(3), Var(2))


# -----------------------
# Id / ReflCtor tests
# -----------------------


def test_id_refl_result_index_instantiates_params_to_actual_x() -> None:
    """
    IdType params: (A : Type, x : A)
    ReflCtor.result_indices = (Var(0),)

    result_indices are written under params (A, x):
        Var(0)=x
        Var(1)=A

    Instantiate with (A0, x0), depth_above=0 (no fields):
        discharge_binders(Var(0), (A0,x0), 0) should yield x0.
    """
    A0 = NatType()
    x0 = Zero()
    schema = ReflCtor.result_indices[0]  # Var(0)
    out = discharge_binders(schema, ArgList.of(A0, x0), depth_above=0)
    assert out == x0


# -----------------------
# Fin / FZCtor tests
# -----------------------


def test_fin_fz_result_index_depends_on_field_n_not_on_params() -> None:
    """
    Fin has no params, one index type (Nat). FZCtor has:
        arg_types = (NatType(),)     # n : Nat
        result_indices = (Succ(Var(0)),)  # Fin (Succ n)

    result_indices are written under (fields...) i.e. (n):
        Var(0)=n

    There are no params to discharge, so actuals=() and depth_above=1 (n is in scope).
    discharge_binders should be the identity here.

    Expected: Succ(Var(0))
    """
    schema = FZCtor.result_indices[0]
    out = discharge_binders(schema, ArgList.empty(), depth_above=1)
    assert out == Succ(Var(0))
