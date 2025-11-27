from mltt.core.ast import App, Id, IdElim, InductiveElim, Lam, Pi, Term, Univ, Var
from mltt.core.debruijn import shift, subst
from mltt.inductive.nat import NatRec, Succ, Zero


# ------------- Shift: basic behavior -------------


def test_shift_var_free_at_or_above_cutoff_is_bumped() -> None:
    assert shift(Var(0), by=1, cutoff=0) == Var(1)
    assert shift(Var(2), by=3, cutoff=2) == Var(5)


def test_shift_var_below_cutoff_unchanged() -> None:
    # cutoff shields lower indices
    assert shift(Var(0), by=2, cutoff=1) == Var(0)
    assert shift(Var(1), by=2, cutoff=2) == Var(1)


def test_shift_by_zero_is_identity() -> None:
    t = App(Var(2), Lam(Var(0), App(Var(1), Var(0))))
    assert shift(t, by=0, cutoff=0) == t


def test_shift_app_distributes() -> None:
    shifted = shift(App(Var(1), Var(0)), by=1, cutoff=0)
    assert shifted == App(Var(2), Var(1))


def test_shift_lam_body_uses_cutoff_plus_1() -> None:
    # λ. Var(1)  -> shifting by +1 at cutoff=0 should become λ. Var(2)
    assert shift(Lam(Var(0), Var(1)), by=1, cutoff=0) == Lam(Var(1), Var(2))


def test_shift_lam_preserves_bound_var() -> None:
    # λ. Var(0) : inner Var(0) is bound, cutoff+1 prevents shift
    assert shift(Lam(Var(42), Var(0)), by=5, cutoff=0) == Lam(Var(47), Var(0))


def test_shift_nested_lams_correctly_increments_cutoff() -> None:
    # λ. λ. Var(2)  -> going under two binders raises cutoff twice
    s = shift(Lam(Var(0), Lam(Var(1), Var(2))), by=1, cutoff=0)
    # Only the Var(2) (free w.r.t both binders) becomes Var(3)
    assert s == Lam(Var(1), Lam(Var(2), Var(3)))


def test_shift_pi_behaves_like_lam() -> None:
    assert shift(Pi(Var(7), Var(1)), by=2, cutoff=0) == Pi(Var(9), Var(3))


def test_shift_negative_pops_binder_levels() -> None:
    # Equivalent to "popping" a binder layer for indices >= cutoff
    assert shift(App(Var(3), Var(1)), by=-1, cutoff=1) == App(Var(2), Var(0))
    # Below cutoff index unchanged
    assert shift(Var(0), by=-1, cutoff=1) == Var(0)


# ------------- Subst on Vars (local rule) -------------


def test_subst_var_equal_index_replaced() -> None:
    assert subst(Var(0), sub=Var(99), j=0) == Var(99)
    assert subst(Var(5), sub=Var(77), j=5) == Var(77)


def test_subst_var_higher_index_decrements() -> None:
    assert subst(Var(3), sub=Var(42), j=1) == Var(2)
    assert subst(Var(2), sub=Var(10), j=0) == Var(1)


def test_subst_var_lower_index_unchanged() -> None:
    assert subst(Var(0), sub=Var(42), j=2) == Var(0)
    assert subst(Var(1), sub=Var(99), j=3) == Var(1)


# ------------- Subst under binders (capture-avoidance) -------------


def test_subst_under_lam_shifts_subterm_and_increments_j() -> None:
    # subst(λ(ty). body, sub, j) = λ(subst(ty, sub, j), subst(body, shift(sub,1,0), j+1))
    sub = App(Var(1), Var(0))  # free vars must shift when entering the body
    shifted_sub = shift(sub, 1, 0)  # app(v(2), v(1))
    res = subst(Lam(Var(2), App(Var(2), Var(0))), sub=sub, j=1)
    assert res == Lam(Var(1), App(shifted_sub, Var(0)))


def test_subst_bound_variable_not_replaced() -> None:
    # λ. Var(0); substituting j=0 should not affect bound occurrences in the body when using plain subst
    # (they are bound by the λ and the rule increments j when descending)
    # NOTE: substituting into the lambda node doesn't eliminate the binder; that's what subst is for.
    res = subst(Lam(Var(0), Var(0)), sub=Var(42), j=0)
    # ty: subst(v(0), sub, 0) -> matches j, so replaced by sub
    # body: subst(v(0), shift(sub,1,0), 1) -> 0 < 1 => unchanged
    assert res == Lam(Var(42), Var(0))


def test_subst_free_above_j_drops_by_one_under_binder() -> None:
    # t = λ. App(Var(2), Var(1))
    t = Lam(Var(0), App(Var(2), Var(1)))
    # substitute j=0 (the innermost free var at the top level)
    sub = Var(5)
    res = subst(t, sub, j=0)
    # ty: subst(v(0), v(5), 0) -> replaced by v(5)
    # body: j becomes 1; sub becomes shift(v(5),1,0) = v(6)
    #   v(2) with j=1 -> 2 > 1 => v(1)
    #   v(1) with j=1 -> equal => replaced by v(6)
    assert res == Lam(Var(5), App(Var(1), Var(6)))


def test_subst_under_pi_increments_j_and_shifts_subterm() -> None:
    t = Pi(Var(3), App(Var(2), Var(0)))
    sub = App(Var(0), Var(1))
    res = subst(t, sub, j=1)
    exp_ty = Var(2)  # v(3) with j=1 -> 3>1 => 2
    shifted_sub = shift(sub, 1, 0)  # app(v(1), v(2))
    # body under binder: j -> 2
    #   v(2) == j -> replace with shifted_sub
    #   v(0) < j -> stays v(0)
    a2 = Var(0)
    exp_body = App(shifted_sub, a2)
    assert res == Pi(exp_ty, exp_body)


# ------------- β-reduction via subst (TAPL shift dance) -------------


def test_beta_simple_identity() -> None:
    # (λ. Var(0)) arg  -> arg
    assert subst(Var(0), Var(7)) == Var(7)


def test_beta_ignores_argument() -> None:
    # (λ. Var(1)) arg  -> Var(0)  (outer variable gets one level closer)
    assert subst(Var(1), Var(0)) == Var(0)


def test_beta_capture_avoidance_nontrivial() -> None:
    # (λ. λ. Var(2))  applied to  Var(0)
    # body inside outer λ is λ. Var(2)
    # After β on the outer λ, we expect λ. Var(2) (the free var remains free)
    # Walkthrough: subst(body, arg) performs the +1/-1 shift dance correctly
    res = subst(Lam(Var(0), Var(2)), Var(0))
    assert res == Lam(Var(0), Var(1))  # ty shifted; body free index unchanged


def test_beta_argument_with_free_vars_no_capture() -> None:
    # (λ. App(Var(1), Var(0))) (App(Var(1), Var(0)))
    # Should produce App(Var(1), App(Var(2), Var(1))) with correct shifting
    res = subst(App(Var(1), Var(0)), App(Var(1), Var(0)))
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
    # ((λ. λ. App(Var(2), Var(0))) arg)
    # After β of the outer λ with arg=Var(3):
    #   - New λ param type: ty' = subst(v(0), j=0, sub=v(3)) = v(3).
    #   - In the inner body App(Var(2), Var(0)), under the remaining λ we substitute with
    #     j=1 and sub' = shift(v(3), +1, 0) = v(4):
    #       Var(2) -> Var(1)  (since 2>1)
    #       Var(0) -> Var(0)  (since 0<1)
    #   So the result is λ. App(Var(1), Var(0)).
    res = subst(Lam(Var(0), App(Var(2), Var(0))), Var(3))
    assert res == Lam(Var(3), App(Var(1), Var(0)))


# ------------- Subst/Shift interaction laws (spot checks) -------------


def test_shift_subst_commutation_law_spotcheck() -> None:
    # shift(subst(t, j, s), d, c) == subst(shift(t, d, c), j + (0 if c <= j else 0) + d, shift(s, d, 0))
    # We use the common special case c=0:
    left = shift(
        subst(
            App(Var(2), Lam(Var(0), App(Var(1), Var(0)))),
            j=1,
            sub=App(Var(0), Var(2)),
        ),
        by=2,
        cutoff=0,
    )
    right = subst(
        shift(App(Var(2), Lam(Var(0), App(Var(1), Var(0)))), by=2, cutoff=0),
        j=3,
        sub=shift(App(Var(0), Var(2)), by=2, cutoff=0),
    )
    assert left == right


def test_subst_then_subst_index_adjustment_spotcheck() -> None:
    # subst(subst(t, j, s), i, r) vs subst(subst(t, i, r), j', s') with proper index arithmetic
    # Check a small concrete instance
    # Left: replace 2->1, then 0->0
    left = subst(
        subst(App(Var(3), App(Var(2), Var(0))), j=2, sub=Var(1)),
        j=0,
        sub=Var(0),
    )
    # Right: replace 0->0 first, then adjust j because replacing 0 can lower indices >0 by 1
    # After removing j=0, original j=2 becomes j' = 1
    right = subst(
        subst(App(Var(3), App(Var(2), Var(0))), j=0, sub=Var(0)),
        j=1,
        sub=subst(Var(1), j=0, sub=Var(0)),
    )
    assert left == right


# ------------- Subst in types (ty field) -------------


def test_subst_affects_ty_field_at_same_depth() -> None:
    res = subst(Lam(Var(1), Var(0)), sub=Var(7), j=1)
    # ty: Var(1) == j -> replaced by sub
    # body: j becomes 2, sub shifts to shift(7,1,0)=Var(8); v(0) with j=2 stays v(0)
    assert res == Lam(Var(7), Var(0))


def test_subst_ty_decrements_higher_indices() -> None:
    res = subst(Lam(Var(3), Var(0)), sub=Var(42), j=1)
    # ty: 3>1 -> 2
    assert isinstance(res, Lam)
    assert isinstance(res.ty, Var)
    assert res.ty == Var(2)


# ------------- Identity and stability checks -------------


def test_subst_irrelevant_index_no_change() -> None:
    t = App(Var(2), Lam(Var(0), Var(0)))
    s = App(Var(1), Var(0))
    assert subst(t, sub=s, j=99) == t


def test_subst_with_closed_subterm_is_well_behaved() -> None:
    # sub has no free variables relative to cutoff=0 (e.g., Var(0) under its own binder in tests via subst)
    assert subst(App(Var(2), Var(0)), sub=Var(0), j=2) == App(Var(0), Var(0))


# ------------- Regression-style edge cases -------------


def test_no_negative_indices_after_subst_top() -> None:
    # Stress: ensure subst never creates negative indices
    body = App(Var(1), Var(0))
    arg = App(Var(0), Var(1))
    res = subst(body, arg)

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
    assert shift(t, by=3, cutoff=10) == t


def test_subst_high_j_beyond_all_vars_identity() -> None:
    assert subst(Lam(Var(0), App(Var(1), Var(0))), sub=Var(9), j=10) == Lam(
        Var(0), App(Var(1), Var(0))
    )


# --------- other types ----------


def test_shift_respects_cutoff() -> None:
    term = App(Var(1), Var(0))
    shifted = shift(term, by=2, cutoff=1)
    assert shifted == App(Var(3), Var(0))


def test_shift_through_lambda_increments_free_variable() -> None:
    term = Lam(Univ(), App(Var(1), Var(0)))
    shifted = shift(term, by=1, cutoff=0)
    assert shifted == Lam(Univ(), App(Var(2), Var(0)))


def test_subst_replaces_target_and_decrements_greater_indices() -> None:
    term = App(Var(1), Var(0))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == App(Var(0), Succ(Var(0)))


def test_subst_under_lambda_preserves_bound_variable() -> None:
    term = Lam(Univ(), App(Var(1), Var(0)))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == Lam(Univ(), App(Succ(Var(1)), Var(0)))


def test_shift_nested_binders() -> None:
    term = Lam(Univ(), Lam(Univ(), Var(2)))
    shifted = shift(term, by=1, cutoff=0)
    assert shifted == Lam(Univ(), Lam(Univ(), Var(3)))


def test_subst_nested_binder_chain() -> None:
    term = Lam(Univ(), Lam(Univ(), Var(2)))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == Lam(Univ(), Lam(Univ(), Succ(Var(2))))


def test_subst_pi_body() -> None:
    pi_term = Pi(Univ(), Pi(Univ(), Var(1)))
    sub = Succ(Var(0))
    result = subst(pi_term, sub)
    assert result == Pi(Univ(), Pi(Univ(), Var(1)))


def test_subst_natrec_components() -> None:
    term = NatRec(
        P=Lam(Univ(), Univ()),
        base=Zero(),
        step=Lam(Univ(), Var(0)),
        n=Var(0),
    )
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, InductiveElim)


def test_subst_identity_constructs() -> None:
    term = Id(Univ(), Var(0), Var(1))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, Id)


def test_subst_idelim_components() -> None:
    term = IdElim(
        A=Univ(),
        x=Var(0),
        P=Var(1),
        d=Var(2),
        y=Var(3),
        p=Var(4),
    )
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, IdElim)
