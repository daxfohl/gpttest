"""Shared helpers for inductive types."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from .ast import Term, Var, Pi, Univ, App, Reducer
from .debruijn import Ctx
from .util import nested_pi, apply_term, decompose_app


@dataclass(frozen=True)
class CtorAnalysis:
    inst_arg_types: tuple[Term, ...]
    recursive_field_positions: tuple[int, ...]
    ih_types: tuple[Term, ...] = ()


def discharge_binders(
    schema: Term, actuals: tuple[Term, ...], *, depth_above: int
) -> Term:
    """
    Substitute ``actuals`` for the outer binder block of ``schema``.

    Schema is assumed written under (actuals)(...) where ``depth_above`` is the number
    of binders *below* the actuals block that remain in scope at substitution time.
    For each actual, eliminate at de Bruijn index:
        index = depth_above + len(actuals) - i - 1
    using the project’s convention:
        schema = schema.subst(actual.shift(index), index)
    """
    t = schema
    k = len(actuals)
    for i, a in enumerate(actuals):
        index = depth_above + k - i - 1
        t = t.subst(a.shift(index), index)
    return t


def instantiate_field_type_in_gamma(
    schema: Term,
    *,
    field_index: int,
    params_actual: tuple[Term, ...],
    fields_prefix_actual: tuple[Term, ...],
) -> Term:
    """
    schema context: (params)(fields_prefix length i)
    output context: Γ
    """
    t = discharge_binders(schema, params_actual, depth_above=field_index)
    t = discharge_binders(t, fields_prefix_actual, depth_above=0)
    return t


def instantiate_field_type_under_fields(
    schema: Term,
    *,
    field_index: int,
    params_actual: tuple[Term, ...],
) -> Term:
    """
    schema context: (params)(fields_prefix length i)
    output context: Γ,(fields_prefix length i)
    (params discharged; fields remain as Vars)
    """
    return discharge_binders(schema, params_actual, depth_above=field_index)


def instantiate_ctor_arg_types_under_fields(
    ctor_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
) -> tuple[Term, ...]:
    return tuple(
        instantiate_field_type_under_fields(
            schema, field_index=i, params_actual=params_actual
        )
        for i, schema in enumerate(ctor_arg_types)
    )


def _instantiate_ctor_result_indices_under_fields(
    result_indices: tuple[Term, ...],
    params_actual_shifted: tuple[Term, ...],  # already shifted by m at callsite
    m: int,  # number of ctor fields in scope
) -> tuple[Term, ...]:
    """
    result_indices schemas are written in context (params)(fields).
    We are currently in context Γ,(fields) (params already in Γ but shifted by m).
    Discharge params binders only; keep field vars (0..m-1) intact.
    """
    p = len(params_actual_shifted)
    out: list[Term] = []
    for schema in result_indices:
        t = schema
        # eliminate param binders outermost → innermost so inner indices stay stable
        for s, param in enumerate(params_actual_shifted):
            index = m + p - s - 1
            t = t.subst(param.shift(p - s - 1), index)
        out.append(t)
    return tuple(out)


def build_ih_types_from_analysis(
    *,
    ind: Ind,
    inst_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
    params_in_fields_ctx: tuple[Term, ...],
    motive: Term,
    recursive_field_positions: tuple[int, ...],
    heads_and_args: tuple[tuple[Term, tuple[Term, ...]], ...],
) -> list[Term]:
    """
    Compute the induction hypothesis telescope for a constructor.

    All inputs are expressed in context Γ; ``inst_arg_types`` are written in
    contexts Γ,(fields_prefix) for each field. IH types are returned in context
    Γ,(fields),(ihs_so_far) so they can be appended directly after the fields.
    """

    p = len(ind.param_types)
    q = len(ind.index_types)
    m = len(inst_arg_types)

    ih_types: list[Term] = []
    for ri, field_index in enumerate(recursive_field_positions):
        _, field_args = heads_and_args[field_index]
        params_field = field_args[:p]
        indices_field = field_args[p : p + q]

        # field types are written under the previous fields only; shift them
        # into the full Γ,(fields) context before comparing.
        shift_into_fields_ctx = m - field_index
        params_field_in_fields_ctx = tuple(
            param.shift(shift_into_fields_ctx) for param in params_field
        )
        if params_field_in_fields_ctx != params_in_fields_ctx:
            raise TypeError(
                "Inductive recursive argument parameters do not match scrutinee:\n"
                f"  ctor fields = {inst_arg_types}\n"
                f"  field index = {field_index}\n"
                f"  params_field (in Γ,fields) = {params_field_in_fields_ctx}\n"
                f"  params_actual (in Γ) = {params_actual}"
            )

        indices_field_in_fields_ctx = tuple(
            index.shift(shift_into_fields_ctx) for index in indices_field
        )

        # field_index is counted outermost → innermost; translate to Var index
        # in Γ,(fields).
        field_var_index = m - 1 - field_index
        ih_type = apply_term(
            motive.shift(m + ri),
            *(index.shift(ri) for index in indices_field_in_fields_ctx),
            Var(field_var_index + ri),
        ).whnf()
        ih_types.append(ih_type)

    return ih_types


def analyze_ctor_for_elim(
    *,
    ind: "Ind",
    ctor: "Ctor",
    params_actual: tuple[Term, ...],
    params_in_fields_ctx: tuple[Term, ...],
    motive: Term,
    want_ih_types: bool,
) -> CtorAnalysis:
    """
    Produces one canonical notion of recursive fields and IH ordering.
    - Recursive detection must use instantiated field schemas and WHNF.
    - IH ordering must match recursive_field_positions order.
    """
    inst_arg_types = instantiate_ctor_arg_types_under_fields(
        ctor.arg_types, params_actual
    )

    heads_and_args = tuple(
        decompose_app(field_ty.whnf()) for field_ty in inst_arg_types
    )
    recursive_field_positions = tuple(
        i for i, (head, _) in enumerate(heads_and_args) if head is ind
    )

    if not want_ih_types:
        return CtorAnalysis(
            inst_arg_types=inst_arg_types,
            recursive_field_positions=recursive_field_positions,
        )

    ih_types = build_ih_types_from_analysis(
        ind=ind,
        inst_arg_types=inst_arg_types,
        params_actual=params_actual,
        params_in_fields_ctx=params_in_fields_ctx,
        motive=motive,
        recursive_field_positions=recursive_field_positions,
        heads_and_args=heads_and_args,
    )
    return CtorAnalysis(
        inst_arg_types=inst_arg_types,
        recursive_field_positions=recursive_field_positions,
        ih_types=tuple(ih_types),
    )


def _instantiate_ctor_arg_types(
    ctor_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
) -> tuple[Term, ...]:
    return instantiate_ctor_arg_types_under_fields(ctor_arg_types, params_actual)


def _build_ih_types(
    ind: Ind,
    inst_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
    params_in_fields_ctx: tuple[Term, ...],
    motive: Term,
) -> list[Term]:
    heads_and_args = tuple(
        decompose_app(field_ty.whnf()) for field_ty in inst_arg_types
    )
    recursive_field_positions = tuple(
        j for j, (head, _) in enumerate(heads_and_args) if head is ind
    )
    return build_ih_types_from_analysis(
        ind=ind,
        inst_arg_types=inst_arg_types,
        params_actual=params_actual,
        params_in_fields_ctx=params_in_fields_ctx,
        motive=motive,
        recursive_field_positions=recursive_field_positions,
        heads_and_args=heads_and_args,
    )


def infer_ind_type(ctx: Ctx, ind: Ind) -> Term:
    """Compute the dependent function type of an inductive.

    The resulting Pi-tower has parameters outermost, then indices,
    finishing with the universe level.
    """

    for b in ind.all_binders:
        b.expect_universe(ctx)
        ctx = ctx.prepend_each(b)
    return nested_pi(*ind.all_binders, return_ty=Univ(ind.level))


def infer_ctor_type(ctor: Ctor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then constructor
    arguments, finishing with the inductive head applied to the result
    indices.
    """
    ind = ctor.inductive
    if len(ctor.result_indices) != len(ind.index_types):
        raise TypeError(
            "Constructor result indices must match inductive index arity:\n"
            f"  ctor = {ctor}\n"
            f"  expected arity = {len(ind.index_types)}\n"
            f"  found arity = {len(ctor.result_indices)}"
        )
    # Parameters bind outermost, then constructor arguments.
    #   [params][args] from outermost to innermost.
    offset = len(ctor.arg_types)
    param_vars = [
        Var(i) for i in reversed(range(offset, offset + len(ind.param_types)))
    ]
    return nested_pi(
        *ind.param_types,
        *ctor.arg_types,
        return_ty=apply_term(ctor.inductive, *param_vars, *ctor.result_indices),
    )


def infer_elim_type(elim: Elim, ctx: Ctx) -> Term:
    """Infer the type of an ``InductiveElim`` while checking its well-formedness."""
    # 1. Infer type of scrutinee and extract params/indices.

    scrut = elim.scrutinee
    ind = elim.inductive
    scrut_ty = scrut.infer_type(ctx).whnf()
    scrut_ty_head, scrut_ty_bindings = decompose_app(scrut_ty)
    if scrut_ty_head is not ind:
        raise TypeError(
            "Eliminator scrutinee not of the right inductive type:\n"
            f"  scrutinee = {scrut}\n"
            f"  expected head = {ind}\n"
            f"  found head = {scrut_ty_head}"
        )

    # 2.1 Partially apply motive to the actual indices
    motive = elim.motive
    p = len(ind.param_types)
    q = len(ind.index_types)
    params_actual = scrut_ty_bindings[:p]
    indices_actual = scrut_ty_bindings[p:]
    motive_applied = apply_term(motive, *indices_actual)

    # 2.2 Infer the type of this partially applied motive
    motive_applied_ty = motive_applied.infer_type(ctx).whnf()
    if not isinstance(motive_applied_ty, Pi):
        raise TypeError(
            "InductiveElim motive must take scrutinee after indices:\n"
            f"  motive = {motive}\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty = {motive_applied_ty}"
        )

    # 2.3 The scrutinee binder domain must match the scrutinee type
    scrut_dom = motive_applied_ty.arg_ty
    if not scrut_dom.type_equal(scrut_ty):
        raise TypeError(
            "InductiveElim motive scrutinee domain mismatch:\n"
            f"  expected scrut_ty = {scrut_ty}\n"
            f"  found scrut_dom = {scrut_dom}"
        )

    # 2.4 The motive codomain must be a universe
    body_ty = motive_applied_ty.return_ty.whnf()
    if not isinstance(body_ty, Univ):
        raise TypeError(
            "InductiveElim motive codomain must be a universe:\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty.return_ty = {motive_applied_ty.return_ty}\n"
            f"  normalized = {body_ty}"
        )

    # 3. For each constructor, compute the expected branch type and check
    for ctor, case in zip(ind.constructors, elim.cases, strict=True):
        # Context legend:
        #   Γ          = current context passed to this eliminator
        #   fields     = constructor arguments (outermost → innermost)
        #   ihs        = induction hypotheses for recursive fields
        #
        # Most ctor schemas are written in context (params)(fields). At this
        # point params are already fixed by the scrutinee in Γ, so we
        # instantiate only those param binders and keep fields as de Bruijn
        # variables.

        m = len(ctor.arg_types)
        params_in_fields_ctx = tuple(p.shift(m) for p in params_actual)

        analysis = analyze_ctor_for_elim(
            ind=ind,
            ctor=ctor,
            params_actual=params_actual,
            params_in_fields_ctx=params_in_fields_ctx,
            motive=motive,
            want_ih_types=True,
        )
        inst_arg_types = analysis.inst_arg_types
        ih_types = list(analysis.ih_types)
        m = len(inst_arg_types)
        r = len(ih_types)

        # 3.2 Work in context Γ,fields: shift Γ-level params and motive by m.
        motive_in_fields_ctx = motive.shift(m)

        # Build a scrutinee-shaped term: C params field_vars.
        # field_vars are Var(m-1) .. Var(0) in the Γ,fields context.
        field_vars = tuple(Var(j) for j in reversed(range(m)))
        scrut_like = apply_term(ctor, *params_in_fields_ctx, *field_vars)

        # 3.3 Instantiate ctor result indices under fields (params only).
        result_indices_inst = _instantiate_ctor_result_indices_under_fields(
            ctor.result_indices, params_in_fields_ctx, m
        )

        # 3.4 Build IH types for recursive fields.
        # Each IH is in context Γ,fields,ihs_so_far, so shift by m + ri.

        # 3.5 Branch codomain in Γ,fields; then shift for inserted IH binders.
        codomain_in_fields_ctx = apply_term(
            motive_in_fields_ctx, *result_indices_inst, scrut_like
        )
        codomain = codomain_in_fields_ctx.shift(r).whnf()

        # 3.6 Expected branch type: Π fields. Π ihs. codomain.
        telescope = (*inst_arg_types, *ih_types)
        branch_ty = nested_pi(*telescope, return_ty=codomain)
        case.type_check(branch_ty, ctx)

    u = motive_applied_ty.return_ty.expect_universe(ctx)  # cod should be Univ(u)

    # target type is P i⃗_actual scrut
    target_ty = App(motive_applied, scrut)

    # sanity check target_ty really is a type in Type u (or ≤ u with cumulativity)
    _ = target_ty.infer_type(ctx).expect_universe(ctx)

    if u < ind.level:
        raise TypeError(
            "Eliminator motive returns too small a universe:\n"
            f"  motive level = {u}\n"
            f"  inductive level = {ind.level}"
        )

    return target_ty


@dataclass(frozen=True, kw_only=True)
class Ind(Term):
    """A generalized inductive type with constructors."""

    name: str
    param_types: tuple[Term, ...] = field(repr=False, default=())
    index_types: tuple[Term, ...] = field(repr=False, default=())
    constructors: tuple["Ctor", ...] = field(repr=False, default=())
    level: int = 0

    @cached_property
    def all_binders(self) -> tuple[Term, ...]:
        return self.param_types + self.index_types

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: "Ctx") -> Term:
        from .ind import infer_ind_type

        return infer_ind_type(ctx, self)

    def shift(self, by: int, cutoff: int = 0) -> Term:  # type: ignore[override]
        return self

    def subst(self, sub: Term, j: int = 0) -> Term:  # type: ignore[override]
        return self

    def _type_check(self, ty: Term, ctx: "Ctx") -> None:
        self._check_against_inferred(ty.whnf(), ctx, label="Inductive type")

    def _type_equal_with(self, other: Term, ctx: "Ctx") -> bool:
        return isinstance(other, Ind) and other is self


@dataclass(frozen=True, kw_only=True)
class Ctor(Term):
    """A constructor for an inductive type."""

    name: str
    inductive: Ind = field(repr=False)
    arg_types: tuple[Term, ...] = field(repr=False, default=())
    result_indices: tuple[Term, ...] = field(repr=False, default=())

    @cached_property
    def all_binders(self) -> tuple[Term, ...]:
        return self.arg_types + self.result_indices

    def index_in_inductive(self) -> int:
        for idx, ctor in enumerate(self.inductive.constructors):
            if ctor is self:
                return idx
        raise TypeError("Constructor does not belong to inductive type")

    def iota_reduce(
        self, cases: tuple[Term, ...], args: tuple[Term, ...], motive: Term
    ) -> Term:
        """Compute the iota-reduction of an eliminator on a fully-applied ctor."""
        from .util import apply_term, decompose_app

        ind = self.inductive
        p = len(ind.param_types)
        params_actual = args[:p]
        ctor_args = args[p:]

        ihs: list[Term] = []
        for i, (arg_term, arg_ty) in enumerate(
            zip(ctor_args, self.arg_types, strict=True)
        ):
            inst_ty = instantiate_field_type_in_gamma(
                arg_ty,
                field_index=i,
                params_actual=params_actual,
                fields_prefix_actual=ctor_args[:i],
            ).whnf()
            head, _ = decompose_app(inst_ty)
            if head is self.inductive:
                ihs.append(
                    Elim(
                        inductive=self.inductive,
                        motive=motive,
                        cases=cases,
                        scrutinee=arg_term,
                    )
                )

        case = cases[self.index_in_inductive()]
        return apply_term(case, *ctor_args, *ihs)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: "Ctx") -> Term:
        from .ind import infer_ctor_type

        return infer_ctor_type(self)

    def shift(self, by: int, cutoff: int = 0) -> Term:  # type: ignore[override]
        return self

    def subst(self, sub: Term, j: int = 0) -> Term:  # type: ignore[override]
        return self

    def _type_check(self, ty: Term, ctx: "Ctx") -> None:
        self._check_against_inferred(ty.whnf(), ctx, label="Constructor")

    def _type_equal_with(self, other: Term, ctx: "Ctx") -> bool:
        return isinstance(other, Ctor) and other is self


@dataclass(frozen=True)
class Elim(Term):
    """Elimination principle for an inductive type."""

    inductive: Ind
    motive: Term
    cases: tuple[Term, ...]
    scrutinee: Term

    # Reduction ----------------------------------------------------------------
    def whnf(self) -> Term:
        from .util import decompose_app

        scrutinee_whnf = self.scrutinee.whnf()
        head, args = decompose_app(scrutinee_whnf)
        if isinstance(head, Ctor) and head.inductive is self.inductive:
            expected_args = len(self.inductive.param_types) + len(head.arg_types)
            if len(args) != expected_args:
                raise ValueError()
            return head.iota_reduce(self.cases, args, self.motive).whnf()
        return Elim(self.inductive, self.motive, self.cases, scrutinee_whnf)

    def _reduce_children(self, reducer: Reducer) -> Term:
        return self._reduce_dataclass_children(reducer)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: "Ctx") -> Term:
        from .ind import infer_elim_type

        return infer_elim_type(self, ctx)

    def _type_check(self, ty: Term, ctx: "Ctx") -> None:
        self._check_against_inferred(ty.whnf(), ctx, label="Eliminator")

    def _type_equal_with(self, other: Term, ctx: "Ctx") -> bool:
        if not isinstance(other, Elim) or other.inductive is not self.inductive:
            return False
        if len(self.cases) != len(other.cases):
            return False
        return (
            self.motive.type_equal(other.motive, ctx)
            and all(
                case1.type_equal(case2, ctx)
                for case1, case2 in zip(self.cases, other.cases, strict=True)
            )
            and self.scrutinee.type_equal(other.scrutinee, ctx)
        )


__all__ = [
    "Ind",
    "Ctor",
    "Elim",
]
