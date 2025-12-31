from types import MappingProxyType

from mltt.inductive import (
    bool as bool_ind,
    list as list_ind,
    vec,
    nat,
    fin,
    maybe,
    sigma,
)
from mltt.kernel.ast import Term
from mltt.kernel.environment import GlobalDecl, Env


def register_value(g: dict[str, GlobalDecl], name: str, value: Term) -> None:
    if name in g:
        raise ValueError("dup")
    ty = value.infer_type(Env(globals=MappingProxyType(g)))
    uarity = getattr(value, "uarity", 0)
    g[name] = GlobalDecl(ty=ty, value=value, reducible=True, uarity=uarity)


def prelude_globals() -> dict[str, GlobalDecl]:
    g: dict[str, GlobalDecl] = {}
    register_value(g, "Nat", nat.Nat)
    register_value(g, "Nat.Zero", nat.ZeroCtor)
    register_value(g, "Nat.Succ", nat.SuccCtor)
    register_value(g, "Maybe", maybe.Maybe)
    register_value(g, "Maybe.Nothing", maybe.NothingCtor)
    register_value(g, "Maybe.Just", maybe.JustCtor)
    register_value(g, "Maybe_U", maybe.Maybe_U)
    register_value(g, "Maybe.Nothing_U", maybe.Nothing_U)
    register_value(g, "Maybe.Just_U", maybe.Just_U)
    register_value(g, "Bool", bool_ind.Bool)
    register_value(g, "Bool.True", bool_ind.TrueCtor)
    register_value(g, "Bool.False", bool_ind.FalseCtor)
    register_value(g, "List", list_ind.List)
    register_value(g, "List.Nil", list_ind.NilCtor)
    register_value(g, "List.Cons", list_ind.ConsCtor)
    register_value(g, "List_U", list_ind.List_U)
    register_value(g, "List.Nil_U", list_ind.Nil_U)
    register_value(g, "List.Cons_U", list_ind.Cons_U)
    register_value(g, "Fin", fin.Fin)
    register_value(g, "Fin.FZ", fin.FZCtor)
    register_value(g, "Fin.FS", fin.FSCtor)
    register_value(g, "Vec", vec.Vec)
    register_value(g, "Vec.Nil", vec.NilCtor)
    register_value(g, "Vec.Cons", vec.ConsCtor)
    register_value(g, "Vec_U", vec.Vec_U)
    register_value(g, "Vec.Nil_U", vec.Nil_U)
    register_value(g, "Vec.Cons_U", vec.Cons_U)
    register_value(g, "Sigma", sigma.Sigma)
    register_value(g, "Sigma.Pair", sigma.PairCtor)
    return g


def prelude_env() -> Env:
    return Env(globals=MappingProxyType(prelude_globals()))
