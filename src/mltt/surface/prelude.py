from types import MappingProxyType

from mltt.inductive import vec, nat, fin
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
    register_value(g, "Fin", fin.Fin)
    register_value(g, "Fin.FZ", fin.FZCtor)
    register_value(g, "Fin.FS", fin.FSCtor)
    register_value(g, "Vec", vec.Vec)
    register_value(g, "Vec.Nil", vec.NilCtor)
    register_value(g, "Vec.Cons", vec.ConsCtor)
    return g


def prelude_env() -> Env:
    return Env(globals=MappingProxyType(prelude_globals()))
