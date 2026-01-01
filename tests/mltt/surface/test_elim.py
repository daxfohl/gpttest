from types import MappingProxyType

from mltt.kernel.env import Env
from mltt.surface.etype import ElabEnv
from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term


def elab_ok_in_env(src: str, env: Env) -> None:
    elab_env = ElabEnv.from_env(env)
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = term.elab_infer(elab_env, state)
    state.solve(elab_env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def test_surface_elim_add_comm() -> None:
    env = Env(globals=MappingProxyType({}))
    src = """
    inductive Nat : Type 0 :=
    | Zero
    | Succ (k : Nat);

    let succ : Nat -> Nat := fun x => Nat.Succ x;
    
    inductive Id (A : Type 0) (x : A) : (y : A) -> Type 0 :=
    | Refl : Id A x x;

    let sym : {A : Type 0} -> {x : A} -> {y : A} -> Id A x y -> Id A y x :=
      fun A x y p =>
        elim p return (fun (y : A) (p : Id A x y) => Id A y x) with
        | Refl => ctor Id.Refl;

    let trans : {A : Type 0} -> {x : A} -> {y : A} -> {z : A} -> Id A x y -> Id A y z -> Id A x z :=
      fun A x y z p q =>
        elim q return (fun (z : A) (q : Id A y z) => Id A x z) with
        | Refl => p;

    let ap : {A : Type 0} -> {B : Type 0} -> (f : A -> B) -> {x : A} -> {y : A} -> Id A x y -> Id B (f x) (f y) :=
      fun A B (f : A -> B) x y p =>
        elim p return (fun (y : A) (p : Id A x y) => Id B (f x) (f y)) with
        | Refl => ctor Id.Refl;
    
    let add : Nat -> Nat -> Nat :=
      fun m n =>
        elim m return Nat with
        | Zero => n
        | Succ k ih => succ ih;

    let add_zero_right : (n : Nat) -> Id Nat (add n Nat.Zero) n :=
      fun n =>
        elim n return (fun (n : Nat) => Id Nat (add n Nat.Zero) n) with
        | Zero => ctor Id.Refl
        | Succ k ih => ap succ ih;

    let succ_add : (n : Nat) -> (m : Nat) -> Id Nat (add (succ n) m) (succ (add n m)) :=
      fun n m => ctor Id.Refl;

    let add_succ_right : (n : Nat) -> (m : Nat) -> Id Nat (add m (succ n)) (succ (add m n)) :=
      fun n m =>
        elim m return (fun (m : Nat) => Id Nat (add m (succ n)) (succ (add m n))) with
        | Zero => ctor Id.Refl
        | Succ k ih => ap succ ih;

    let add_comm : (n : Nat) -> (m : Nat) -> Id Nat (add n m) (add m n) :=
      fun n m =>
        (elim n return (fun (n : Nat) => (m : Nat) -> Id Nat (add n m) (add m n)) with
          | Zero => fun m => sym (add_zero_right m)
          | Succ k ih => fun m =>
            trans 
              (trans (succ_add k m) (ap succ (ih m)))
              (sym (add_succ_right k m))) m;
    add_comm
    """
    elab_ok_in_env(src, env)


def test_surface_elim_as_return() -> None:
    env = Env(globals=MappingProxyType({}))
    src = """
    inductive Nat : Type 0 :=
    | Zero
    | Succ (k : Nat);
    let pred : Nat -> Nat :=
      fun (n : Nat) =>
        elim n as z return Nat with
        | Zero => Nat.Zero
        | Succ k => k;
    pred
    """
    elab_ok_in_env(src, env)
