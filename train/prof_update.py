"""Chronométrage de la mise à jour PPO par morceaux, sur des observations synthétiques aux formes réelles
(pas de compilation du moteur). Usage : python train/prof_update.py [--g 128] [--steps 64] [--d 128] [--n 5]
--g : parties par minibatch (B / mb), --steps : longueur du rollout."""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gen"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402
import model  # noqa: E402
import obs as obsm  # noqa: E402
import ppo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--g", type=int, default=128)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--card-layers", type=int, default=2)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--bf16", action="store_true")
ap.add_argument("--no-act", action="store_true", help="sans la mesure de act (évite de compiler new_game)")
a = ap.parse_args()
model.set_compute(a.bf16)
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; {a.g} parties × {a.steps} steps, d={a.d}, "
      f"{'bf16' if a.bf16 else 'f32'}", flush=True)

cset = cardgen.Gen(0).make_set(400, 80)
bank = cardgen.to_bank(cset)
C = int(bank.atk.shape[0])
d, x = cardgen.random_decks(cset, np.random.default_rng(0), 1)
sh = jax.eval_shape(lambda: obsm.observe(bank, core.new_game(bank, jnp.asarray(d[0, 0]), jnp.asarray(d[0, 1]),
                                                             jax.random.PRNGKey(0), jnp.asarray(x[0, 0]), jnp.asarray(x[0, 1]))))
rng = np.random.default_rng(1)
S, G = a.steps, a.g


def fake(sd):
    shp = (S, G) + sd.shape
    if sd.dtype == jnp.bool_:
        return jnp.asarray(rng.random(shp) < 0.3)
    return jnp.asarray(rng.integers(0, 8, shp).astype(sd.dtype))


o = jax.tree.map(fake, sh)
o = o._replace(valid=jnp.asarray(rng.random(o.valid.shape) < 0.7), cid=jnp.asarray(rng.integers(-1, C, o.cid.shape), jnp.int32),
               atk=jnp.asarray(rng.integers(0, 3000, o.atk.shape), jnp.int32))
o = o._replace(special=o.special.at[..., 0].set(True))
T = o.valid.shape[-1]
nl = T * core.NV + core.N_SPECIAL
bt = dict(o=o, idx=jnp.full((S, G), nl - 14, jnp.int32), logp=jnp.full((S, G), -1.0), v=jnp.zeros((S, G)),
          need=jnp.ones((S, G), bool), pl=jnp.zeros((S, G), jnp.int32), adv=jnp.asarray(rng.normal(size=(S, G)), jnp.float32),
          ret=jnp.zeros((S, G)))
print(f"T={T}, logits {nl}, cartes {C}", flush=True)

args = argparse.Namespace(gamma=0.999, lam=0.95, clip=0.2, vf=0.5, ent=0.01)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
params = model.init(jax.random.PRNGKey(0), a.d, a.card_layers, a.layers)
ost = opt.init(params)
act, values, refill, gae, update, enc, obs_b = ppo.make_fns(args, opt)
tok, msk = model.card_tokens(bank)


def timeit(name, f, *xs):
    t0 = time.time()
    r = f(*xs)
    jax.block_until_ready(r)
    tc = time.time() - t0
    t0 = time.time()
    for _ in range(a.n):
        r = f(*xs)
    jax.block_until_ready(r)
    dt = (time.time() - t0) / a.n
    print(f"{name:40s} {1000 * dt:8.1f} ms  (1er appel {tc:.1f} s) ; {S * G / dt:,.0f} échantillons/s", flush=True)


timeit("update complet", update, params, ost, tok, msk, bt)
timeit("encodeur de cartes, avant", enc, params, tok, msk)
cemb = enc(params, tok, msk)
fwd = jax.jit(lambda p, ce, o: model.policy(p, ce, o))
timeit("politique, avant", fwd, params, cemb, o)


@jax.jit
def pol_grad(p, ce, o):
    return jax.grad(lambda p: model.policy(p, ce, o)[0].max(-1).sum() + model.policy(p, ce, o)[1].sum())(p)


timeit("politique, avant + arrière (cartes fixes)", pol_grad, params, cemb, o)


@jax.jit
def enc_grad(p):
    return jax.grad(lambda p: model.encode_cards(p["cards"], tok, msk).sum())(p)


timeit("encodeur de cartes, avant + arrière", enc_grad, params)


@jax.jit
def feats(o):
    return model.dyn_feats(o).sum() + model.glob_feats(o).sum()


timeit("features d'entrée", feats, o)
mb = jax.jit(lambda bt, ch: jax.tree.map(lambda x: x[:, ch], bt))
timeit("extraction d'un minibatch (x[:, chunk])", mb, bt, jnp.arange(G))

# act (observation + politique) sur B parties réelles, comme à chaque appel du rollout
if a.no_act:
    sys.exit()
B = S * G if S * G <= 4096 else 1024
dd, xx = cardgen.random_decks(cset, np.random.default_rng(2), B)
env = ppo.Env(B, 0.05)
s, m = env.new(bank, jnp.asarray(dd), jnp.asarray(xx), jax.random.split(jax.random.PRNGKey(3), B))
f = jnp.full(B, -1, jnp.int32)
S, G = 1, B
timeit(f"observe seul (B={B})", jax.jit(lambda s, m: obs_b(bank, s, m, s.dec_player)), s, m)
timeit(f"act (B={B})", act, params, cemb, bank, s, m, f, jax.random.PRNGKey(4))
