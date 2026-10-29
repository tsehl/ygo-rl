"""Vérifie que le moteur se vmappe : B parties aléatoires en lot = mêmes états que jouées une par une."""
import jax
import jax.numpy as jnp
import numpy as np

import cards
import core

B, D, C, T = 16, 40, 670, 400
key = jax.random.PRNGKey(0)
bank = cards.test_bank(key, C)
CT = bank.code.shape[0]
kd = jax.random.fold_in(key, 3)
decks = jnp.where(jax.random.uniform(kd, (B, 2, D)) < 0.3,          # 30 % de cartes à effet
                  jax.random.randint(jax.random.fold_in(kd, 1), (B, 2, D), C, CT),
                  jax.random.randint(jax.random.fold_in(kd, 2), (B, 2, D), 0, C))


def pick(s, k):
    m = core.legal(bank, s)
    return jax.random.categorical(k, jnp.where(m, 0.0, -1e9))


def rollout(d, k):
    s = core.new_game(bank, d[0], d[1])

    def body(s, kt):
        a = pick(s, kt)
        s2 = core.step(bank, s, a)
        return jax.tree.map(lambda x, y: jnp.where(s.done, x, y), s, s2), None
    return jax.lax.scan(body, s, jax.random.split(k, T))[0]


keys = jax.random.split(jax.random.fold_in(key, 4), B)
batched = jax.jit(jax.vmap(rollout))(decks, keys)
single = jax.jit(rollout)
for b in range(B):
    sb = single(decks[b], keys[b])
    assert all(np.array_equal(x[b], y) for x, y in zip(batched, sb)), b
print("ok : lot == séquentiel ;", int(batched.done.sum()), "/", B, "parties finies ; tours :", np.asarray(batched.turn))
