"""Vérifie que step_batch (frac = 0, chemin d'entraînement) donne exactement les mêmes états et masques que
vmap(step) (chemin vérifié contre l'oracle par compare.py), sur des parties aléatoires jouées en lot."""
import jax
import jax.numpy as jnp
import numpy as np

import cards
import core

B, D, C, T = 32, 40, 670, 150
key = jax.random.PRNGKey(1)
bank = cards.test_bank(key, C)
CT = bank.code.shape[0]
kd = jax.random.fold_in(key, 3)
decks = jnp.where(jax.random.uniform(kd, (B, 2, D)) < 0.4,          # 40 % de cartes à effet
                  jax.random.randint(jax.random.fold_in(kd, 1), (B, 2, D), C, CT),
                  jax.random.randint(jax.random.fold_in(kd, 2), (B, 2, D), 0, C))
s0 = jax.jit(jax.vmap(lambda d, k: core.new_game(bank, d[0], d[1], k)))(decks, jax.random.split(key, B))
legal = jax.jit(jax.vmap(lambda s: core.legal(bank, s)))
step_v = jax.jit(jax.vmap(lambda s, a: core.step(bank, s, a)))
step_b = jax.jit(lambda s, a: core.step_batch(bank, s, a, 0.0))

sa, sb = s0, s0
m = legal(s0)
for t in range(T):
    a = jax.random.categorical(jax.random.fold_in(key, 100 + t), jnp.where(m, 0.0, -1e9)).astype(jnp.int32)
    a = jnp.where(sa.done, 0, a)
    sa2 = step_v(sa, a)
    sa = jax.tree.map(lambda x, y: jnp.where(sa.done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), sa, sa2)
    sb, mb, f = step_b(sb, jnp.where(sb.done, -1, a))
    assert not np.asarray(f >= 0).any(), "frac = 0 : aucune action forcée ne doit rester en attente"
    for name, x, y in zip(core.State._fields, sa, sb):
        assert np.array_equal(np.asarray(x), np.asarray(y)), f"t={t} : champ {name} différent"
    m = legal(sa)
    assert np.array_equal(np.asarray(m & ~sa.done[:, None]), np.asarray(mb & ~sb.done[:, None])), f"t={t} : masques"
print(f"ok : step_batch == vmap(step) sur {B} parties × {T} décisions ; {int(sa.done.sum())} finies")
