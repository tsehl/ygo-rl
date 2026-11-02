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
decks, extras = cards.test_decks(jax.random.fold_in(key, 3), bank, B, D, C, 0.4, 6)
s0 = jax.jit(jax.vmap(lambda d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])))(decks, extras,
                                                                                       jax.random.split(key, B))
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

# frac > 0 : des parties sont rendues en route (action forcée en attente, ou « continuer » en cours de résolution) et
# avancent à des rythmes différents. Avec une politique déterministe par partie (numéro de la partie, nombre de ses
# décisions), chaque partie doit passer par les mêmes états et masques, décision après décision, que sous vmap(step).
FRAC, K = 0.25, 60
NA = core.n_actions(s0.cid.shape[1])


def pick(m, g, k):
    idx = np.nonzero(m)[0]
    return int(idx[(g * 7919 + k * 104729) % len(idx)])


def snap(s, g):
    return [np.asarray(x[g]) for x in s]


ref = [[] for _ in range(B)]                         # vmap(step) : (état, masque) à chaque décision
s, m = s0, np.asarray(legal(s0))
for k in range(K):
    done = np.asarray(s.done)
    for g in range(B):
        if not done[g]:
            ref[g].append((snap(s, g), m[g]))
    a = jnp.asarray([0 if done[g] else pick(m[g], g, k) for g in range(B)], jnp.int32)
    s2 = step_v(s, a)
    s = jax.tree.map(lambda x, y: jnp.where(s.done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), s, s2)
    m = np.asarray(legal(s))

step_f = jax.jit(lambda s, a: core.step_batch(bank, s, a, FRAC))
cnt = np.zeros(B, int)
s, m, f = s0, np.asarray(legal(s0)), np.full(B, -1)
n_cont = n_wait = 0
for t in range(6 * K):
    done = np.asarray(s.done)
    a = np.zeros(B, int)
    for g in range(B):
        if done[g] or cnt[g] >= K:
            a[g] = -1
        elif f[g] >= 0:
            a[g] = f[g]
        else:                                           # décision de la politique
            x, mx = ref[g][cnt[g]] if cnt[g] < len(ref[g]) else (None, None)
            assert x is not None, f"partie {g} : décision {cnt[g]} en trop"
            for name, u, v in zip(core.State._fields, x, snap(s, g)):
                assert np.array_equal(u, v), f"frac={FRAC}, partie {g}, décision {cnt[g]} : champ {name} différent"
            assert np.array_equal(mx, m[g]), f"frac={FRAC}, partie {g}, décision {cnt[g]} : masques"
            a[g] = pick(m[g], g, cnt[g])
            cnt[g] += 1
    if (a < 0).all():
        break
    s, m, f = step_f(s, jnp.asarray(a, jnp.int32))
    m, f = np.asarray(m), np.asarray(f)
    n_cont += int((f == NA).sum())
    n_wait += int(((f >= 0) & (f < NA)).sum())
fin = sum(cnt[g] >= min(K, len(ref[g])) for g in range(B))
assert fin == B, f"frac={FRAC} : {B - fin} parties n'ont pas fini leurs décisions"
print(f"ok : step_batch(frac={FRAC}) suit vmap(step) décision par décision ({B} parties × {K} décisions ; "
      f"{n_cont} parties rendues en cours de résolution, {n_wait} avec une action forcée en attente)")
