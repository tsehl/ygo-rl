"""Profil interne de step_batch : où part le temps d'un appel, mesuré dans la fonction compilée elle-même.

Chronométrer une brique seule depuis Python ne marche pas (passer les 111 tableaux de l'état coûte ~1 ms par appel) et
le coût de refresh_b, apb, etc. dépend du nombre de parties modifiées à chaque tour. Ici, une copie de core.py est
modifiée à la volée (jamais écrite dans engine/) : chaque composant de step_batch est enveloppé dans une boucle qui
le répète REPS[site] fois sur la même entrée et garde le dernier résultat. Avec REPS = 1 partout, le résultat est
celui de step_batch. Sur un appel fixé (même état, mêmes actions), passer un site à 3 répétitions ajoute exactement
deux fois le coût total de ce composant dans l'appel, sans changer le déroulement : coût = (T_3 − T_1) / 2.
La répétition ne peut pas être sortie de la boucle ni supprimée par XLA : l'entrée est perturbée par un zéro inconnu
du compilateur, porté par la boucle.

Les sites sont comptés à part pour le premier tour (lot complet) et pour les tours suivants (sous-lot de B/4).
Un site témoin (identité, une fois par tour de run_b) donne le coût de l'enveloppe par invocation.

Usage : python bench/sb_profile.py [--b 1024] [--calls 100] [--points 3] [--n 8] [--proc 0] [--frac 0.1]"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")
ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=1024)
ap.add_argument("--calls", type=int, default=100)
ap.add_argument("--points", type=int, default=3, help="états de mesure (un tous les 10 appels après --calls)")
ap.add_argument("--n", type=int, default=8, help="répétitions du chronométrage par site")
ap.add_argument("--proc", type=int, default=0)
ap.add_argument("--frac", type=float, default=0.1)
a = ap.parse_args()

SITES = ["exec_one (run)", "refresh_b (run)", "trig_eval", "auto_pass (apb)", "passes", "apply", "refresh_b (tour)",
         "refresh_ex_b", "témoin identité (run)", "legal", "forced_m", "sous-lot : extraction", "sous-lot : écriture"]
OFF = 16
PATCH = [
    ("def step_batch(bank, s, a, frac=0.0, stats=False):", '''def _rep(i, f, *xs):
    n = REPS[_OFF[0] + i]
    out0 = jax.tree.map(lambda t: jnp.zeros(t.shape, t.dtype), jax.eval_shape(f, *xs))
    pert = lambda z: jax.tree.map(lambda x: (x ^ (z != 0)) if x.dtype == jnp.bool_ else x + z.astype(x.dtype), xs)
    return lax.while_loop(lambda c: c[0] < n, lambda c: (c[0] + 1, c[1] * 2, f(*pert(c[1] * 2))), (_i(0), REPS[-1], out0))[2]


REPS, _OFF = None, [0]


def step_batch(bank, s, a, frac=0.0, stats=False):'''),
    ("            return refresh_b(bank, s, _where(r, jax.vmap(lambda s: exec_one(bank, s))(s), s)), i + 1, n + r.sum()",
     '''            e = _rep(0, lambda s: jax.vmap(lambda s: exec_one(bank, s))(s), s)
            s2 = _rep(8, lambda x: x, _where(r, e, s))
            return _rep(1, lambda p, q: refresh_b(bank, p, q), s, s2), i + 1, n + r.sum()'''),
    ("""        s = lax.cond(need.any(), lambda s: _where(need, jax.vmap(lambda s: trig_eval(bank, s))(s), s),
                     lambda s: s, s)
        ap = apb(s, r0 | need, ap)""",
     """        s = _rep(2, lambda s: lax.cond(need.any(), lambda s: _where(need, jax.vmap(lambda s: trig_eval(bank, s))(s), s),
                     lambda s: s, s), s)
        ap = _rep(3, lambda s, ap: apb(s, r0 | need, ap), s, ap)"""),
    ("        s, ap_new, p2 = lax.cond(ap.any(), passes, lambda s: (s, ap, _i(0)), s)",
     "        s, ap_new, p2 = _rep(4, lambda s: lax.cond(ap.any(), passes, lambda s: (s, ap, _i(0)), s), s)"),
    ("        s1 = jax.vmap(lambda s, a: apply(bank, s, a))(s, jnp.clip(a, 0, na - 1))",
     "        s1 = _rep(5, lambda s, a: jax.vmap(lambda s, a: apply(bank, s, a))(s, jnp.clip(a, 0, na - 1)), s, a)"),
    ("        s2 = refresh_b(bank, s, _where(act & (a < na), s1, s))",
     "        s2 = _rep(6, lambda p, q: refresh_b(bank, p, q), s, _where(act & (a < na), s1, s))"),
    ("        s2, k, ap = settle_b(s2, frac * act.sum(), k, apb(s2, act, jnp.zeros_like(act)))",
     "        s2, k, ap = settle_b(s2, frac * act.sum(), k, _rep(3, lambda s: apb(s, act, jnp.zeros_like(act)), s2))"),
    ("        s = refresh_ex_b(bank, s, act & ~left)", "        s = _rep(7, lambda s: refresh_ex_b(bank, s, act & ~left), s)"),
    ("""        lg = lax.cond(fl, lambda s: jax.vmap(lambda s: legal(bank, s))(s),
                      lambda s: jax.vmap(lambda s: legal(bank, s, fus=False))(s), s)""",
     """        lg = _rep(9, lambda s: lax.cond(fl, lambda s: jax.vmap(lambda s: legal(bank, s))(s),
                      lambda s: jax.vmap(lambda s: legal(bank, s, fus=False))(s), s), s)"""),
    ("        f = jnp.where(left, na, jax.vmap(lambda s, m: forced_m(bank, s, m))(s, m))",
     "        f = jnp.where(left, na, _rep(10, lambda s, m: jax.vmap(lambda s, m: forced_m(bank, s, m))(s, m), s, m))"),
    ("        ss, ms, fs, k = rnd(jax.tree.map(g, s), g(m), jnp.where(idx < nb, g(f), -1), k)",
     """        gs, gm = _rep(11, lambda s, m: (jax.tree.map(g, s), g(m)), s, m)
        ss, ms, fs, k = rnd(gs, gm, jnp.where(idx < nb, g(f), -1), k)"""),
    ("        return jax.tree.map(put, s, ss), put(m, ms), put(f, fs), i + 1, k",
     """        ps, pm = _rep(12, lambda s, ss, m, ms: (jax.tree.map(put, s, ss), put(m, ms)), s, ss, m, ms)
        return ps, pm, put(f, fs), i + 1, k"""),
    ("    s, m, f, k = rnd(s, jnp.zeros((nb, n_actions(s.cid.shape[1])), bool), a.astype(I32), jnp.zeros(11, I32))",
     """    _OFF[0] = 0
    s, m, f, k = rnd(s, jnp.zeros((nb, n_actions(s.cid.shape[1])), bool), a.astype(I32), jnp.zeros(11, I32))
    k1 = k
    _OFF[0] = %d""" % OFF),
    ("    return (s, m, f, k) if stats else (s, m, f)", "    return (s, m, f, jnp.concatenate([k1, k])) if stats else (s, m, f)"),
]
src = open(os.path.join(ENG, "core.py")).read()
for old, new in PATCH:
    assert src.count(old) == 1, f"core.py a changé, site introuvable ou ambigu :\n{old}"
    src = src.replace(old, new)
tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "core.py"), "w").write(src)
sys.path[:0] = [tmp, ENG, os.path.join(HERE, "..", "gen")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import core  # noqa: E402  (la copie instrumentée, avant cardgen qui importe core)

assert core.__file__.startswith(tmp)
import cardgen  # noqa: E402
NR = 2 * OFF


def _sb(bank, s, x, reps):
    core.REPS = reps
    return core.step_batch(bank, s, x, a.frac, stats=True)


sb = jax.jit(_sb)
cset = cardgen.Gen(a.proc).make_set(400, 80)
bank = cardgen.to_bank(cset)
d, x = cardgen.random_decks(cset, np.random.default_rng(1), a.b)
ng = jax.jit(jax.vmap(lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1]), in_axes=(None, 0, 0, 0)))
lg = jax.jit(jax.vmap(lambda bank, s: core.legal(bank, s), in_axes=(None, 0)))
pol = jax.jit(lambda k, m, f: jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9))).astype(jnp.int32))
s = ng(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(jax.random.PRNGKey(0), a.b))
m = lg(bank, s)
f = jnp.full(a.b, -1, jnp.int32)
key = jax.random.PRNGKey(1)
ones = np.ones(NR, np.int32)
ones[-1] = 0                                             # dernier élément : le zéro inconnu du compilateur
base = jnp.asarray(ones)
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} frac={a.frac}", flush=True)


def timed(s, act, reps):
    r = sb(bank, s, act, reps)
    jax.block_until_ready(r)
    t1 = time.time()
    for _ in range(a.n):
        r = sb(bank, s, act, reps)
    jax.block_until_ready(r)
    return 1000 * (time.time() - t1) / a.n, r


tot = np.zeros((2, len(SITES)))
tb, kk = 0.0, np.zeros(22)
t0 = time.time()
pt = 0
for t in range(a.calls + 10 * a.points):
    key, k = jax.random.split(key)
    act = pol(k, m, f)
    if t >= a.calls and (t - a.calls) % 10 == 0:
        T1, r = timed(s, act, base)
        ref = jax.tree.map(np.asarray, r[:3])
        tb += T1
        kk += np.asarray(r[3])
        for ph in (0, 1):
            for i in range(len(SITES)):
                rp = ones.copy()
                rp[ph * OFF + i] = 3
                T3, r3 = timed(s, act, jnp.asarray(rp))
                same = all(np.array_equal(u, np.asarray(v)) for u, v in zip(jax.tree.leaves(ref), jax.tree.leaves(r3[:3])))
                assert same, f"site {SITES[i]} : la répétition a changé le résultat"
                tot[ph, i] += (T3 - T1) / 2
        pt += 1
        print(f"point {pt} (appel {t}) : {T1:.1f} ms", flush=True)
    s, m, f, c = sb(bank, s, act, base)
    if t == 0:
        jax.block_until_ready(f)
        print(f"compilation {time.time() - t0:.0f} s", flush=True)
tot, tb, kk = tot / pt, tb / pt, kk / pt
names = ["ext", "settle", "run", "trig", "pass", "pass2", "fus", "n_run", "n_trig", "n_pass", "n_pas2"]
print(f"\nappel moyen : {tb:.1f} ms sur {pt} états de milieu de partie")
print("compteurs, premier tour : " + " ".join(f"{n}={v:.1f}" for n, v in zip(names, kk[:11])))
print("compteurs, tout l'appel : " + " ".join(f"{n}={v:.1f}" for n, v in zip(names, kk[11:])))
print(f"\n{'composant':28s}{'1er tour (B)':>14s}{'tours suivants (B/4)':>22s}{'total':>9s}{'part':>7s}")
for i, nm in enumerate(SITES):
    print(f"{nm:28s}{tot[0, i]:14.2f}{tot[1, i]:22.2f}{tot[:, i].sum():9.2f}{100 * tot[:, i].sum() / tb:6.1f} %")
w = SITES.index("témoin identité (run)")
known = tot.sum() - tot[:, w].sum()
print(f"{'somme (hors témoin)':28s}{'':36s}{known:9.2f}{100 * known / tb:6.1f} %")
print(f"{'reste (conditions de boucle, _where, appel Python)':28s} {tb - known:9.2f}{100 * (tb - known) / tb:6.1f} %")
