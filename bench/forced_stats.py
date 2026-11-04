"""Diagnostic : actions forcées (option unique, jouées sans la politique) par type de décision, selon l'avancement des
parties. Une itération = une action (politique aléatoire ou forcée) puis résolution, comme un tour de la boucle de step.

Usage : python bench/forced_stats.py [--b 64] [--iters 3000] [--proc 0]
Sortie : pour les tours 1-5, 6-15 et 16+, actions forcées par décision de la politique, par (décision, fenêtre)."""
import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.join(HERE, "..", "gen"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=64)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--proc", type=int, default=0)
a = ap.parse_args()

cset = cardgen.Gen(a.proc).make_set(400, 80)
bank = cardgen.to_bank(cset)
d, x = cardgen.random_decks(cset, np.random.default_rng(1), a.b)
ng = jax.jit(jax.vmap(lambda d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])))


def cheap_ok(s):
    """[E, N] : parties bon marché d'act_ok (verrou de nom, une fois par tour par nom / par carte)."""
    p = s.dec_player
    rows = []
    for e in range(core.E):
        rows.append(~core.name_in(s.lock_cid[p], s.cid) & ~((bank.eopt[s.cid, e] == core.OPT_NAME) & core.name_in(s.opt_cid[p], s.cid))
                    & ~((bank.eopt[s.cid, e] >= core.OPT_CARD) & ((s.st & ((core.ST_CNT | core.ST_CNTN) << e)) != 0)))
    return jnp.stack(rows)


diag = jax.jit(jax.vmap(lambda s: ((core.act_cand(bank, s) & cheap_ok(s)).any(), core.act_cand(bank, s).sum())))


@jax.jit
def it(s, f, key):
    m = jax.vmap(lambda s: core.legal(bank, s))(s)
    r = jax.random.categorical(key, jnp.where(m, 0.0, -1e9)).astype(jnp.int32)
    a = jnp.where(f >= 0, f, r)
    s = jax.vmap(lambda s, a: core.refresh_ex(bank, core.settle(bank, core.refresh_if(bank, s, core.apply(bank, s, a)))))(s, a)
    return s, jax.vmap(lambda s: core.forced(bank, s))(s)


s = ng(jnp.asarray(d), jnp.asarray(x), jax.random.split(jax.random.PRNGKey(0), a.b))
f = jax.vmap(lambda s: core.forced(bank, s))(s)
key = jax.random.PRNGKey(1)
DN = "NONE IDLE BATTLE TRIBUTE ZONE TARGET DISCARD SELECT POS RUN CHAIN TRIG YESNO".split()
cnt = {b: collections.Counter() for b in range(3)}
pol = np.zeros(3)
for t in range(a.iters):
    live = ~np.asarray(s.done)
    if not live.any():
        break
    turn = np.asarray(s.turn)
    bucket = np.where(turn <= 5, 0, np.where(turn <= 15, 1, 2))
    fo = np.asarray(f) >= 0
    dec, wk, ret = np.asarray(s.dec), np.asarray(s.wk), np.asarray(s.ret)
    cheap, ncand = (np.asarray(z) for z in diag(s))
    for g in np.nonzero(live)[0]:
        if fo[g] and DN[dec[g]] == "CHAIN":
            cnt[bucket[g]][("cheap_ok vide" if not cheap[g] else "test_ok", -1, -1)] += 1
            cnt[bucket[g]][(f"ncand={min(int(ncand[g]), 5)}", -2, -1)] += 1
        if fo[g]:
            cnt[bucket[g]][(DN[dec[g]], int(wk[g]), int(ret[g]) if DN[dec[g]] == "CHAIN" else -1)] += 1
        else:
            pol[bucket[g]] += 1
    key, k = jax.random.split(key)
    s, f = it(s, f, k)
for b, name in enumerate(("tours 1-5", "tours 6-15", "tours 16+")):
    tot = sum(cnt[b].values())
    print(f"{name} : {int(pol[b])} décisions de la politique, {tot} forcées ({tot / max(pol[b], 1):.2f} par décision)")
    for (dn, wk, ret), n in cnt[b].most_common(10):
        print(f"   {dn:8s} wk={wk} ret={ret:3d} : {n / max(pol[b], 1):.3f} par décision")
