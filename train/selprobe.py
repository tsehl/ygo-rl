"""Sonde supervisée : le tronc du réseau représente-t-il « quel candidat a la plus forte ATK » ?

Sépare deux explications d'un choix de cible au hasard (Monster Reborn, résurrections) : la représentation (le
tronc ne porte pas l'information) ou le signal de RL (elle y est, la tête de politique ne s'en sert pas).

1. collecte (`collect`) : parties jouées par un réseau fixe (joueur 0, tiré) contre l'heuristique, sur des banques
   procédurales d'évaluation riches en résurrections et sur les vraies cartes. On garde l'observation de chaque
   décision D_SELECT dont les candidats sont >= 2 monstres connus du cimetière, d'ATK de base distinctes.
   Les mêmes états servent à tous les checkpoints.
2. sonde (`probe`) : pour chaque checkpoint (et un réseau non entraîné), tronc figé → sortie par token ; une tête
   linéaire d → 1 (score par candidat, softmax sur les candidats) apprend à désigner le plus fort sur les banques
   procédurales d'apprentissage, testée sur une banque procédurale jamais vue et sur les vraies cartes.
   À côté : le choix de la politique elle-même sur les mêmes états, et le hasard.

Usage : selprobe.py collect CKPT OUT.npz [--b 128 --eval-steps 600]   (SP_BANKS=4, SP_REVIVE=4, SP_REAL=2 passes)
        selprobe.py probe OUT.npz CKPT [CKPT...]                      (« init » = réseau non entraîné)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "gen")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import core  # noqa: E402
import model  # noqa: E402
import obs as obsm  # noqa: E402
import ppo  # noqa: E402
import realcards  # noqa: E402

REBORN = 83764718
REAL = -1   # identifiant de banque des vraies cartes


def banks(a):
    nb, rv = int(os.environ.get("SP_BANKS", "4")), int(os.environ.get("SP_REVIVE", "4"))
    return [(a.eval_bank + i, rv) for i in range(nb)]


def collect(path, out, a):
    params = ppo.load_params(path)
    fns = ppo.make_fns(a, optax.adam(1e-4))

    @jax.jit
    def tick(bank, cemb, s, m, f, key):
        k1, k2 = jax.random.split(key)
        ae, tr = fns[0](params, cemb, bank, s, m, f, k1)
        a_ = jnp.where(s.dec_player == 0, ae, ppo.heuristic(bank, s, m, f, k2))
        o = tr["o"]
        c = jnp.clip(o.cid, 0, bank.atk.shape[0] - 1)
        cand = o.act[:, :, core.V_SELECT]
        ok = cand & o.known & (o.zone == core.Z_GRAVE) & ((bank.ctype[c] & core.TYPE_MONSTER) != 0)
        batk = bank.atk[c]
        hi, lo = jnp.where(cand, batk, -1).max(1), jnp.where(cand, batk, 1 << 30).min(1)
        keep = (s.dec == core.D_SELECT) & (f < 0) & ~s.done & (cand.sum(1) >= 2) & (ok == cand).all(1) & (hi > lo)
        src = jnp.maximum(s.ch_src, 0)
        reb = (s.ch_src >= 0) & (bank.code[jnp.take_along_axis(s.cid, src[:, None], 1)[:, 0]] == REBORN)
        s2, m2, f2 = core.step_batch(bank, s, a_, a.frac)
        return o, keep, reb, batk, s.dec_player, a_, s2, m2, f2

    rows = []

    def play(bid, bank, s, m, key):
        cemb = fns[5](params, *model.card_tokens(bank))
        f = jnp.full(a.b, -1, jnp.int32)
        n0 = len(rows)
        for t in range(a.eval_steps):
            key, k = jax.random.split(key)
            o, keep, reb, batk, pl, a_, s, m, f = tick(bank, cemb, s, m, f, k)
            g = np.nonzero(np.asarray(keep))[0]
            if len(g):
                on = jax.tree.map(lambda x: np.asarray(x)[g], o)
                rows.append(dict(o=on, bid=np.full(len(g), bid), reb=np.asarray(reb)[g], batk=np.asarray(batk)[g],
                                 pl=np.asarray(pl)[g], game=g + 100_000 * (len(rows) + 1)))
            if np.asarray(s.done).all():
                break
        print(f"banque {bid} : {sum(len(r['bid']) for r in rows[n0:])} décisions gardées en {t + 1} appels", flush=True)

    env = ppo.Env(a.b, a.frac, "structured")
    for i, (seed, rv) in enumerate(banks(a)):
        cset, bank = ppo.bank_of(seed, a, rv)
        key = jax.random.PRNGKey(100 + i)
        s, m = env.games(bank, cset, np.random.default_rng(100 + i), key)
        play(i, bank, s, m, key)
    bank, names, main_, extra = realcards.load()
    nd = len(names)
    for r in range(int(os.environ.get("SP_REAL", "2"))):
        cfg = np.array([(i, j) for i in range(nd) for j in range(nd)])[(np.arange(a.b) + r) % (nd * nd)]
        D, X = np.stack([main_[cfg[:, 0]], main_[cfg[:, 1]]], 1), np.stack([extra[cfg[:, 0]], extra[cfg[:, 1]]], 1)
        key = jax.random.PRNGKey(900 + r)
        s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, a.b))
        play(REAL, bank, s, m, key)
    flat = {}
    for k in ("bid", "reb", "batk", "pl", "game"):
        flat[k] = np.concatenate([r[k] for r in rows])
    for k in obsm.Obs._fields:
        flat["o_" + k] = np.concatenate([getattr(r["o"], k) for r in rows])
    np.savez_compressed(out, **flat)
    print(f"{out} : {len(flat['bid'])} décisions ({int((flat['bid'] == REAL).sum())} sur vraies cartes, "
          f"dont {int(flat['reb'].sum())} Monster Reborn)", flush=True)


def trunk(p, cemb, o, heads=4):
    """model.policy jusqu'à la sortie du tronc. Rend (entrée par token [.., T, d], sortie par token, logits)."""
    known = o.known[..., None]
    cid = jnp.clip(o.cid, 0, cemb.shape[0] - 1)
    x0 = jnp.where(known, cemb[cid, 0], p["unk"]) + model.dense(p["dyn"], model.dyn_feats(o))
    g = (model.dense(p["glob"], model.glob_feats(o)) + model.deck_feats(p, cemb, o))[..., None, :]
    x = jnp.concatenate([g, x0], -2)
    m = jnp.concatenate([jnp.ones(o.valid.shape[:-1] + (1,), bool), o.valid], -1)
    for b in p["blocks"]:
        x = model.block(b, x, m, heads)
    x = model.ln(p["out"], x)
    return x0, x[..., 1:, :], model.policy(p, cemb, o, heads)[0]


def fit(X, cand, best, tr, steps=400, lr=3e-2, wd=1e-3):
    """Tête linéaire sur les candidats : X [M, T, d], cand / best [M, T] bool, tr [M] bool (apprentissage).
    Rend le choix [M] (token)."""
    mu, sd = X[tr][cand[tr]].mean(0), X[tr][cand[tr]].std(0) + 1e-6
    Xn = jnp.asarray((X - mu) / sd)
    cand_, best_ = jnp.asarray(cand), jnp.asarray(best)
    w = jnp.asarray(tr, jnp.float32)

    def loss(q):
        lg = jnp.where(cand_, Xn @ q, -1e9)
        lp = jax.nn.log_softmax(lg, -1)
        ll = jax.nn.logsumexp(jnp.where(best_, lp, -1e9), -1)     # masse sur les candidats les plus forts
        return -(ll * w).sum() / w.sum() + wd * (q ** 2).sum()

    opt = optax.adam(lr)
    q = jnp.zeros(X.shape[-1])
    st = opt.init(q)

    @jax.jit
    def upd(q, st):
        g = jax.grad(loss)(q)
        u, st = opt.update(g, st)
        return optax.apply_updates(q, u), st
    for _ in range(steps):
        q, st = upd(q, st)
    return np.asarray(jnp.where(cand_, Xn @ q, -1e9).argmax(-1))


def probe(data, paths, a):
    z = np.load(data)
    o = obsm.Obs(**{k: z["o_" + k] for k in obsm.Obs._fields})
    bid, reb, batk = z["bid"], z["reb"], z["batk"]
    cand = o.act[:, :, core.V_SELECT]
    best = cand & (batk == np.where(cand, batk, -1).max(1, keepdims=True))
    chance = (best.sum(1) / cand.sum(1))
    M, T = cand.shape
    pb = sorted(b for b in set(bid.tolist()) if b != REAL)
    held = pb[-1]                                            # dernière banque procédurale : jamais vue par la sonde
    tr = (bid != REAL) & (bid != held)
    groups = [("procédural, apprentissage", tr), (f"procédural, banque tenue à l'écart", bid == held),
              ("vraies cartes, tout", bid == REAL), ("vraies cartes, Monster Reborn", (bid == REAL) & reb)]
    print(f"{M} décisions ; candidats par décision : {cand.sum(1).mean():.1f}")
    print(f"{'':34s}" + "".join(f"{n[:30]:>32s}" for n, _ in groups))
    print(f"{'occasions':34s}" + "".join(f"{int(g.sum()):>32d}" for _, g in groups))
    print(f"{'hasard':34s}" + "".join(f"{100 * chance[g].mean() if g.any() else float('nan'):>31.1f}%" for _, g in groups))
    bk = {}
    for i, (seed, rv) in enumerate(banks(a)):
        if i in pb:
            bk[i] = ppo.bank_of(seed, a, rv)[1]
    if (bid == REAL).any():
        bk[REAL] = realcards.load()[0]
    toks = {b: model.card_tokens(bank) for b, bank in bk.items()}
    enc = jax.jit(lambda p, tok, msk: model.encode_cards(p["cards"], tok, msk))
    tk = jax.jit(trunk)
    row = lambda name, pick, ref=best: print(f"{name:34s}" + "".join(
        f"{100 * ref[np.arange(M), pick][g].mean() if g.any() else float('nan'):>31.1f}%" for _, g in groups), flush=True)
    worst = cand & (batk == np.where(cand, batk, 1 << 30).min(1, keepdims=True))
    first = cand & (np.cumsum(cand, 1) == 1)                 # premier jeton candidat : le plus récent de son cimetière
    lv = z["o_level"]
    hilv = cand & (lv == np.where(cand, lv, -1).max(1, keepdims=True))
    for nm, ref in (("hasard : le plus faible", worst), ("hasard : premier jeton", first), ("hasard : plus haut niveau", hilv)):
        print(f"{nm:34s}" + "".join(f"{100 * (ref.sum(1) / cand.sum(1))[g].mean() if g.any() else float('nan'):>31.1f}%" for _, g in groups))
    for path in paths:
        if path == "init":
            p = model.init(jax.random.PRNGKey(0), a.d, a.card_layers, a.layers)
        else:
            p = ppo.load_params(path)
        d = p["glob"]["w"].shape[1]
        X0, X1, LG = np.zeros((M, T, d), np.float32), np.zeros((M, T, d), np.float32), np.zeros((M, T), np.float32)
        for b in bk:
            cemb = enc(p, *toks[b])
            idx = np.nonzero(bid == b)[0]
            for j in range(0, len(idx), 256):
                ii = idx[j:j + 256]
                pad = np.concatenate([ii, np.repeat(ii[-1:], 256 - len(ii))])   # lot de taille fixe : une compilation
                x0, x1, lg = tk(p, cemb, jax.tree.map(lambda x: jnp.asarray(x[pad]), o))
                lg = np.asarray(lg)[:, :T * core.NV].reshape(256, T, core.NV)[:, :, core.V_SELECT]
                X0[ii], X1[ii], LG[ii] = np.asarray(x0, np.float32)[:len(ii)], np.asarray(x1, np.float32)[:len(ii)], lg[:len(ii)]
        print(f"== {path}")
        pol = np.where(cand, LG, -1e9).argmax(1)
        row("politique : le plus fort", pol)
        row("politique : le plus faible", pol, worst)
        row("politique : premier jeton", pol, first)
        row("politique : plus haut niveau", pol, hilv)
        row("sonde linéaire, entrée du tronc", fit(X0, cand, best, tr))
        row("sonde linéaire, sortie du tronc", fit(X1, cand, best, tr))


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "collect":
        path, out = sys.argv.pop(1), sys.argv.pop(1)
        a = ppo.parse()
        model.set_compute(False)
        collect(path, out, a)
    else:
        data = sys.argv.pop(1)
        paths = []
        while len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
            paths.append(sys.argv.pop(1))
        a = ppo.parse()
        model.set_compute(False)
        probe(data, paths, a)
