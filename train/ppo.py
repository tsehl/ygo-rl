"""PPO en self-play sur cartes et decks procéduraux (jalon 6, étape 3 ; ETAPE3.md).

Boucle :
- une banque procédurale (gen/cardgen.py) toutes les `--bank-every` itérations, decks de 40 + Extra Deck de 15 tirés
  sans structure ; B parties en parallèle (`core.step_batch`) ; une partie finie est remplacée par une neuve ;
- les deux joueurs jouent avec la même politique ; chaque décision est stockée avec son joueur. Les avantages (GAE)
  sont calculés par joueur : en remontant le temps, chaque partie garde pour chacun des deux joueurs la valeur et
  l'avantage de sa décision suivante ; la fin de partie donne +1 au gagnant, -1 au perdant (0 : nulle) ;
- PPO (clip), minibatchs sur l'axe des parties ; l'encodage des cartes de la banque est recalculé dans chaque
  minibatch (gradient jusqu'aux briques) ;
- évaluation : le réseau (échantillonnage) contre la politique aléatoire uniforme, moitié des parties de chaque
  côté, sur une banque tenue à l'écart (graine `--eval-bank`), jamais vue à l'entraînement.

Usage (test CPU) : uv run --with jax --with optax python train/ppo.py --b 32 --steps 16 --iters 3 --d 64
"""
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

if os.environ.get("YGO_JAX_CACHE"):
    jax.config.update("jax_compilation_cache_dir", os.environ["YGO_JAX_CACHE"])


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=256, help="parties en parallèle")
    ap.add_argument("--steps", type=int, default=64, help="appels de step_batch par rollout")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--card-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=4, help="minibatchs par époque")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--frac", type=float, default=0.1, help="arrêt anticipé de step_batch (optimum mesuré sur L4 le 2026-11-04)")
    ap.add_argument("--bank-every", type=int, default=10)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--old-lag", type=int, default=50, help="évaluation aussi contre la politique d'il y a X itérations")
    ap.add_argument("--eval-b", type=int, default=0, help="parties d'évaluation (défaut : b)")
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--bench-every", type=int, default=50, help="benchmark sur les vraies cartes (multiple de --eval-every)")
    ap.add_argument("--eval-bank", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--bf16", action="store_true", help="calcul du réseau en bfloat16 (paramètres et optimiseur en f32)")
    ap.add_argument("--prof", type=int, default=0, help="à la fin : chronométrer act, step_batch, act_step... X fois sur l'état du rollout")
    return ap.parse_args()


class Env:
    """Fonctions compilées du moteur (la banque est un argument : une compilation pour toutes les banques)."""

    def __init__(self, b, frac):
        self.b = b
        ng = jax.vmap(lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1]), in_axes=(None, 0, 0, 0))
        lg = jax.vmap(lambda bank, s: core.legal(bank, s), in_axes=(None, 0))
        self.new = jax.jit(lambda bank, d, x, k: (lambda s: (s, lg(bank, s)))(ng(bank, d, x, k)))
        self.step = jax.jit(lambda bank, s, a: core.step_batch(bank, s, a, frac))

    def games(self, bank, cset, rng, key):
        d, x = cardgen.random_decks(cset, rng, self.b)
        return self.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, self.b))


def bank_of(seed, a):
    cset = cardgen.Gen(seed).make_set(a.main, a.extra)
    bank = cardgen.to_bank(cset)
    nt = int(model.n_tokens(model.card_tokens(bank)[1]).max())
    assert nt <= model.LC, f"banque {seed} : une carte a {nt} tokens utiles > model.LC = {model.LC}"
    return cset, bank


def _where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


# adversaire heuristique : poids par verbe (V_SUMMON, V_MSET, V_REPOS, V_ATTACK, V_SELECT, V_SSET, puis V_ACT0 + e)
H_VERB = [4.0, 1.5, 0.3, 3.0, 1.0, 2.0] + [5.0] * core.E
# actions spéciales : vers BP, vers M2, vers EP, 7 zones, position ATK / DEF, passer, oui
H_SPECIAL = [2.5, 0.5, 0.2] + [1.0] * 7 + [1.2, 0.8, 0.1, 1.0]


def heuristic(bank, s, m, f, key):
    """Adversaire de référence simple, plus exigeant que l'aléatoire : active tout ce qui peut l'être, invoque (le plus
    fort d'abord), attaque avec tout, pose ses magies / pièges, puis passe ; choix (cibles, matériaux, zones) au hasard.
    Rend l'action du moteur [B] (l'action forcée si f >= 0)."""
    n = s.cid.shape[1]
    assert len(H_VERB) == core.NV and len(H_SPECIAL) == core.N_SPECIAL
    w = jnp.concatenate([jnp.repeat(jnp.asarray(H_VERB), n), jnp.asarray(H_SPECIAL)])          # [A]
    atk = bank.atk[s.cid].astype(jnp.float32) / 10000.0                                          # [B, n]
    vb = jnp.zeros((s.cid.shape[0], core.NV, n)).at[:, core.V_SUMMON].set(atk).at[:, core.V_ATTACK].set(atk)
    bonus = jnp.concatenate([vb.reshape(s.cid.shape[0], -1), jnp.zeros((s.cid.shape[0], core.N_SPECIAL))], -1)
    sc = w + bonus + 0.01 * jax.random.gumbel(key, m.shape)
    return jnp.where(f >= 0, f, jnp.argmax(jnp.where(m, sc, -1e9), -1).astype(jnp.int32))


def make_fns(a, opt):
    obs_b = jax.vmap(lambda bank, s, m, p: obsm.observe(bank, s, p, m), in_axes=(None, 0, 0, 0))

    def _act(params, cemb, bank, s, m, f, key):
        n = s.cid.shape[1]
        o = obs_b(bank, s, m, s.dec_player)
        logits, v = model.policy(params, cemb, o)
        idx = jax.random.categorical(key, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), idx[:, None], -1)[:, 0]
        need = (f < 0) & ~s.done
        ae = jnp.where(f >= 0, f, model.to_engine(o, n, idx))
        return ae, dict(o=o, idx=idx, logp=logp, v=v, need=need, pl=s.dec_player)

    act = jax.jit(_act)

    @jax.jit
    def values(params, cemb, bank, s, m):
        vs = [model.policy(params, cemb, obs_b(bank, s, m, jnp.full_like(s.dec_player, p)))[1] for p in (0, 1)]
        return jnp.stack(vs, -1)                                                       # [B, 2]

    def refill(s, m, f, s2, m2, used):
        """Parties finies remplacées par la partie de la réserve de même indice, si elle n'a pas déjà servi."""
        d = s.done & ~used
        return _where(d, s2, s), _where(d, m2, m), jnp.where(d, -1, f), used | d

    @jax.jit
    def act_step(params, cemb, bank, s, m, f, key, side, pool, pm, used, opp=None):
        """Un appel du rollout, en une seule fonction compilée (pas de synchronisation hôte) : politique, step_batch,
        puis remplacement des parties finies par la réserve. side [B] : -1 = le réseau joue les deux côtés ; sinon le
        réseau joue le joueur `side` et l'adversaire l'autre (évaluation, avec used = vrai : pas de remplacement).
        opp = (paramètres, cemb, kind) : adversaire aléatoire uniforme (kind 0), réseau (1) ou heuristique (2) ;
        lax.switch sur un scalaire hors vmap : une seule branche calculée. Rend aussi done / winner juste après
        step_batch."""
        k1, k2 = jax.random.split(key)
        ae, tr = _act(params, cemb, bank, s, m, f, k1)
        rnd = lambda: jnp.where(f >= 0, f, jax.random.categorical(k2, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))
        if opp is None:
            ar = rnd()
        else:
            ar = jax.lax.switch(opp[2], [rnd, lambda: _act(opp[0], opp[1], bank, s, m, f, k2)[0],
                                         lambda: heuristic(bank, s, m, f, k2)])
        s, m, f = core.step_batch(bank, s, jnp.where((side < 0) | (s.dec_player == side), ae, ar), a.frac)
        done, win = s.done, s.winner
        s, m, f, used = refill(s, m, f, pool, pm, used)
        return s, m, f, used, tr, done, win

    @jax.jit
    def gae(v, need, pl, done, win, last):
        """v, need, pl, done, win : [S, B] ; last : [B, 2]. Rend (avantages, retours) [S, B]."""
        g, lam = a.gamma, a.lam
        pid = jnp.arange(2)

        def body(c, x):
            nv, adv, pend = c
            v, need, pl, done, win = x
            r = jnp.where(win[:, None] < 0, 0.0, jnp.where(win[:, None] == pid, 1.0, -1.0))
            dd = done[:, None]
            nv, adv, pend = jnp.where(dd, 0.0, nv), jnp.where(dd, 0.0, adv), jnp.where(dd, r, pend)
            sel = (pid == pl[:, None]) & need[:, None]                                 # [B, 2]
            pick = lambda t: (t * sel).sum(-1)
            delta = pick(pend) + g * pick(nv) - v
            A = delta + g * lam * pick(adv)
            nv, adv, pend = jnp.where(sel, v[:, None], nv), jnp.where(sel, A[:, None], adv), jnp.where(sel, 0.0, pend)
            return (nv, adv, pend), jnp.where(need, A, 0.0)
        z = jnp.zeros_like(last)
        _, A = jax.lax.scan(body, (last, z, z), (v, need, pl, done, win), reverse=True)
        return A, A + v

    def loss_fn(params, tok, msk, bt):
        cemb = model.encode_cards(params["cards"], tok, msk)
        logits, v = model.policy(params, cemb, bt["o"])
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, bt["idx"][..., None], -1)[..., 0]
        w = bt["need"].astype(jnp.float32)
        nw = jnp.maximum(w.sum(), 1.0)
        A = bt["adv"]
        ratio = jnp.exp(lp - bt["logp"])
        pg = -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * A)
        p_all = jnp.exp(lp_all)
        ent = -(p_all * jnp.where(p_all > 0, lp_all, 0.0)).sum(-1)
        vl = (v - bt["ret"]) ** 2
        tot = ((pg + a.vf * vl - a.ent * ent) * w).sum() / nw
        stats = dict(pg=(pg * w).sum() / nw, vl=(vl * w).sum() / nw, ent=(ent * w).sum() / nw,
                     kl=((bt["logp"] - lp) * w).sum() / nw, clipfrac=((jnp.abs(ratio - 1) > a.clip) * w).sum() / nw)
        return tot, stats

    @jax.jit
    def update(params, opt_state, tok, msk, bt):
        (l, st), gr = jax.value_and_grad(loss_fn, has_aux=True)(params, tok, msk, bt)
        up, opt_state = opt.update(gr, opt_state, params)
        st["gnorm"] = optax.global_norm(gr)
        return optax.apply_updates(params, up), opt_state, st

    enc = jax.jit(lambda params, tok, msk: model.encode_cards(params["cards"], tok, msk))
    return act, values, act_step, gae, update, enc, obs_b


def paired(d, x):
    """Decks [P, 2, ...] → [4P, 2, ...] et côté du réseau [4P] : chaque paire de decks (A, B) est jouée 4 fois, le réseau
    avec A puis avec B, en premier puis en second. La force des decks et l'avantage du premier joueur s'annulent."""
    sw = lambda z: z[:, ::-1]
    D = np.stack([d, d, sw(d), sw(d)], 1).reshape((-1,) + d.shape[1:])
    X = np.stack([x, x, sw(x), sw(x)], 1).reshape((-1,) + x.shape[1:])
    return D, X, np.tile([0, 1, 0, 1], len(d))


def play_eval(a, env, fns, params, bank, d, x, side, key, kind, oparams=None):
    """Parties d [B, 2, 40], x [B, 2, 15] ; le réseau joue le joueur side[b], l'adversaire est aléatoire (kind 0), le
    réseau oparams (1) ou l'heuristique (2). Rend (finies [B], résultat [B] : 1 victoire, -1 défaite, 0 nul / pas fini)."""
    act, values, act_step, gae, update, enc, obs_b = fns
    tok, msk = model.card_tokens(bank)
    cemb = enc(params, tok, msk)
    opp = (params if oparams is None else oparams, cemb if oparams is None else enc(oparams, tok, msk), jnp.asarray(kind))
    s, m = env.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    side = jnp.asarray(side)
    sd = np.asarray(side)
    fin = np.zeros(env.b, bool)
    res = np.zeros(env.b, int)
    used = jnp.ones(env.b, bool)                       # pas de remplacement des parties finies
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        s, m, f, used, _, dn, w = act_step(params, cemb, bank, s, m, f, k, side, s, m, used, opp)
        dn, w = np.asarray(dn), np.asarray(w)
        new = dn & ~fin
        res[new] = (w[new] == sd[new]).astype(int) - (w[new] == 1 - sd[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fin, res


def summary(fin, res, pre="eval"):
    n = max(int(fin.sum()), 1)
    return {f"{pre}_games": int(fin.sum()), f"{pre}_win": float((res[fin] == 1).sum() / n),
            f"{pre}_loss": float((res[fin] == -1).sum() / n)}


def evaluate(a, env, fns, params, key, kind, oparams=None, pre="eval"):
    """Banque procédurale d'évaluation (jamais vue), decks aléatoires joués par paires (voir paired), graine fixe."""
    cset, bank = bank_of(a.eval_bank, a)
    d, x = cardgen.random_decks(cset, np.random.default_rng(12345), env.b // 4)
    D, X, side = paired(d, x)
    return summary(*play_eval(a, env, fns, params, bank, D, X, side, key, kind, oparams), pre)


_REAL = {}


def real_bench(a, env, fns, params, key, kind, oparams=None, pre="real"):
    """Vraies cartes (train/realcards.py) : toutes les rencontres (deck du réseau, deck adverse, côté), répétées pour
    remplir le lot. Rend le taux de victoire global et par deck du réseau."""
    import realcards
    if not _REAL:
        _REAL["v"] = realcards.load()
    bank, names, main, extra = _REAL["v"]
    nd = len(names)
    cfg = np.array([(i, j, sd) for i in range(nd) for j in range(nd) for sd in (0, 1)])
    cfg = cfg[np.arange(env.b) % len(cfg)]
    i, j, side = cfg[:, 0], cfg[:, 1], cfg[:, 2]
    p0, p1 = np.where(side == 0, i, j), np.where(side == 0, j, i)
    D = np.stack([main[p0], main[p1]], 1)
    X = np.stack([extra[p0], extra[p1]], 1)
    fin, res = play_eval(a, env, fns, params, bank, D, X, side, key, kind, oparams)
    out = summary(fin, res, pre)
    for k, nm in enumerate(names):
        sel = fin & (i == k)
        out[f"{pre}_{nm}"] = float((res[sel] == 1).sum() / max(int(sel.sum()), 1))
    return out


def fmt(ev):
    return " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items())


def main():
    a = parse()
    a.eval_b = a.eval_b or a.b
    model.set_compute(a.bf16)
    key = jax.random.PRNGKey(a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} steps={a.steps} d={a.d}", flush=True)
    params = model.init(jax.random.fold_in(key, 0), a.d, a.card_layers, a.layers)
    print("paramètres :", sum(x.size for x in jax.tree.leaves(params)), flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = opt.init(params)
    snaps = {0: params}                                # versions passées pour l'évaluation (itération → paramètres)
    env = Env(a.b, a.frac)
    eval_env = env if a.eval_b == a.b else Env(a.eval_b, a.frac)
    fns = act, values, act_step, gae, update, enc, obs_b = make_fns(a, opt)
    bank_seed = a.seed * 100_000
    tot_dec, tot_games = 0, 0
    for it in range(a.iters):
        t0 = time.time()
        t_games = 0.0                         # parties neuves : decks NumPy + new_game + legal (hors boucle comprise)

        def games(k, prev):
            """env.games chronométré à part : on attend d'abord le travail GPU déjà en file (prev), sinon il serait compté."""
            jax.block_until_ready(prev)
            t1 = time.time()
            r = env.games(bank, cset, rng, k)
            jax.block_until_ready(r)
            return r, time.time() - t1

        if it % a.bank_every == 0:
            cset, bank = bank_of(bank_seed, a)
            bank_seed += 1
            tok, msk = model.card_tokens(bank)
            key, k = jax.random.split(key)
            (s, m), dt = games(k, params)
            t_games += dt
            f = jnp.full(a.b, -1, jnp.int32)
        key, k = jax.random.split(key)
        (pool, pm), dt = games(k, params)
        t_games += dt
        used = jnp.zeros(a.b, bool)
        cemb = enc(params, tok, msk)
        buf, dones, wins = [], [], []
        side = jnp.full(a.b, -1, jnp.int32)
        opp0 = (params, cemb, jnp.asarray(0))
        jax.block_until_ready(cemb)
        t1 = time.time()
        t_in = 0.0                            # parties neuves pendant la boucle (à retrancher de la boucle)
        for t in range(a.steps):
            key, k = jax.random.split(key)
            s, m, f, used, tr, d, w = act_step(params, cemb, bank, s, m, f, k, side, pool, pm, used, opp0)
            buf.append(tr)
            dones.append(d)
            wins.append(w)
            if t % 8 == 7 and float(used.mean()) > 0.25:      # réserve neuve : chaque partie de la réserve ne sert qu'une fois
                key, k = jax.random.split(key)
                (pool, pm), dt = games(k, f)
                t_in += dt
                used = jnp.zeros(a.b, bool)
        jax.block_until_ready(f)
        t_games += t_in
        t_loop = time.time() - t1 - t_in      # les 64 appels d'act_step seuls (+ les synchronisations tous les 8 appels)
        last = values(params, cemb, bank, s, m)
        bt = jax.tree.map(lambda *x: jnp.stack(x), *buf)
        done, win = jnp.stack(dones), jnp.stack(wins)
        adv, ret = gae(bt["v"], bt["need"], bt["pl"], done, win, last)
        need = bt["need"]
        nd = int(need.sum())
        mu = (adv * need).sum() / max(nd, 1)
        sd = jnp.sqrt((((adv - mu) * need) ** 2).sum() / max(nd, 1)) + 1e-8
        bt["adv"], bt["ret"] = (adv - mu) / sd, ret
        t_roll = time.time() - t0
        st = None
        for ep in range(a.epochs):
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, a.b)
            for chunk in jnp.split(perm, a.mb):
                mb = jax.tree.map(lambda x: x[:, chunk], bt)
                params, opt_state, st = update(params, opt_state, tok, msk, mb)
        jax.block_until_ready(params)
        ng = int(done.sum())
        tot_dec += nd
        tot_games += ng
        line = (f"it {it} : {nd} décisions ({nd / (time.time() - t0):.0f}/s), {ng} parties finies ({time.time() - t0:.1f} s ; "
                f"rollout {t_roll:.1f} s, dont parties neuves {t_games:.1f} s, act_step {t_loop:.1f} s = "
                f"{1000 * t_loop / a.steps:.0f} ms/appel) | " + " ".join(f"{k} {float(v):.3f}" for k, v in st.items()))
        if a.eval_every and (it + 1) % a.eval_every == 0:
            key, k1, k2, k3 = jax.random.split(key, 4)
            line += " | " + fmt(evaluate(a, eval_env, fns, params, k1, 0, pre="rnd"))
            line += " | " + fmt(evaluate(a, eval_env, fns, params, k2, 2, pre="heur"))
            # contre la version d'il y a --old-lag itérations (ou la plus ancienne gardée, au début : l'initialisation)
            old = max([i for i in snaps if i <= it + 1 - a.old_lag] or [min(snaps)])
            ev = evaluate(a, eval_env, fns, params, k3, 1, snaps[old])
            line += f" | contre it {old} : win {ev['eval_win']:.3f} loss {ev['eval_loss']:.3f}"
            if a.bench_every and (it + 1) % a.bench_every == 0:
                key, k1, k2 = jax.random.split(key, 3)
                line += " | " + fmt(real_bench(a, eval_env, fns, params, k1, 0, pre="vrai_rnd"))
                line += " | " + fmt(real_bench(a, eval_env, fns, params, k2, 2, pre="vrai_heur"))
            snaps[it + 1] = params
            snaps = {i: q for i, q in snaps.items() if i >= it + 1 - 2 * a.old_lag}
        print(line, flush=True)
        if a.save:
            import pickle
            with open(a.save, "wb") as fh:
                pickle.dump(jax.device_get(params), fh)
    print(f"total : {tot_dec} décisions, {tot_games} parties", flush=True)
    if a.prof:
        profile(a, fns, params, bank, tok, msk, s, m, f, pool, pm, used, key)


def profile(a, fns, params, bank, tok, msk, s, m, f, pool, pm, used, key):
    """Chaque morceau d'un appel du rollout chronométré seul, a.prof fois sur le même état (fonctions pures) : l'état de
    fin de rollout (milieu de partie) et la réserve (débuts de partie)."""
    act, values, act_step, gae, update, enc, obs_b = fns
    cemb = enc(params, tok, msk)
    sb = jax.jit(lambda bank, s, x: core.step_batch(bank, s, x, a.frac))
    side = jnp.full(a.b, -1, jnp.int32)
    opp0 = (params, cemb, jnp.asarray(0))

    def tm(name, fn, *xs):
        r = fn(*xs)
        jax.block_until_ready(r)
        t0 = time.time()
        for _ in range(a.prof):
            r = fn(*xs)
        jax.block_until_ready(r)
        print(f"prof {name:45s} {1000 * (time.time() - t0) / a.prof:7.1f} ms", flush=True)
        return r

    for lab, (s1, m1) in (("milieu", (s, m)), ("début", (pool, pm))):
        f1 = f if lab == "milieu" else jnp.full(a.b, -1, jnp.int32)
        print(f"prof {lab} de partie : tour moyen {float(s1.turn.mean()):.1f}, finies {int(s1.done.sum())}", flush=True)
        ae, _ = tm(f"{lab} act", act, params, cemb, bank, s1, m1, f1, key)
        rnd = jax.jit(lambda k, m, f: jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9)).astype(jnp.int32)))
        ar = rnd(key, m1, f1)
        tm(f"{lab} step_batch (actions du réseau)", sb, bank, s1, ae)
        tm(f"{lab} step_batch (actions aléatoires)", sb, bank, s1, ar)
        tm(f"{lab} act_step", act_step, params, cemb, bank, s1, m1, f1, key, side, pool, pm, used, opp0)


if __name__ == "__main__":
    main()
