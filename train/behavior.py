"""Tests de comportement sur les vraies cartes : le joueur testé (réseau tiré au hasard, réseau déterministe,
heuristique ou aléatoire) joue les 4 decks de référence contre l'heuristique. À chacune de ses décisions, on compte les
occasions et les erreurs de jeu de base (attaques perdues d'avance, fin de tour sans invoquer, mauvaise cible de Monster
Reborn, piège défensif pas activé...), plus, par carte, les activations et les fins de tour où elle restait jouable.

Usage : python train/behavior.py CKPT [--b 1024 --eval-steps 600 --bf16 ...] (mêmes options que ppo.py) ;
BEHAV_WHO=reseau,aleatoire pour ne tester que certains joueurs"""
import os
import sys
import time

import numpy as np

import ppo
import jax
import jax.numpy as jnp
import optax

import core
import model
import realcards
from ppo import load_params

C = dict(raigeki=12580477, dark_hole=53129443, mst=5318639, hfd=18144506, reborn=83764718,
         mirror=44095762, sakuretsu=56120475, trap_hole=4206964)
PROBES = [  # (nom, description de l'erreur)
    ("cible", "attaque une cible plus forte (ATK, recto) que l'attaquant"),
    ("cible_ratee", "ne prend pas une cible qu'il détruirait, alors qu'il y en a une"),
    ("attaque_perdue", "déclare une attaque alors que tous les monstres adverses (recto, ATK) sont plus forts"),
    ("directe_ratee", "terrain adverse vide, attaque possible, mais quitte la phase de combat"),
    ("fin_sans_invoquer", "finit son tour alors qu'il pouvait invoquer ou poser un monstre"),
    ("pas_le_plus_fort", "invoque depuis la main un monstre moins fort qu'un autre invocable"),
    ("magie_gardee", "finit son tour avec une magie activable en main"),
    ("dark_hole_perdant", "Dark Hole alors que son ATK totale sur le terrain dépasse celle de l'adversaire"),
    ("mst_sans_cible", "MST sans magie / piège adverse sur le terrain"),
    ("reborn_cible", "Monster Reborn : ne reprend pas le monstre le plus fort disponible"),
    ("piege_passe", "passe alors que Mirror Force / Sakuretsu / Trap Hole est activable"),
    ("pose_dominant", "pose en défense un monstre plus fort que tout le terrain adverse recto"),
]
K = len(PROBES)


def probe(bank, s, m, a):
    """Une partie : drapeaux [K, 2] (occasion, erreur) pour la décision a prise dans l'état s (masque légal m) ;
    puis [2, C] par carte de la banque : activée, gardée jouable en main en fin de tour."""
    n = s.cid.shape[0]
    me, op = s.dec_player, 1 - s.dec_player
    code = bank.code[s.cid]
    mon = s.zone == core.Z_MZONE
    my_mon, op_mon = mon & (s.ctrl == me), mon & (s.ctrl == op)
    fu = (s.pos & core.P_FACEUP) != 0
    fatk = s.pos == core.P_FU_ATK
    atk, df = s.atk_c, s.df_c
    batk = bank.atk[s.cid]
    hand = (s.zone == core.Z_HAND) & (s.ctrl == me)
    op_st = (s.zone == core.Z_SZONE) & (s.ctrl == op)
    nva = core.NV * n
    mv = m[:nva].reshape(core.NV, n)
    verb, slot = jnp.where(a < nva, a // n, -1), a % n
    sp = a - nva
    idle, bat = s.dec == core.D_IDLE, s.dec == core.D_BATTLE
    to_ep = idle & (sp == core.A_TO_EP)
    act = (verb >= core.V_ACT0)
    act_on = lambda c: act & (code[slot] == c)
    legal_act = mv[core.V_ACT0:].any(0)                                  # [n] carte activable (un de ses effets)

    # attaque : choix de la cible (D_TARGET, V_SELECT sur la cible)
    tg = (s.dec == core.D_TARGET) & (verb == core.V_SELECT)
    ak = atk[jnp.maximum(s.atk, 0)]
    kills = op_mon & fu & ((fatk & (atk < ak)) | (~fatk & (df < ak))) & mv[core.V_SELECT]
    f = [
        (tg, tg & fatk[slot] & (atk[slot] > ak)),
        (tg & kills.any(), tg & kills.any() & ~kills[slot]),
        (bat & (verb == core.V_ATTACK), bat & (verb == core.V_ATTACK) & op_mon.any()
         & (~op_mon | (fatk & (atk > atk[slot]))).all()),
        (bat & ~op_mon.any() & mv[core.V_ATTACK].any(), bat & ~op_mon.any() & mv[core.V_ATTACK].any()
         & ((sp == core.A_TO_M2) | (sp == core.A_TO_EP))),
        (to_ep, to_ep & ((mv[core.V_SUMMON] & hand).any() | mv[core.V_MSET].any())),
        (idle & (verb == core.V_SUMMON) & hand[slot], idle & (verb == core.V_SUMMON) & hand[slot]
         & (mv[core.V_SUMMON] & hand & (batk > batk[slot])).any()),
        (to_ep, to_ep & (legal_act & hand & ((bank.ctype[s.cid] & core.TYPE_SPELL) != 0)).any()),
        (act_on(C["dark_hole"]), act_on(C["dark_hole"]) & ((atk * my_mon).sum() > (atk * op_mon).sum())),
        (act_on(C["mst"]), act_on(C["mst"]) & ~op_st.any()),
    ]
    # Monster Reborn : cible choisie (D_SELECT pendant l'activation / la résolution de Reborn)
    rb = (s.dec == core.D_SELECT) & (verb == core.V_SELECT) & (s.ch_src >= 0) \
        & (bank.code[s.cid[jnp.maximum(s.ch_src, 0)]] == C["reborn"])
    f.append((rb, rb & (mv[core.V_SELECT] & (batk > batk[slot])).any()))
    trap = (code == C["mirror"]) | (code == C["sakuretsu"]) | (code == C["trap_hole"])
    tw = (s.dec == core.D_CHAIN) & (legal_act & trap).any()
    f.append((tw, tw & (sp == core.A_PASS)))
    ms = idle & (verb == core.V_MSET)
    f.append((ms, ms & (batk[slot] >= 1000) & ~(op_mon & fu & (atk >= batk[slot])).any()))
    flags = jnp.array([[o, e & o] for o, e in f], jnp.int32)
    nc = bank.code.shape[0]
    used = jnp.zeros(nc, jnp.int32).at[s.cid[slot]].add(act.astype(jnp.int32))
    held = jnp.zeros(nc, jnp.int32).at[s.cid].add((to_ep & legal_act & hand).astype(jnp.int32))
    return flags, jnp.stack([used, held])


def run(a, env, fns, params, bank, D, X, side, key, who):
    """who : 'reseau', 'reseau_det', 'heuristique' ou 'aleatoire' (joueur side) contre l'heuristique."""
    act = fns[0]
    tok, msk = model.card_tokens(bank)
    cemb = fns[5](params, tok, msk)

    @jax.jit
    def tick(s, m, f, key):
        k1, k2, k3 = jax.random.split(key, 3)
        if who.startswith("reseau"):
            mine = act(params, cemb, bank, s, m, f, k1, jnp.asarray(who == "reseau_det"))[0]
        elif who == "heuristique":
            mine = ppo.heuristic(bank, s, m, f, k1)
        else:
            mine = jnp.where(f >= 0, f, jax.random.categorical(k1, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))
        sd = jnp.asarray(side)
        a_ = jnp.where(s.dec_player == sd, mine, ppo.heuristic(bank, s, m, f, k2))
        dec = (s.dec_player == sd) & (f < 0) & ~s.done
        fl, pc = jax.vmap(lambda s, m, a: probe(bank, s, m, a))(s, m, a_)
        fl = (fl * dec[:, None, None]).sum(0)
        pc = (pc * dec[:, None, None]).sum(0)
        s, m, f = core.step_batch(bank, s, a_, a.frac)
        return s, m, f, fl, pc

    s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    fl_t, pc_t = 0, 0
    fin, res = np.zeros(env.b, bool), np.zeros(env.b, int)
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        s, m, f, fl, pc = tick(s, m, f, k)
        fl_t, pc_t = fl_t + np.asarray(fl), pc_t + np.asarray(pc)
        dn, w = np.asarray(s.done), np.asarray(s.winner)
        new = dn & ~fin
        res[new] = (w[new] == side[new]).astype(int) - (w[new] == 1 - side[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fl_t, pc_t, fin, res


def main():
    path = sys.argv[1]
    sys.argv = sys.argv[:1] + sys.argv[2:]
    a = ppo.parse()
    model.set_compute(a.bf16)
    params = load_params(path)
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    bank, names, main_, extra = realcards.load()
    nd = len(names)
    cfg = np.array([(i, j, s) for i in range(nd) for j in range(nd) for s in (0, 1)])[np.arange(env.b) % (2 * nd * nd)]
    p0 = np.where(cfg[:, 2] == 0, cfg[:, 0], cfg[:, 1])
    p1 = np.where(cfg[:, 2] == 0, cfg[:, 1], cfg[:, 0])
    D, X, side = np.stack([main_[p0], main_[p1]], 1), np.stack([extra[p0], extra[p1]], 1), cfg[:, 2]
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; {path}", flush=True)
    cnames = realcards.card_names()
    out = {}
    for who in os.environ.get("BEHAV_WHO", "reseau,reseau_det,heuristique,aleatoire").split(","):
        t0 = time.time()
        fl, pc, fin, res = run(a, env, fns, params, bank, D, X, side, jax.random.PRNGKey(3), who)
        out[who] = (fl, pc)
        win = (res[fin] == 1).mean() if fin.any() else float("nan")
        print(f"== {who} ({time.time() - t0:.0f} s) : {int(fin.sum())} parties finies, victoire contre l'heuristique "
              f"{win:.3f}", flush=True)
    print("\nerreur (% des occasions ; occasions) :", flush=True)
    print(f"{'':20s}" + "".join(f"{w:>22s}" for w in out), flush=True)
    for k, (nm, desc) in enumerate(PROBES):
        cells = "".join(f"{100 * o[0][k, 1] / max(o[0][k, 0], 1):14.1f} ({o[0][k, 0]:>6d})" for o in out.values())
        print(f"{nm:20s}{cells}   {desc}", flush=True)
    code = np.asarray(bank.code)
    print("\npar carte (activations / fins de tour où elle restait activable en main) :", flush=True)
    tot = sum(o[1] for o in out.values())
    for c in np.argsort(-tot[1]):
        if tot[0, c] + tot[1, c] == 0 or not cnames[c]:
            continue
        nm = cnames[c]
        print(f"{nm[:32]:32s}" + "".join(f"{o[1][0, c]:>10d} / {o[1][1, c]:<9d}" for o in out.values()), flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
