"""Comparaison moteur JAX ↔ oracle ygopro-core sur des parties aléatoires.

Decks : 40 monstres normaux, ou (mode « mixed ») 28 monstres normaux + 12 cartes à effet de `cards.CARDS`, ou
(mode « chain ») 22 monstres normaux + 18 cartes à effet dont beaucoup de pièges / jeu-rapide (longues chaînes), ou
(mode « trig ») 20 monstres normaux + 10 monstres à effet déclenché + 10 autres cartes à effet, ou (mode « cont »)
les cartes du jalon 4 (archétypes, effets continus, modificateurs temporaires) avec des E-HERO et des Dark Magician,
ou (mode « dmg ») les cartes du volet 2 (damage step, SET_ATTACK*) avec quelques cartes du volet 1 ; le mode « ds »
reprend ces decks mais garde le plus souvent les cartes du volet 2 pour les fenêtres du damage step.

À chaque décision à plusieurs options de l'oracle : on compare l'état (zones, positions, ATK/DEF, LP, tour,
phase) et l'ensemble des actions légales, puis on joue la même option aléatoire dans les deux moteurs.
Les options « annuler » / « retirer » de l'oracle sont exclues (le moteur JAX n'en a pas).
Usage : uv run --with jax python compare.py [n_parties] [seed0] [normal|mixed|chain|trig|quick|cont|dmg|ds]
"""
import os
import collections
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../oracle"))
import driver  # noqa: E402
from driver import K  # noqa: E402

import cards  # noqa: E402
import core  # noqa: E402
from core import NV, V_ACT0, V_ATTACK, V_MSET, V_REPOS, V_SELECT, V_SSET, V_SUMMON  # noqa: E402

ZNAME = {"deck": core.Z_DECK, "hand": core.Z_HAND, "mzone": core.Z_MZONE, "szone": core.Z_SZONE,
         "grave": core.Z_GRAVE, "removed": core.Z_REMOVED, "extra": core.Z_EXTRA}
D = 40
N = 2 * D


class Mismatch(Exception):
    pass


def make_bank(db, codes):
    codes = sorted(set(codes))
    r = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [x[4] for x in r], [x[5] for x in r], [x[6] & 0xFF for x in r],
                          ctype=[x[3] for x in r], race=[x[7] for x in r], attr=[x[8] for x in r],
                          setcode=[cards.setcodes(x[2]) for x in r], **cards.assemble(codes))
    return bank, {c: i for i, c in enumerate(codes)}


def jax_snapshot(bank, s):
    s = jax.tree.map(np.asarray, s)
    code, atk, df = np.asarray(bank.code)[s.cid], s.atk_c, s.df_c
    out = dict(lp=tuple(int(x) for x in s.lp), turn=int(s.turn), turn_player=int(s.tp), phase=int(s.phase))
    for p in (0, 1):
        for name, z in ZNAME.items():
            idx = np.where((s.zone == z) & (s.ctrl == p))[0]
            if z in (core.Z_MZONE, core.Z_SZONE):
                lst = [None] * (7 if z == core.Z_MZONE else 8)
                for i in idx:
                    lst[s.seq[i]] = (int(code[i]), int(s.pos[i]), int(atk[i]), int(df[i]))
            else:
                lst = [(int(code[i]), int(s.pos[i]), int(atk[i]), int(df[i])) for i in idx[np.argsort(s.seq[idx])]]
                if len(set(s.seq[idx])) != len(idx) or (len(idx) and s.seq[idx].max() != len(idx) - 1):
                    raise Mismatch(f"seq incohérents en {p},{name}: {sorted(s.seq[idx])}")
            out[(p, name)] = lst
    return out


def slot(s, con, loc, seq):
    m = (np.asarray(s.zone) == ZNAME[loc]) & (np.asarray(s.ctrl) == con) & (np.asarray(s.seq) == seq)
    idx = np.where(m)[0]
    if len(idx) != 1:
        raise Mismatch(f"pas de slot unique pour {con},{loc},{seq}")
    return int(idx[0])


def translate(dec, opt, s):
    """Option de l'oracle → suite d'actions JAX ; None si l'option est exclue."""
    kind = opt[0]
    sp = lambda k: NV * N + k
    if kind in ("summon", "mset", "repos", "attack", "sset", "activate", "chain"):
        verb = dict(summon=V_SUMMON, mset=V_MSET, repos=V_REPOS, attack=V_ATTACK, sset=V_SSET, activate=V_ACT0,
                    chain=V_ACT0)[kind]
        return [verb * N + slot(s, opt[2], opt[3], opt[4])]
    if kind == "to_bp":
        return [sp(core.A_TO_BP)]
    if kind == "to_m2":
        return [sp(core.A_TO_M2)]
    if kind == "to_ep":
        return [sp(core.A_TO_EP)]
    if kind == "place":
        if opt[1] != dec.player or opt[2] not in ("mzone", "szone"):
            raise Mismatch(f"place inattendu {opt}")
        return [sp(core.A_ZONE0 + opt[3])]
    if kind == "pos":
        return [sp({core.P_FU_ATK: core.A_POS_ATK, core.P_FU_DEF: core.A_POS_DEF}[opt[1]])]
    if kind == "cards":
        return [V_SELECT * N + slot(s, c[1], c[2], c[3]) for c in opt[1]]
    if kind == "pick":
        c = opt[1]
        return [V_SELECT * N + slot(s, c[1], c[2], c[3])]
    if kind in ("cancel", "unpick"):
        return None
    if kind == "finish":
        return []
    if kind == "pass":
        return [sp(core.A_PASS)]
    raise Mismatch(f"option non gérée {opt}")


def filtered(dec):
    """Indices des options retenues (sans annuler / retirer ; « terminer » seul dès qu'il est possible)."""
    idx = [i for i, o in enumerate(dec.options) if o[0] not in ("cancel", "unpick")]
    if any(dec.options[i][0] == "finish" for i in idx):
        idx = [i for i in idx if dec.options[i][0] == "finish"]
    return idx


def resync_hand(s, so, players):
    """ShuffleHand mélange la main de l'oracle avec son vrai RNG : on recopie l'ordre (par code) côté JAX."""
    seq = np.array(s.seq)
    zone, ctrl, code = np.asarray(s.zone), np.asarray(s.ctrl), np.asarray(BANK_CODE)[np.asarray(s.cid)]
    for p in players:
        free = [i for i in np.where((zone == core.Z_HAND) & (ctrl == p))[0]]
        for k, c in enumerate(so[(p, "hand")]):
            j = next((i for i in free if code[i] == c[0]), None)
            if j is None:
                raise Mismatch(f"main de {p} : {c[0]} introuvable après mélange")
            free.remove(j)
            seq[j] = k
    return s._replace(seq=jnp.asarray(seq))


BANK_CODE = None
COVER = collections.Counter()  # options jouées (activations par carte, poses, mélanges…)
# compilé une fois : appelé hors jit, le lax.cond final de new_game recompilerait step à chaque partie (fuite mémoire)
NEW_GAME = jax.jit(core.new_game)


def play(db, bank, cmap, seed, step, legal, verbose=False, mixed=False, mode=None):
    global BANK_CODE
    BANK_CODE = bank.code
    rng = random.Random(seed)
    pool = db.normal_monsters()
    mode = mode or ("mixed" if mixed else "normal")
    # les effets rapides (volet 3) sont exclus des autres modes, pour que les graines anciennes rejouent les mêmes decks
    new = cards.J4 | cards.J4B  # cartes du jalon 4 : seulement dans les modes cont et dmg
    base = [c for c, effs in cards.CARDS.items() if effs[0][0] != core.K_QUICK and c not in new]
    if mode == "chain":
        fast = list(cards.ST_TYPES)
        decks = [[rng.choice(pool) for _ in range(22)] + [rng.choice(base) for _ in range(6)]
                 + [rng.choice(fast) for _ in range(12)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "trig":
        trig = [c for c, effs in cards.CARDS.items() if effs[0][0] in (core.K_TRIG_F, core.K_TRIG_O) and c not in new]
        other = [c for c in base if c not in trig]
        decks = [[rng.choice(pool) for _ in range(20)] + [rng.choice(trig) for _ in range(10)]
                 + [rng.choice(other) for _ in range(10)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "quick":
        qk = [c for c, effs in cards.CARDS.items() if effs[0][0] == core.K_QUICK and c not in new]
        other = [c for c in cards.CARDS if c not in qk and c not in new]
        decks = [[rng.choice(pool) for _ in range(20)] + [rng.choice(qk) for _ in range(9)]
                 + [rng.choice(other) for _ in range(11)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "cont":
        heroes = [c for c in pool if any(sc == cards.SET_ELEMENTAL_HERO for sc in cards.setcodes(db.rows[c][2]))]
        j4 = sorted(cards.J4)
        decks = [[rng.choice(pool) for _ in range(14)] + [rng.choice(heroes) for _ in range(5)]
                 + [cards.DARK_MAGICIAN] * 2 + [rng.choice(j4) for _ in range(13)] + [rng.choice(base) for _ in range(6)]
                 for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode in ("dmg", "ds"):
        j4b, j4 = sorted(cards.J4B), sorted(cards.J4)
        decks = [[rng.choice(pool) for _ in range(16)] + [rng.choice(j4b) for _ in range(14)]
                 + [rng.choice(j4) for _ in range(6)] + [rng.choice(base) for _ in range(4)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "mixed":
        decks = [[rng.choice(pool) for _ in range(28)] + [rng.choice(base) for _ in range(12)]
                 for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    else:
        decks = [[rng.choice(pool) for _ in range(D)] for _ in range(2)]
    duel = driver.Duel(db, decks[0], decks[1], seed=seed)
    s = NEW_GAME(bank, jnp.array([cmap[c] for c in decks[0]]), jnp.array([cmap[c] for c in decks[1]]))
    ndec = 0
    try:
        while True:
            dec = duel.next_decision()
            while dec is not None and len(filtered(dec)) == 1:
                if verbose and dec.msg != K["MSG_SELECT_CHAIN"]:
                    print("   auto", dec.msg, dec.options[filtered(dec)[0]])
                duel.respond(dec, filtered(dec)[0])
                dec = duel.next_decision()
            if dec is None:
                if not bool(s.done) or int(s.winner) != duel.winner:
                    raise Mismatch(f"fin : oracle winner={duel.winner}, jax done={bool(s.done)} winner={int(s.winner)}")
                return dict(ok=True, decisions=ndec, turns=duel.turn, winner=duel.winner)
            if bool(s.done):
                raise Mismatch(f"JAX fini (winner {int(s.winner)}) mais l'oracle demande {dec.msg} {dec.options}")
            so = duel.snapshot()
            shuffled = {ev[1] for ev in duel.events if ev[0] == "shuffle_hand"}
            duel.events.clear()
            if shuffled:
                COVER["shuffle_hand"] += 1
                s = resync_hand(s, so, shuffled)
            if int(s.ovf):
                raise Mismatch(f"débordement des tables du moteur (ovf={int(s.ovf)})")
            sj = jax_snapshot(bank, s)
            diff = [k for k in so if so[k] != sj[k]]
            if diff:
                raise Mismatch("état différent : " + "; ".join(f"{k}: oracle {so[k]} / jax {sj[k]}" for k in diff)
                               + f" | jax légales {np.where(np.asarray(legal(bank, s)))[0].tolist()}")
            idx = filtered(dec)
            seqs = {i: translate(dec, dec.options[i], s) for i in idx}
            m = np.asarray(legal(bank, s))
            want = {a for q in seqs.values() if q for a in (q if dec.options[0][0] == "cards" else q[:1])}
            got = set(np.where(m)[0].tolist())
            if want != got:
                raise Mismatch(f"actions légales : oracle {sorted(want)} / jax {sorted(got)} (msg {dec.msg}, "
                               f"{[dec.options[i] for i in idx]})")
            pick = idx
            if mode == "ds" and int(s.dstep) == core.DS_NONE:
                # ds : les cartes du volet 2 sont gardées pour le damage step
                keep = [i for i in idx if not (dec.options[i][0] in ("activate", "chain")
                                               and dec.options[i][1] in cards.J4B_SPELLS)]
                if keep and rng.random() < 0.9:
                    pick = keep
            i = rng.choice(pick)
            o = dec.options[i]
            COVER[(o[0], db.names.get(o[1], o[1]), o[3]) if o[0] in ("activate", "sset", "chain") else (dec.msg, o[0])] += 1
            if verbose:
                print(duel.turn, hex(duel.phase), dec.player, dec.options[i])
            duel.respond(dec, i)
            for a in seqs[i]:
                if not bool(legal(bank, s)[a]):
                    raise Mismatch(f"action {a} illégale côté JAX en cours de sélection {dec.options[i]}")
                s = step(bank, s, a)
            ndec += 1
    except Mismatch as e:
        return dict(ok=False, decisions=ndec, turn=duel.turn, err=str(e))
    finally:
        duel.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mode = sys.argv[3] if len(sys.argv) > 3 else "normal"
    db = driver.CardDB()
    bank, cmap = make_bank(db, db.normal_monsters() + list(cards.CARDS))  # une seule banque : pas de recompilation
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    nok = 0
    for sd in range(seed0, seed0 + n):
        r = play(db, bank, cmap, sd, step, legal, mode=mode)
        nok += r["ok"]
        print(sd, r, flush=True)
    print(f"{nok}/{n} parties identiques")
    for k, v in sorted(COVER.items(), key=str):
        print(" ", k, v)
