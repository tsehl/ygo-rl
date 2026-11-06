"""Comparaison moteur JAX ↔ oracle ygopro-core sur des parties aléatoires.

Decks : 40 monstres normaux, ou (mode « mixed ») 28 monstres normaux + 12 cartes à effet de `cards.CARDS`, ou
(mode « chain ») 22 monstres normaux + 18 cartes à effet dont beaucoup de pièges / jeu-rapide (longues chaînes), ou
(mode « trig ») 20 monstres normaux + 10 monstres à effet déclenché + 10 autres cartes à effet, ou (mode « cont »)
les cartes du jalon 4 (archétypes, effets continus, modificateurs temporaires) avec des E-HERO et des Dark Magician,
ou (mode « dmg ») les cartes du volet 2 (damage step, SET_ATTACK*) avec quelques cartes du volet 1 ; le mode « ds »
reprend ces decks mais garde le plus souvent les cartes du volet 2 pour les fenêtres du damage step, ou (mode « flag »)
les cartes du volet 3a (interdictions, protections) avec des cartes des autres volets et des anciens modes, ou (mode
« neg ») les cartes du volet 3b (annulation d'effets) avec des monstres à effet de tous les volets.

À chaque décision à plusieurs options de l'oracle : on compare l'état (zones, positions, ATK/DEF, LP, tour,
phase) et l'ensemble des actions légales, puis on joue la même option aléatoire dans les deux moteurs.
Les options « annuler » / « retirer » de l'oracle sont exclues (le moteur JAX n'en a pas).
Usage : uv run --with jax python compare.py [n_parties] [seed0] [normal|mixed|chain|trig|quick|cont|dmg|ds|flag|neg|fus|rit|syn|xyz|lnk|xlk|kn|pen]
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
GEN_CODE0 = 600_000_000  # cartes procédurales (gen/cardgen.py CODE0)


class Mismatch(Exception):
    pass


class Inconclusive(Mismatch):
    """Le pilote ne peut pas établir une correspondance indépendante."""


def make_bank(db, codes):
    codes = sorted(set(codes))
    r = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [x[4] for x in r], [x[5] for x in r], [x[6] & 0xFF for x in r],
                          lsc=[(x[6] >> 24) & 0xFF for x in r], rsc=[(x[6] >> 16) & 0xFF for x in r],
                          ctype=[x[3] for x in r], race=[x[7] for x in r], attr=[x[8] for x in r],
                          setcode=[cards.setcodes(x[2]) for x in r], **cards.assemble(codes))
    return bank, {c: i for i, c in enumerate(codes)}


def jax_snapshot(bank, s):
    s = jax.tree.map(np.asarray, s)
    code, atk, df = np.asarray(bank.code)[s.cid], s.atk_c, s.df_c
    out = dict(lp=tuple(int(x) for x in s.lp), turn=int(s.turn), turn_player=int(s.tp), phase=int(s.phase))

    def card(i):  # (code, pos, atk, def) + matériaux Xyz attachés s'il y en a (comme l'instantané de l'oracle)
        ov = np.where((s.zone == core.Z_OVERLAY) & (s.ov_par == i))[0]
        ov = tuple(int(code[j]) for j in ov[np.argsort(s.seq[ov])])
        return (int(code[i]), int(s.pos[i]), int(atk[i]), int(df[i])) + ((ov,) if ov else ())
    for p in (0, 1):
        for name, z in ZNAME.items():
            idx = np.where((s.zone == z) & (s.ctrl == p))[0]
            if z in (core.Z_MZONE, core.Z_SZONE):
                lst = [None] * (7 if z == core.Z_MZONE else 8)
                for i in idx:
                    lst[s.seq[i]] = card(i)
            else:
                lst = [card(i) for i in idx[np.argsort(s.seq[idx])]]
                if len(set(s.seq[idx])) != len(idx) or (len(idx) and s.seq[idx].max() != len(idx) - 1):
                    raise Mismatch(f"seq incohérents en {p},{name}: {sorted(s.seq[idx])}")
            out[(p, name)] = lst
    return out


def slot(s, con, loc, seq):
    if loc == "overlay":  # matériau Xyz : (case du Xyz, rang)
        par = slot(s, con, "mzone", seq[0])
        m = (np.asarray(s.zone) == core.Z_OVERLAY) & (np.asarray(s.ov_par) == par) & (np.asarray(s.seq) == seq[1])
        idx = np.where(m)[0]
        if len(idx) != 1:
            raise Mismatch(f"pas de matériau unique pour {con},{seq}")
        return int(idx[0])
    m = (np.asarray(s.zone) == ZNAME[loc]) & (np.asarray(s.ctrl) == con) & (np.asarray(s.seq) == seq)
    idx = np.where(m)[0]
    if len(idx) != 1:
        raise Mismatch(f"pas de slot unique pour {con},{loc},{seq}")
    return int(idx[0])


EFFECT_EVENTS = {}
ORACLE_EVENTS = {driver.K[name]: getattr(core, event) for name, event in {
    "EVENT_SUMMON_SUCCESS": "EV_SUMMON", "EVENT_FLIP_SUMMON_SUCCESS": "EV_FLIP_SUMMON",
    "EVENT_SPSUMMON_SUCCESS": "EV_SPSUMMON", "EVENT_ATTACK_ANNOUNCE": "EV_ATTACK",
    "EVENT_TO_GRAVE": "EV_TO_GRAVE", "EVENT_BATTLE_DESTROYED": "EV_BATTLE_DESTROYED",
    "EVENT_FLIP": "EV_FLIP", "EVENT_PRE_DAMAGE_CALCULATE": "EV_PRE_DMG"}.items()}
EFFECT_KINDS = {}  # types statiques des effets traduits (aucune légalité JAX consultée)
DESC = {}  # code → description (aux.Stringid) de chaque effet, cartes traduites (translate/validate.py)


def translate(dec, opt, s, m=None, option_index=None):
    """Option de l'oracle → suite d'actions JAX ; None si l'option est exclue."""
    kind = opt[0]
    N = s.cid.shape[0]
    sp = lambda k: NV * N + k
    if kind in ("summon", "mset", "repos", "attack", "sset", "activate", "chain"):
        verb = dict(summon=V_SUMMON, mset=V_MSET, repos=V_REPOS, attack=V_ATTACK, sset=V_SSET, activate=V_ACT0,
                    chain=V_ACT0)[kind]
        if kind in ("activate", "chain") and opt[1] >= GEN_CODE0 and opt[5] >> 20 == opt[1]:
            verb += opt[5] & 0xFFFFF  # carte procédurale : effet i décrit par aux.Stringid(id, i) (gen/procgen.lua)
        elif kind == "chain" and opt[1] >= GEN_CODE0 and opt[5] == 221:
            # déclencheur optionnel seul : oui / non (SelectEffectYesNo, description 221) : l'effet K_TRIG_O de la carte
            ek = np.asarray(BANK.ekind[int(s.cid[slot(s, opt[2], opt[3], opt[4])])])
            verb += int(np.argmax(ek == core.K_TRIG_O))
        elif kind in ("activate", "chain") and opt[1] in DESC and len(opt) > 5:  # carte traduite (translate/validate.py) : l'effet de cette description
            i = slot(s, opt[2], opt[3], opt[4])
            if kind == "chain" and opt[5] == 221:  # SelectEffectYesNo : déclencheur optionnel
                es = [e for e, k in enumerate(EFFECT_KINDS.get(opt[1], [])) if k == core.K_TRIG_O]
            else:
                ds = DESC[opt[1]]
                es = [e for e, d in enumerate(ds) if d == opt[5]] or [e for e, d in enumerate(ds) if d is None]
                kinds = EFFECT_KINDS.get(opt[1])
                if kinds is not None:
                    allowed = {core.K_ACTIVATE, core.K_QUICK}
                    allowed |= {core.K_IGNITION} if kind == "activate" else {core.K_TRIG_O, core.K_TRIG_F}
                    es = [e for e in es if kinds[e] in allowed]
            if len(es) > 1 and dec is not None and "effect_query" in dec.raw:
                if option_index is None:
                    indices = [j for j, value in enumerate(dec.options) if value == opt]
                    if len(indices) != 1:
                        raise Inconclusive("options identiques sans index oracle")
                    option_index = indices[0]
                raw_event = dec.raw["effect_query"](dec, option_index)
                event = ORACLE_EVENTS.get(raw_event)
                masks = EFFECT_EVENTS.get(opt[1], [])
                if event is not None and len(masks) == len(DESC[opt[1]]):
                    es = [e for e in es if masks[e] & (1 << event)]
            if len(es) != 1:
                raise Inconclusive(f"correspondance d'effet ambiguë : code={opt[1]} description={opt[5]} effets={es}")
            verb += es[0]
        return [verb * N + slot(s, opt[2], opt[3], opt[4])]
    if kind == "spsummon":  # invocation par procédure depuis l'Extra Deck, ou pendule (échelle gauche) : V_SUMMON
        if opt[3] == "hand":
            es = [e for e, k in enumerate(EFFECT_KINDS.get(opt[1], ())) if k == core.K_PROC]
            if len(es) != 1:
                raise Inconclusive(f"procédure de main non identifiée : {opt}")
            return [(V_ACT0 + es[0]) * N + slot(s, opt[2], opt[3], opt[4])]
        if opt[3] not in ("extra", "szone"):
            raise Mismatch(f"spsummon depuis une zone non prise en charge {opt}")
        return [V_SUMMON * N + slot(s, opt[2], opt[3], opt[4])]
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
        q = [V_SELECT * N + slot(s, c[1], c[2], c[3]) for c in opt[1]]
        if int(s.dec) == core.D_SELECT and bool(core.sel_var(s)) and len(q) < int(s.need):
            q.append(sp(core.A_PASS))  # sélection à taille variable terminée avant le maximum
        return q
    if kind == "pick":
        c = opt[1]
        return [V_SELECT * N + slot(s, c[1], c[2], c[3])]
    if kind in ("cancel", "unpick"):
        return None
    if kind == "finish":  # sélection des matériaux Link terminée (ailleurs : jouée d'office, voir filtered)
        return [sp(core.A_PASS)] if int(s.psel) in (core.PS_LINK, core.PS_PEND) else []
    if kind in ("pass", "no"):
        return [sp(core.A_PASS)]
    if kind == "yes":
        return [sp(core.A_YES)]
    raise Mismatch(f"option non gérée {opt}")


def decode(bank, s, acts):
    """Actions → (verbe, code de la carte) pour les messages d'écart."""
    n = N_(s)
    names = ["summon", "mset", "repos", "attack", "select", "sset"] + [f"act{e}" for e in range(core.E)]
    zn = {v: k for k, v in ZNAME.items()}
    zn[core.Z_OVERLAY] = "overlay"
    return [(names[a // n], int(bank.code[int(s.cid[a % n])]), int(s.ctrl[a % n]), zn.get(int(s.zone[a % n])),
             int(s.seq[a % n]), int(s.pos[a % n]), hex(int(s.st[a % n]))) if a < NV * n else ("spécial", a - NV * n)
            for a in sorted(acts)]


def N_(s):
    return s.cid.shape[0]


def filtered(dec, s=None):
    """Indices des options retenues (sans annuler / retirer ; « terminer » seul dès qu'il est possible, sauf pendant
    le choix des matériaux Link et des cartes d'une invocation pendule, où « terminer » et ajouter un matériau sont deux choix du joueur)."""
    idx = [i for i, o in enumerate(dec.options) if o[0] not in ("cancel", "unpick")]
    if s is not None and int(s.psel) in (core.PS_LINK, core.PS_PEND):
        return idx
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


def link_cover(bank, s, o):
    """Couverture du mode lnk : cases désignées par les flèches (les siennes, celles de l'adversaire), Link compté pour
    son lien, choix « terminer »."""
    lk = lambda i: (int(bank.ctype[int(s.cid[i])]) & core.TYPE_LINK) != 0
    if int(s.psel) == core.PS_LINK:
        sel = np.where((np.asarray(s.st) & core.ST_SEL) != 0)[0]
        if o[0] == "finish":
            COVER["lnk: terminer"] += 1
            if len(sel) < int(bank.level[int(s.cid[int(s.pend)])]):
                COVER["lnk: terminer, un Link compté pour son lien"] += 1
        if o[0] == "pick" and (int(bank.ctype[bank_index(bank, o[1][0])]) & core.TYPE_LINK):
            COVER["lnk: Link choisi comme matériau"] += 1


def bank_index(bank, code):
    return int(np.where(np.asarray(bank.code) == code)[0][0])


def link_arrivals(bank, before, s):
    """Couverture du mode lnk : monstres Link arrivés de l'Extra Deck depuis l'état before, et la case qu'ils ont prise
    (zone extra, case principale désignée par une flèche à soi / de l'adversaire)."""
    lk = (np.asarray(bank.ctype)[np.asarray(s.cid)] & core.TYPE_LINK) != 0
    COVER["lnk: Link réanimé"] += int((lk & (np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(before.zone) == core.Z_GRAVE)).sum())
    new = lk & (np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(before.zone) == core.Z_EXTRA)
    for c in np.where(new)[0]:
        q, p = int(s.seq[c]), int(s.ctrl[c])
        if q >= 5:
            COVER["lnk: invoqué en zone extra"] += 1
            other = (np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(s.ctrl) == p) & (np.asarray(s.seq) == 11 - q)
            COVER["lnk: Extra Link (l'autre zone extra est à soi)"] += bool(other.any())
            continue
        arw = np.asarray(core.link_arrows(bank, s, p))[:, q]
        mine = np.asarray(s.ctrl) == p
        COVER["lnk: invoqué en case principale, flèche à soi"] += bool(arw[mine].any())
        COVER["lnk: invoqué en case principale, flèche adverse"] += bool(arw[~mine].any())


EXTRA_LINK = jax.jit(lambda bank, s, p: core.extra_link(bank, s, p, jnp.zeros((1, 7), bool)))


def xlink_cover(bank, s):
    """Couverture du mode xlk : états où le joueur à la décision a une seule zone extra et l'autre libre ; où une
    chaîne de co-liens la relie à une case qui désigne l'autre zone extra (l'Extra Link n'attend plus qu'un Link de la
    bonne flèche)."""
    p = int(s.dec_player)
    me = (np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(s.ctrl) == p)
    sq = np.asarray(s.seq)
    if bool((me & (sq == 5)).any()) != bool((me & (sq == 6)).any()):
        COVER["xlk: une seule zone extra à soi"] += 1
        e, x = EXTRA_LINK(bank, s, p)
        COVER["xlk: chaîne de co-liens vers l'autre zone extra"] += bool(int(x[0]) != 0)
        return (int(e[0]), int(x[0])) if int(x[0]) != 0 else None
    return None


KN_DESTROY = {12580477, 53129443, 66788016, 74131780, 5318639, 4206964, 53582587, 56120475, 44095762, 54652250,
              59438930, 72426662}  # cartes qui détruisent des monstres (ou une carte ciblée)


def kn_cover(bank, s, dec, idx):
    """Couverture du mode kn : co-liens, protections, matériaux de même nom, activations de destruction face à un
    monstre indestructible par les effets."""
    zone, ctrl, cid = np.asarray(s.zone), np.asarray(s.ctrl), np.asarray(s.cid)
    cl, fl, ib = np.asarray(s.cl_c), np.asarray(s.fl_c), np.asarray(s.ib_c)
    COVER["kn: monstre co-lié sur le terrain"] += bool(cl.any())
    indes = (fl & core.FL_INDES_EFF) != 0
    COVER["kn: indestructible par les effets (Cerberus)"] += bool(indes.any())
    COVER["kn: indestructible au combat (Phoenix)"] += bool((cl & (ib == 0)).any())
    if dec.options[idx[0]][0] in ("activate", "chain") and indes.any():
        COVER["kn: destruction activable face à un indestructible"] += sum(
            dec.options[i][0] in ("activate", "chain") and dec.options[i][1] in KN_DESTROY for i in idx)
    if int(s.psel) == core.PS_LINK and int(bank.code[cid[int(s.pend)]]) in cards.J5E:
        p = int(s.dec_player)
        codes = np.asarray(bank.code)[cid[(zone == core.Z_MZONE) & (ctrl == p)]]
        COVER["kn: matériaux Knightmare, deux de même nom sur le terrain"] += len(set(codes.tolist())) < len(codes)


def pen_pick(bank, s, dec, idx, rng):
    """Biais du mode pen : activer les pendules, invoquer par pendule, choisir plusieurs cartes."""
    opts = dec.options
    pact = [i for i in idx if opts[i][0] in ("activate", "chain") and opts[i][1] in cards.J5F and opts[i][3] == "hand"]
    psum = [i for i in idx if opts[i][0] == "spsummon" and opts[i][3] == "szone"]
    if psum:
        COVER["pen: invocation pendule possible"] += 1
    # Magic Jammer en réponse à l'activation d'un pendule
    top = int(s.n_chain) - 1
    jam = [i for i in idx if opts[i][0] == "chain" and opts[i][1] == 77414722]
    if jam and top >= 0 and int(bank.code[int(s.cid[int(s.lk_src[top])])]) in cards.J5F:
        COVER["pen: Magic Jammer possible contre un pendule"] += 1
        if rng.random() < 0.9:
            return jam
    if psum and rng.random() < 0.9:
        return psum
    if pact and rng.random() < 0.7:
        return pact
    if int(s.psel) == core.PS_PEND:
        picks = [i for i in idx if opts[i][0] == "pick"]
        if picks and rng.random() < 0.75:
            return picks
    return None


def pen_cover(bank, before, s, o):
    """Couverture du mode pen : pendules arrivés dans l'Extra Deck (depuis la zone monstres / pendule), invoqués par
    pendule (depuis la main / l'Extra Deck), activations de pendule niées, invocation pendule terminée."""
    z0, z1 = np.asarray(before.zone), np.asarray(s.zone)
    pen = (np.asarray(bank.ctype)[np.asarray(s.cid)] & core.TYPE_PENDULUM) != 0
    for zf, name in ((core.Z_MZONE, "zone monstres"), (core.Z_SZONE, "zone pendule")):
        COVER[f"pen: pendule vers l'Extra Deck depuis la {name}"] += int((pen & (z0 == zf) & (z1 == core.Z_EXTRA)).sum())
    COVER["pen: pendule au cimetière depuis la zone pendule (activation niée)"] += int(
        (pen & (z0 == core.Z_SZONE) & (z1 == core.Z_GRAVE)).sum())
    COVER["pen: pendule au cimetière depuis le terrain (matériau Xyz)"] += int(
        (pen & (z0 == core.Z_OVERLAY) & (z1 == core.Z_GRAVE)).sum())
    ps = (np.asarray(s.st) & core.ST_SPSUM) != 0
    for zf, name in ((core.Z_HAND, "main"), (core.Z_EXTRA, "Extra Deck")):
        COVER[f"pen: invoqué par pendule depuis {name}"] += int(((z0 == zf) & (z1 == core.Z_MZONE) & ps & (
            int(before.psel) == core.PS_PEND or int(before.zctx) == core.ZC_PEND)).sum())
    if bool(s.ps_used) and not bool(before.ps_used):
        st, pz = np.asarray(s.st), np.asarray(s.pzone)
        n = int(((z1 == core.Z_MZONE) & (np.asarray(s.ctrl) == int(s.tp)) & ((st & core.ST_SUMMON_TURN) != 0)
                 & ((st & core.ST_SPSUM) != 0) & ((pz == core.Z_HAND) | ((pz == core.Z_EXTRA) & pen))).sum())
        COVER[f"pen: invocation pendule de {n} carte(s)"] += 1
    if int(before.zctx) == core.ZC_PEND and int(before.dec) == core.D_ZONE and o[0] == "place":
        COVER[f"pen: case choisie {'zone extra' if o[3] >= 5 else 'principale'}"] += 1
    fdx = (z1 == core.Z_EXTRA) & (np.asarray(s.pos) == core.P_FD_DEF) & (z0 != core.Z_EXTRA)
    if fdx.any() and ((z1 == core.Z_EXTRA) & (np.asarray(s.pos) == core.P_FU_DEF)).any():
        COVER["pen: monstre verso rendu à un Extra Deck qui a des pendules recto"] += 1


BANK_CODE = None
LAST_LOG = []
LAST_STATE = None  # dernier état JAX comparé (diagnostic)
COVER = collections.Counter()  # options jouées (activations par carte, poses, mélanges…)
NEW_GAME = jax.jit(core.new_game)  # sans step : la première décision forcée est jouée par le step de play()
FORCED = jax.jit(core.forced)


def play(db, bank, cmap, seed, step, legal, verbose=False, mixed=False, mode=None, given=None, chooser=None):
    """given : (decks, extra) imposés (cartes procédurales, gen/compare_gen.py) ; sinon tirés selon le mode."""
    global BANK_CODE, BANK
    BANK_CODE, BANK = bank.code, bank
    rng = random.Random(seed)
    pool = db.normal_monsters()
    mode = mode or ("mixed" if mixed else "normal")
    # les effets rapides (volet 3) sont exclus des autres modes, pour que les graines anciennes rejouent les mêmes decks
    new = cards.J4 | cards.J4B | cards.J4C | cards.J4D | cards.J5A | cards.J5B | cards.J5C | cards.J5D | cards.J5E | cards.J5F  # cartes des jalons 4-5 : seulement dans leurs modes
    extra = [[], []]
    base = [c for c, effs in cards.CARDS.items() if effs[0][0] != core.K_QUICK and c not in new]
    if given is not None:
        decks, extra = given
    elif mode == "chain":
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
    elif mode == "fus":
        mats = [34460851, 44287299, 6368038, 28279543, 89631139]
        decks = [[rng.choice(pool) for _ in range(20)] + [rng.choice(mats) for _ in range(12)] + [24094653] * 3
                 + [83764718] * 2 + [rng.choice(base) for _ in range(3)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[45231177, 45231177, 66889139, 66889139, 23995346, 23995346] for _ in range(2)]
    elif mode == "rit":
        rit = [55761792] * 3 + [5405694] * 3 + [8198712] * 3 + [72426662] * 3
        decks = [[rng.choice(pool) for _ in range(22)] + rit + [83764718] * 2 + [rng.choice(base) for _ in range(4)]
                 for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "syn":
        tuners = sorted(cards.J5C)
        low = [c for c in pool if (db.rows[c][6] & 0xFF) <= 4]
        decks = [[rng.choice(low) for _ in range(20)] + [rng.choice(tuners) for _ in range(12)] + [83764718] * 2
                 + [rng.choice(base) for _ in range(6)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[97204936, 97204936, 45815891, 45815891, 16527176, 16527176] for _ in range(2)]
    elif mode == "xyz":
        l4 = [c for c in pool if (db.rows[c][6] & 0xFF) == 4]
        l3 = [c for c in pool if (db.rows[c][6] & 0xFF) == 3]
        decks = [[rng.choice(l4) for _ in range(16)] + [rng.choice(l3) for _ in range(10)] + [rng.choice(pool) for _ in range(4)]
                 + [83764718] * 2 + [53129443] * 2 + [rng.choice(base) for _ in range(6)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[71594310, 71594310, 9053187, 9053187, 69610924, 69610924] for _ in range(2)]
    elif mode == "lnk":
        low = [c for c in pool if (db.rows[c][6] & 0xFF) <= 4]
        l4 = [c for c in low if (db.rows[c][6] & 0xFF) == 4]
        decks = [[rng.choice(low) for _ in range(20)] + [rng.choice(l4) for _ in range(8)] + [83764718] * 2
                 + [53129443] * 2 + [rng.choice(base) for _ in range(8)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[77637979, 77637979, 67598234, 67598234, 41248270, 41248270, 5402805, 5402805, 71594310, 71594310]
                 for _ in range(2)]
    elif mode == "xlk":
        # Extra Link : decks du mode lnk ; Extra Deck : les trois Link fictifs (driver.TEST_CARDS) et des Link dont les
        # flèches du bas désignent les cases 0-4 depuis une zone extra
        low = [c for c in pool if (db.rows[c][6] & 0xFF) <= 4]
        l4 = [c for c in low if (db.rows[c][6] & 0xFF) == 4]
        decks = [[rng.choice(low) for _ in range(22)] + [rng.choice(l4) for _ in range(8)] + [83764718] * 4
                 + [rng.choice(base) for _ in range(6)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[990000001] * 3 + [990000002] * 2 + [990000003] * 2 + [41248270] * 2 + [77637979] * 3
                 + [67598234] * 2 for _ in range(2)]
    elif mode == "kn":
        # Knightmare : peu de noms différents parmi les monstres (matériaux de même nom refusés), des magies / pièges
        # (cibles de Phoenix, destructions face à Cerberus) ; Extra Deck : Link dont les flèches du bas désignent les
        # cases principales (pour y poser les Knightmare et former des co-liens)
        low = [c for c in pool if (db.rows[c][6] & 0xFF) <= 4]
        few = rng.sample(low, 6)
        decks = [[rng.choice(few) for _ in range(20)] + [rng.choice(low) for _ in range(6)] + [83764718] * 3
                 + [53129443, 12580477] + [rng.choice(base) for _ in range(9)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[2857636] * 3 + [75452921] * 3 + [67598234] * 2 + [41248270] * 2 + [77637979] * 2 for _ in range(2)]
    elif mode == "pen":
        # pendule : monstres de tous niveaux (entre les échelles), pendules, Monster Reborn, destructions (pendules du
        # terrain vers l'Extra Deck), Magic Jammer (activation d'un pendule niée) ; Extra Deck : Link (cases désignées
        # par les flèches pour les pendules de l'Extra Deck) et Xyz (pendules en matériau : au cimetière)
        pens = sorted(cards.J5F)
        decks = [[rng.choice(pool) for _ in range(16)] + [rng.choice(pens) for _ in range(12)] + [83764718] * 2
                 + [5318639] * 2 + [77414722] * 2 + [53129443, 12580477] + [rng.choice(base) for _ in range(4)]
                 for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
        extra = [[77637979] * 3 + [67598234] * 2 + [41248270] * 2 + [71594310] * 2 for _ in range(2)]
    elif mode == "neg":
        j4d = sorted(cards.J4D)
        mons = sorted(set(cards.J4_MONSTERS + cards.J4B_MONSTERS + cards.J4C_MONSTERS)
                      | {c for c, effs in cards.CARDS.items() if effs[0][0] in (core.K_TRIG_F, core.K_TRIG_O, core.K_IGNITION)
                    and c not in cards.J5B | cards.J5D | cards.J5E})
        decks = [[rng.choice(pool) for _ in range(14)] + [rng.choice(j4d) for _ in range(10)]
                 + [rng.choice(mons) for _ in range(12)] + [rng.choice(base) for _ in range(4)] for _ in range(2)]
        for d in decks:
            rng.shuffle(d)
    elif mode == "flag":
        j4c, j4 = sorted(cards.J4C), sorted(cards.J4 | cards.J4B)
        decks = [[rng.choice(pool) for _ in range(14)] + [rng.choice(j4c) for _ in range(14)]
                 + [rng.choice(j4) for _ in range(4)] + [rng.choice(base) for _ in range(8)] for _ in range(2)]
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
    duel = driver.Duel(db, decks[0], decks[1], seed=seed, extra0=extra[0], extra1=extra[1])
    global LAST_LOG
    LAST_LOG = duel.log  # messages de l'oracle (erreurs Lua des scripts)
    ndec = 0
    trace = []
    def coverage():
        return [dict(code=c, stage=stage, description=desc, count=n)
                for (c, stage, desc), n in sorted(duel.coverage.items(), key=str)]
    try:
        ex = [jnp.array([cmap[c] for c in e], jnp.int32) if e else None for e in extra]
        s = NEW_GAME(bank, jnp.array([cmap[c] for c in decks[0]]), jnp.array([cmap[c] for c in decks[1]]), None, *ex)
        a0 = int(FORCED(bank, s))
        if a0 >= 0:
            s = step(bank, s, a0)
        while True:
            dec = duel.next_decision()
            while dec is not None and len(filtered(dec, s)) == 1:
                if verbose and dec.msg != K["MSG_SELECT_CHAIN"]:
                    print("   auto", dec.msg, dec.options[filtered(dec, s)[0]])
                duel.respond(dec, filtered(dec, s)[0])
                dec = duel.next_decision()
            if dec is None:
                if not bool(s.done) or int(s.winner) != duel.winner:
                    raise Mismatch(f"fin : oracle winner={duel.winner}, jax done={bool(s.done)} winner={int(s.winner)}")
                return dict(ok=True, decisions=ndec, turns=duel.turn, winner=duel.winner, status="passed", coverage=coverage(), trace=trace, oracle_log=list(duel.log))
            if bool(s.done):
                raise Mismatch(f"JAX fini (winner {int(s.winner)}) mais l'oracle demande {dec.msg} {dec.options}")
            global LAST_STATE
            LAST_STATE = s
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
                               + f" | jax légales {decode(bank, s, np.where(np.asarray(legal(bank, s)))[0].tolist())}")
            idx = filtered(dec, s)
            ready = mode == "xlk" and xlink_cover(bank, s)
            m = np.asarray(legal(bank, s))
            seqs = {i: translate(dec, dec.options[i], s, m, option_index=i) for i in idx}
            # sélection d'un ensemble : union des cartes ; A_PASS seulement pour l'ensemble vide (taille variable)
            want = {a for q in seqs.values() if q for a in (q if dec.options[0][0] == "cards" else q[:1])
                    if not (dec.options[0][0] == "cards" and len(q) > 1 and a == q[-1] and a >= NV * N_(s))}
            got = set(np.where(m)[0].tolist())
            if want != got:
                raise Mismatch(f"actions légales : oracle {sorted(want)} / jax {sorted(got)} (msg {dec.msg}, "
                               f"{[dec.options[i] for i in idx]}) ; écart {decode(bank, s, want ^ got)}")
            pick = idx
            if mode == "ds" and int(s.dstep) == core.DS_NONE:
                # ds : les cartes du volet 2 sont gardées pour le damage step
                keep = [i for i in idx if not (dec.options[i][0] in ("activate", "chain")
                                               and dec.options[i][1] in cards.J4B_SPELLS)]
                if keep and rng.random() < 0.9:
                    pick = keep
            if mode == "kn":
                kn_cover(bank, s, dec, idx)
            if mode == "pen":
                pick = pen_pick(bank, s, dec, idx, rng) or pick
            if mode in ("lnk", "xlk", "kn"):
                # lnk : invocations par procédure plus fréquentes, et un Link gardé sur le terrain plus souvent (pour
                # que le suivant se place selon ses flèches)
                sps = [i for i in idx if dec.options[i][0] == "spsummon"]
                sus = [i for i in idx if dec.options[i][0] in ("summon", "mset")]
                if sps and rng.random() < 0.5:
                    pick = sps
                elif sus and rng.random() < 0.6:
                    pick = sus
                if int(s.psel) == core.PS_LINK and rng.random() < (0.85 if mode == "xlk" else 0.6 if mode == "lnk" else 0.8):
                    keep = [i for i in idx if not (dec.options[i][0] == "pick" and int(
                        bank.ctype[bank_index(bank, dec.options[i][1][0])]) & core.TYPE_LINK)]
                    pick = keep or pick
            if mode == "kn":
                # kn : déclencheurs des Knightmare activés ; avec un Link à soi en zone extra, un Knightmare invoqué
                # dans une case principale (sous sa flèche du bas : co-lien par la flèche du haut du Knightmare)
                o0 = dec.options[idx[0]][0]
                kact = [i for i in idx if dec.options[i][0] in ("chain", "activate") and dec.options[i][1] in cards.J5E]
                ksp = [i for i in idx if dec.options[i][0] == "spsummon" and dec.options[i][1] in cards.J5E]
                ct = np.asarray(bank.ctype)[np.asarray(s.cid)]
                emz = bool(((np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(s.ctrl) == int(s.dec_player))
                            & (np.asarray(s.seq) >= 5) & ((ct & core.TYPE_LINK) != 0)).any())
                if kact and rng.random() < 0.9:
                    pick = kact
                elif ksp and emz and rng.random() < 0.8:
                    pick = ksp
                elif o0 == "place" and int(s.zctx) == core.ZC_PROC and int(bank.code[int(s.cid[int(s.pend)])]) in cards.J5E:
                    main = [i for i in idx if dec.options[i][3] < 5]
                    if main and rng.random() < 0.9:
                        pick = main
            if ready:
                # xlk : chaîne de co-liens prête : invoquer par procédure, et prendre l'autre zone extra
                e, x = ready
                sps = [i for i in idx if dec.options[i][0] == "spsummon" and int(core.arrows_at(
                    bank.lmark[bank_index(bank, dec.options[i][1])], jnp.int32(e))) & x]
                emz = [i for i in idx if dec.options[i][0] == "place" and dec.options[i][2] == "mzone"
                       and dec.options[i][3] >= 5]
                if sps:
                    COVER["xlk: chaîne prête, un Link de la bonne flèche invocable"] += 1
                COVER["xlk: chaîne prête par une carte adverse"] += bool(x & 0xF80)
                # matériaux hors chaîne : monstres recto non-Link du joueur (sinon la sélection défait la chaîne)
                ct = np.asarray(bank.ctype)[np.asarray(s.cid)]
                nl = int(((np.asarray(s.zone) == core.Z_MZONE) & (np.asarray(s.ctrl) == int(s.dec_player))
                          & ((np.asarray(s.pos) & core.P_FACEUP) != 0) & ((ct & core.TYPE_LINK) == 0)).sum())
                more = [i for i in idx if dec.options[i][0] in ("summon", "repos")
                        or (dec.options[i][0] == "activate" and dec.options[i][1] == 83764718)]
                nolk = [i for i in idx if dec.options[i][0] == "pick" and not (int(
                    bank.ctype[bank_index(bank, dec.options[i][1][0])]) & core.TYPE_LINK)]
                if sps and nl >= 2 and rng.random() < 0.9:
                    pick = sps
                elif more and nl < 2 and rng.random() < 0.9:
                    pick = more
                elif nl < 2 and rng.random() < 0.9 and [i for i in idx if dec.options[i][0] not in ("spsummon", "attack")]:
                    pick = [i for i in idx if dec.options[i][0] not in ("spsummon", "attack")]  # garder la chaîne
                elif nolk:
                    pick = nolk
                elif emz and rng.random() < 0.9:
                    pick = emz
            elif mode == "xlk" and dec.options[idx[0]][0] == "place" and int(s.dec) == core.D_ZONE:
                # xlk : la case 2 (désignée par les deux zones extra) gardée pour les Link, qui y vont de préférence
                z2 = [i for i in pick if dec.options[i][2] == "mzone" and dec.options[i][3] == 2]
                no2 = [i for i in pick if i not in z2]
                if int(s.zctx) == core.ZC_SUMMON and no2 and rng.random() < 0.9:
                    pick = no2
                elif int(s.zctx) != core.ZC_SUMMON and z2 and rng.random() < 0.8:
                    pick = z2
            i = rng.choice(pick) if chooser is None else chooser(dec, pick)
            if i not in pick:
                raise ValueError("politique de test : option hors des choix autorisés")
            o = dec.options[i]
            if mode in ("lnk", "xlk", "kn"):
                link_cover(bank, s, o)
            if mode == "kn" and o[0] in ("activate", "chain") and o[1] in KN_DESTROY:
                COVER["kn: destruction jouée face à un indestructible"] += bool(
                    ((np.asarray(s.fl_c) & core.FL_INDES_EFF) != 0).any())
            COVER[(o[0], db.names.get(o[1], o[1]), o[3]) if o[0] in ("activate", "sset", "chain", "spsummon") else (dec.msg, o[0])] += 1
            if int(s.dec) == core.D_YESNO and int(s.zctx) == core.ZC_REPLAY:
                COVER[f"relance d'attaque : {o[0]}"] += 1
            if int(s.dec) == core.D_TARGET and bool(s.rpl):
                COVER["relance d'attaque : nouvelle cible choisie"] += 1
            if verbose:
                print(duel.turn, hex(duel.phase), dec.player, dec.options[i])
            trace.append(dict(message=dec.msg, player=dec.player, option=o, actions=seqs[i], oracle_event=dec.raw.get("effect_events", {}).get(i)))
            duel.respond(dec, i)
            before = s
            for a in seqs[i]:
                if dec.msg == K["MSG_SELECT_SUM"] and (int(s.rsel) == 0 or int(s.st[a % N_(s)]) & core.ST_SEL):
                    continue  # sélection par somme : cartes déjà choisies d'office (forced_m), ou sélection terminée
                if not bool(legal(bank, s)[a]):
                    raise Mismatch(f"action {a} illégale côté JAX en cours de sélection {dec.options[i]}")
                s = step(bank, s, a)
            if mode in ("lnk", "xlk", "kn"):
                link_arrivals(bank, before, s)
            if mode == "pen":
                pen_cover(bank, before, s, o)
            ndec += 1
    except Mismatch as e:
        return dict(ok=False, decisions=ndec, turn=duel.turn, err=str(e), status="inconclusive" if isinstance(e, Inconclusive) else "failed", coverage=coverage(), trace=trace, oracle_log=list(duel.log))
    finally:
        duel.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    if n <= 0:
        raise SystemExit("n_parties doit être > 0")
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mode = sys.argv[3] if len(sys.argv) > 3 else "normal"
    db = driver.CardDB()
    bank, cmap = make_bank(db, db.normal_monsters() + list(cards.CARDS) + list(cards.FUSION) + list(cards.RITUAL)
                         + list(cards.SYNCHRO) + list(cards.TUNER_STATS) + list(cards.XYZ) + list(cards.LINK))  # une seule banque
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    nok = 0
    for sd in range(seed0, seed0 + n):
        r = play(db, bank, cmap, sd, step, legal, mode=mode)
        nok += r["ok"]
        print(sd, {k: v for k, v in r.items() if k not in ("coverage", "trace", "oracle_log")}, flush=True)
    print(f"{nok}/{n} parties identiques")
    for k, v in sorted(COVER.items(), key=str):
        print(" ", k, v)

    return 0 if nok == n else 1


if __name__ == "__main__":
    sys.exit(main())
