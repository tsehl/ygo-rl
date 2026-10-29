"""Vraies cartes traduites en briques (jalons 2-3) et assembleur vers les tableaux de la banque.

Un effet = genre + programmes coût / cible / résolution. Les instructions sont des tuples (op, a, b, c) ;
dans FILTER, b est un filtre écrit en ligne (F(...)), que l'assembleur range dans la table des filtres.
Joueurs relatifs : ME = celui qui active, OPP = son adversaire.
"""
import numpy as np

from core import (BIG, E, EV_ATTACK, EV_BATTLE_DESTROYED, EV_FLIP, EV_FLIP_SUMMON, EV_SPSUMMON, EV_SUMMON, EV_TO_GRAVE,
                  FW, K_TRIG_F, K_TRIG_O, OP_COND_EVP, OP_COND_SELF, OP_COND_TURN, OP_EVENT, OP_LOCK, K_ACTIVATE,
                  K_IGNITION, NF, OP_CHAINCARD, OP_CHOOSE, OP_COND_CHAIN, OP_DAMAGE, OP_DRAW,
                  OP_FILTER, OP_MINATK, OP_MOVE, OP_NEGATE, OP_PAY_LP, OP_RECOVER, OP_REQ, OP_REQ_FREE, OP_SELF,
                  OP_SHUFFLE_HAND, OP_SPSUMMON, P, P_FACEUP, P_FU_ATK, P_FU_DEF, TYPE_COUNTER, TYPE_MONSTER,
                  P_FD_ATK, TYPE_QUICKPLAY, TYPE_SPELL, TYPE_TRAP, Z_DECK, Z_GRAVE, Z_HAND, Z_MZONE, Z_SZONE,
                  K_QUICK, Z_REMOVED)

ME, OPP = 0, 1


def F(me=0, op=0, pos=0, typ=0, not_self=0, atk=(-BIG, BIG), df=(-BIG, BIG), attr=0):
    return (me, op, pos, typ, not_self, atk[0], atk[1], df[0], df[1], attr)


def FILTER(r, f, within=None): return (OP_FILTER, r, f, 0 if within is None else within + 1)  # ∩ registre within
def SELF(r): return (OP_SELF, r, 0, 0)
def REQ(r, n): return (OP_REQ, r, n, 0)
def REQ_FREE(who, zone, n): return (OP_REQ_FREE, who, zone, n)
def MINATK(dst, src): return (OP_MINATK, dst, src, 0)
def CHOOSE(dst, src, n, who=ME, upto=False):  # exactement n cartes (upto : au plus n, éventuellement aucune)
    return (OP_CHOOSE, dst, src, (0 if upto else n) | (n << 4) | (who << 8))
def MOVE(r, zone=Z_GRAVE): return (OP_MOVE, r, zone, 0)
def SPSUMMON(r, pos): return (OP_SPSUMMON, r, pos, 0)
def DRAW(who, n): return (OP_DRAW, who, n, 0)
def DAMAGE(who, n, per=None): return (OP_DAMAGE, who, n, 0 if per is None else per + 1)  # n × |registre per|
def RECOVER(who, n): return (OP_RECOVER, who, n, 0)
def SHUFFLE_HAND(who): return (OP_SHUFFLE_HAND, who, 0, 0)
def COND_CHAIN(typ): return (OP_COND_CHAIN, typ, 0, 0)  # le maillon précédent est une activation de carte typ
def COND_CHAIN_FIELD(): return (OP_COND_CHAIN, 0, 1, 0)  # … est un effet (pas une activation de carte) d'une carte sur le terrain
def CHAINCARD(r, zone=0): return (OP_CHAINCARD, r, zone, 0)  # sa carte, si elle est toujours dans zone (0 : magie/piège)
def NEGATE(): return (OP_NEGATE, 0, 0, 0)               # nie son activation
def PAY_LP(n): return (OP_PAY_LP, n, 0, 0)
def EVENT(r, ev): return (OP_EVENT, r, ev, 0)            # cartes concernées par les événements ev (masque)
def COND_TURN(who): return (OP_COND_TURN, who, 0, 0)     # c'est le tour de who
def COND_EVP(ev, who): return (OP_COND_EVP, ev, who, 0)  # l'un des événements ev vient de who
def COND_SELF(zone=0, prev=0): return (OP_COND_SELF, zone, prev, 0)  # la carte est dans zone / venait de prev
def LOCK(r): return (OP_LOCK, r, 0, 0)  # ME ne peut plus activer les effets des cartes de ce nom ce tour


def ev(*codes):
    return sum(1 << c for c in codes)


# effet = (genre, on, opt, rng, coût, cible, résolution)
def act(cost=(), target=(), resolve=(), on=0):
    """on : masque d'événements (ev(...)) ; 0 = activation libre."""
    return (K_ACTIVATE, on, 0, 0, cost, target, resolve)


def ignition(cost=(), target=(), resolve=()):
    return (K_IGNITION, 0, 0, 0, cost, target, resolve)


def trigger(on, cost=(), target=(), resolve=(), optional=False, opt=False):
    """Effet déclenché (SINGLE) par les événements `on` qui concernent la carte ; opt : une fois par tour (nom)."""
    return (K_TRIG_O if optional else K_TRIG_F, on, int(opt), 0, cost, target, resolve)


def quick(rng, cost=(), target=(), resolve=(), opt=False):
    """Effet rapide de monstre (vitesse 2) activable depuis les zones rng (main, zone monstres recto)."""
    return (K_QUICK, 0, int(opt), rng, cost, target, resolve)


MY_DECK = F(me=Z_DECK)
OPP_MON = F(op=Z_MZONE)
ALL_MON = F(me=Z_MZONE, op=Z_MZONE)
OPP_FACEUP_MON = F(op=Z_MZONE, pos=P_FACEUP)
OPP_ST = F(op=Z_MZONE | Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP, not_self=1)
GRAVE_MON = F(me=Z_GRAVE, op=Z_GRAVE, typ=TYPE_MONSTER)
BIG_FACEUP = F(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP, atk=(1000, BIG))
OPP_ATK_MON = F(op=Z_MZONE, pos=P_FU_ATK | P_FD_ATK)
ONFIELD = Z_MZONE | Z_SZONE
DARK = 0x20
SEARCH_ATK1500 = F(me=Z_DECK, typ=TYPE_MONSTER, atk=(-BIG, 1500))
SEARCH_DEF1500 = F(me=Z_DECK, typ=TYPE_MONSTER, df=(-BIG, 1500))
DECK_DARK1500 = F(me=Z_DECK, typ=TYPE_MONSTER, atk=(-BIG, 1500), attr=DARK)
FIELD_ST = F(me=Z_MZONE | Z_SZONE, op=Z_MZONE | Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP, not_self=1)

CARDS = {
    55144522: [act(target=[FILTER(0, MY_DECK), REQ(0, 2)], resolve=[DRAW(ME, 2)])],            # Pot of Greed
    12580477: [act(target=[FILTER(0, OPP_MON), REQ(0, 1)], resolve=[FILTER(0, OPP_MON), MOVE(0)])],  # Raigeki
    53129443: [act(target=[FILTER(0, ALL_MON), REQ(0, 1)], resolve=[FILTER(0, ALL_MON), MOVE(0)])],  # Dark Hole
    46130346: [act(resolve=[DAMAGE(OPP, 500)])],                                                # Hinotama
    84257639: [act(resolve=[RECOVER(ME, 1000)])],                                               # Dian Keto
    70368879: [act(target=[FILTER(0, MY_DECK), REQ(0, 1)], resolve=[DRAW(ME, 1), RECOVER(OPP, 1000)])],  # Upstart
    66788016: [act(target=[FILTER(0, OPP_FACEUP_MON), REQ(0, 1)],                               # Fissure
                   resolve=[FILTER(0, OPP_FACEUP_MON), REQ(0, 1), MINATK(1, 0), CHOOSE(2, 1, 1), MOVE(2)])],
    83764718: [act(target=[REQ_FREE(ME, Z_MZONE, 1), FILTER(0, GRAVE_MON), CHOOSE(1, 0, 1)],    # Monster Reborn
                   resolve=[SPSUMMON(1, P_FU_ATK | P_FU_DEF)])],
    79571449: [act(target=[FILTER(0, MY_DECK), REQ(0, 3)],                                      # Graceful Charity
                   resolve=[DRAW(ME, 3), SHUFFLE_HAND(ME), FILTER(0, F(me=Z_HAND)), CHOOSE(1, 0, 2), MOVE(1)])],
    18144506: [act(target=[FILTER(0, OPP_ST), REQ(0, 1)], resolve=[FILTER(0, OPP_ST), MOVE(0)])],  # Feather Duster
    74131780: [ignition(cost=[SELF(0), MOVE(0)],                                                # Exiled Force
                        target=[FILTER(1, F(me=Z_MZONE, op=Z_MZONE, not_self=1)), CHOOSE(2, 1, 1)],
                        resolve=[MOVE(2)])],
    11384280: [ignition(cost=[FILTER(0, F(me=Z_MZONE)), CHOOSE(1, 0, 1), MOVE(1)],              # Cannon Soldier
                        resolve=[DAMAGE(OPP, 500)])],
    # jalon 3a : chaînes
    83968380: [act(target=[FILTER(0, MY_DECK), REQ(0, 1)], resolve=[DRAW(ME, 1)])],               # Jar of Greed
    24068492: [act(target=[FILTER(0, OPP_MON), REQ(0, 1)],                                       # Just Desserts
                   resolve=[FILTER(0, OPP_MON), DAMAGE(OPP, 500, per=0)])],
    5318639: [act(target=[FILTER(0, FIELD_ST), CHOOSE(1, 0, 1)], resolve=[MOVE(1)])],            # Mystical Space Typhoon
    77414722: [act(cost=[COND_CHAIN(TYPE_SPELL), FILTER(0, F(me=Z_HAND)), CHOOSE(1, 0, 1), MOVE(1)],  # Magic Jammer
                   resolve=[NEGATE(), CHAINCARD(0), MOVE(0)])],
    3819470: [act(cost=[COND_CHAIN(TYPE_TRAP), PAY_LP(1000)], resolve=[NEGATE(), CHAINCARD(0), MOVE(0)])],  # Seven Tools
    # jalon 3b : pièges sur événement
    4206964: [act(on=ev(EV_SUMMON, EV_FLIP_SUMMON),                                              # Trap Hole
                  target=[COND_EVP(ev(EV_SUMMON, EV_FLIP_SUMMON), OPP), EVENT(0, ev(EV_SUMMON, EV_FLIP_SUMMON)),
                          FILTER(1, BIG_FACEUP, within=0), REQ(1, 1)],
                  resolve=[FILTER(1, BIG_FACEUP, within=1), MOVE(1)])],
    53582587: [act(on=ev(EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON),                                # Torrential Tribute
                   target=[FILTER(0, ALL_MON), REQ(0, 1)], resolve=[FILTER(0, ALL_MON), MOVE(0)])],
    56120475: [act(on=ev(EV_ATTACK), target=[COND_TURN(OPP), EVENT(0, ev(EV_ATTACK)),            # Sakuretsu Armor
                                             FILTER(0, ALL_MON, within=0), REQ(0, 1)], resolve=[MOVE(0)])],
    44095762: [act(on=ev(EV_ATTACK), target=[COND_TURN(OPP), FILTER(0, OPP_ATK_MON), REQ(0, 1)],  # Mirror Force
                   resolve=[FILTER(0, OPP_ATK_MON), MOVE(0)])],
    26202165: [trigger(ev(EV_TO_GRAVE), opt=True, target=[COND_SELF(prev=ONFIELD)],                       # Sangan
                       resolve=[FILTER(0, SEARCH_ATK1500), CHOOSE(1, 0, 1, upto=True), MOVE(1, Z_HAND), LOCK(1)])],
    78010363: [trigger(ev(EV_TO_GRAVE), opt=True, target=[COND_SELF(prev=ONFIELD)],         # Witch of the Black Forest
                       resolve=[FILTER(0, SEARCH_DEF1500), CHOOSE(1, 0, 1, upto=True), MOVE(1, Z_HAND), LOCK(1)])],
    54652250: [trigger(ev(EV_FLIP), target=[FILTER(0, ALL_MON), CHOOSE(1, 0, 1, upto=True)],              # Man-Eater Bug
                       resolve=[MOVE(1)])],
    83011277: [trigger(ev(EV_BATTLE_DESTROYED), optional=True,                                               # Mystic Tomato
                       target=[COND_SELF(zone=Z_GRAVE), REQ_FREE(ME, Z_MZONE, 1), FILTER(0, DECK_DARK1500), REQ(0, 1)],
                       resolve=[REQ_FREE(ME, Z_MZONE, 1), FILTER(0, DECK_DARK1500), CHOOSE(1, 0, 1, upto=True),
                                SPSUMMON(1, P_FU_ATK)])],
    # jalon 3b, volet 3 : effets rapides de monstre
    24508238: [quick(Z_HAND, cost=[SELF(0), MOVE(0)],                                                       # D.D. Crow
                     target=[FILTER(1, F(op=Z_GRAVE)), CHOOSE(2, 1, 1)], resolve=[MOVE(2, Z_REMOVED)])],
    59438930: [quick(Z_HAND | Z_MZONE, opt=True, cost=[COND_CHAIN_FIELD(), SELF(0), MOVE(0)],      # Ghost Ogre & Snow Rabbit
                     target=[CHAINCARD(1, ONFIELD)], resolve=[MOVE(1)])],
    18430390: [quick(Z_MZONE, cost=[COND_CHAIN(TYPE_TRAP)],                                                  # Winged Rhynos
                     resolve=[SELF(0), FILTER(0, F(me=Z_MZONE, pos=P_FACEUP), within=0), MOVE(0, Z_HAND)])],
}


def assemble(codes, cards=CARDS):
    """codes : [C] ids → (ekind [C,E], prog [C,E,3,P,4], filt [C,E,NF,FW]) en numpy."""
    c = len(codes)
    ekind = np.zeros((c, E), np.int32)
    prog = np.zeros((c, E, 3, P, 4), np.int32)
    filt = np.zeros((c, E, NF, FW), np.int32)
    etrig = np.zeros((c, E), np.int32)
    eopt = np.zeros((c, E), np.int32)
    erng = np.zeros((c, E), np.int32)
    for i, code in enumerate(codes):
        for e, (kind, on, opt, rng, *progs) in enumerate(cards.get(code, ())):
            ekind[i, e] = kind
            etrig[i, e] = on
            eopt[i, e] = opt
            erng[i, e] = rng
            fl = []
            for g, pr in enumerate(progs):
                assert len(pr) < P, (code, g)  # toujours au moins un OP_END
                for j, (op, a, b, cc) in enumerate(pr):
                    if op == OP_FILTER:
                        if b not in fl:
                            fl.append(b)
                        b = fl.index(b)
                    prog[i, e, g, j] = (op, a, b, cc)
            assert len(fl) <= NF, code
            for j, f in enumerate(fl):
                filt[i, e, j] = f
    return dict(ekind=ekind, prog=prog, filt=filt, etrig=etrig, eopt=eopt, erng=erng)


# types / stats des cartes ci-dessus (pour les bancs de test sans base de données)
SPELL_CODES = [55144522, 12580477, 53129443, 46130346, 84257639, 70368879, 66788016, 83764718, 79571449, 18144506]
ST_TYPES = {5318639: TYPE_SPELL | TYPE_QUICKPLAY, 83968380: TYPE_TRAP, 24068492: TYPE_TRAP,
            77414722: TYPE_TRAP | TYPE_COUNTER, 3819470: TYPE_TRAP | TYPE_COUNTER,
            4206964: TYPE_TRAP, 53582587: TYPE_TRAP, 56120475: TYPE_TRAP, 44095762: TYPE_TRAP}
EFFECT_MONSTERS = {74131780: (1000, 1000, 2), 11384280: (1400, 1300, 4),  # Exiled Force, Cannon Soldier
                   26202165: (1000, 600, 3), 78010363: (1100, 1200, 4),    # Sangan, Witch of the Black Forest
                   54652250: (450, 600, 2), 83011277: (1400, 1100, 4),     # Man-Eater Bug, Mystic Tomato
                   24508238: (100, 100, 1), 59438930: (0, 1800, 3),        # D.D. Crow, Ghost Ogre & Snow Rabbit
                   18430390: (1800, 500, 4)}                               # Winged Rhynos


def test_bank(key, c):
    """c monstres normaux aléatoires (codes 0..c-1) + les cartes de CARDS (indices c..)."""
    import jax
    import core
    st = SPELL_CODES + list(ST_TYPES)
    codes = list(range(c)) + st + list(EFFECT_MONSTERS)
    atk = np.concatenate([np.asarray(jax.random.randint(key, (c,), 0, 30)) * 100, np.zeros(len(st), int),
                          [v[0] for v in EFFECT_MONSTERS.values()]])
    df = np.concatenate([np.asarray(jax.random.randint(jax.random.fold_in(key, 1), (c,), 0, 30)) * 100,
                         np.zeros(len(st), int), [v[1] for v in EFFECT_MONSTERS.values()]])
    lv = np.concatenate([np.asarray(jax.random.randint(jax.random.fold_in(key, 2), (c,), 1, 13)),
                         np.zeros(len(st), int), [v[2] for v in EFFECT_MONSTERS.values()]])
    ctype = [core.TYPE_MONSTER | core.TYPE_NORMAL] * c + [core.TYPE_SPELL] * len(SPELL_CODES) \
        + list(ST_TYPES.values()) + [core.TYPE_MONSTER | 0x20] * len(EFFECT_MONSTERS)
    return core.make_bank(np.array(codes), atk, df, lv, ctype=ctype, **assemble(codes))
