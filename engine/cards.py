"""Vraies cartes traduites en briques (jalons 2-3) et assembleur vers les tableaux de la banque.

Un effet = genre + programmes coût / cible / résolution. Les instructions sont des tuples (op, a, b, c) ;
dans FILTER, b est un filtre écrit en ligne (F(...)), que l'assembleur range dans la table des filtres.
Joueurs relatifs : ME = celui qui active, OPP = son adversaire.
"""
import numpy as np

import core

from core import (BIG, TRIG_FIELD, E, PT, PG_RESOLVE, EV_ATTACK, EV_BATTLE_DESTROYED, EV_FLIP, EV_FLIP_SUMMON, EV_SPSUMMON, EV_SUMMON, EV_TO_GRAVE,
                  FW, K_TRIG_F, K_TRIG_O, OP_COND_EVP, OP_COND_SELF, OP_COND_TURN, OP_EVENT, OP_LOCK, K_ACTIVATE,
                  K_IGNITION, NF, OP_CHAINCARD, OP_CHOOSE, OP_COND_CHAIN, OP_DAMAGE, OP_DRAW,
                  OP_FILTER, OP_MINATK, OP_MOVE, OP_NEGATE, OP_PAY_LP, OP_RECOVER, OP_REQ, OP_REQ_FREE, OP_SELF,
                  OP_SHUFFLE_HAND, OP_SPSUMMON, P, P_FACEUP, P_FU_ATK, P_FU_DEF, TYPE_COUNTER, TYPE_MONSTER,
                  P_FD_ATK, TYPE_QUICKPLAY, TYPE_SPELL, TYPE_TRAP, Z_DECK, Z_GRAVE, Z_HAND, Z_MZONE, Z_SZONE,
                  K_QUICK, Z_REMOVED, K_CONT, MW, M_CODE, M_SELF, M_VAL, M_PER, M_TP, M_PH, M_CNT, M_CMIN, M_CMAX, MC_ATK, MC_DEF, OP_MODIFY,
                  NSET, PH_BATTLE, TYPE_CONTINUOUS, MV_CONST, MV_HALF_BASE_ATK, MV_HALF_ATK, TD_TURN, TD_DCAL,
                  OP_COND_ATK, OP_ATTACKER, DMG_STEP, DMG_CAL, DMG_STAT, MC_BASE_ATK, MC_SET_ATK, MC_SET_DEF,
                  MC_FINAL_ATK, EV_PRE_DMG, OPT_CARD_NR, MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP,
                  MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP, ST_SUMMON_TURN, ST_FLIP_TURN, Z_EXTRA, MC_DISABLE, OP_COND_PHASE,
                  PH_MAIN1, PH_MAIN2, OP_FUSMAT, FK, TYPE_FUSION, OP_RITMAT, RIT_GREATER, RIT_EQUAL,
                  TYPE_RITUAL, Z_OVERLAY, MC_NO_DIRECT, TD_PERM, OPT_CARD, OP_LABEL, OP_COND_LABEL, OP_YESNO,
                  ST_SPSUM, ST_LINKSUM, ZX_COLINK, ZX_MAIN, MC_INDES_EFF)

ME, OPP = 0, 1


def F(me=0, op=0, pos=0, typ=0, not_self=0, atk=(-BIG, BIG), df=(-BIG, BIG), attr=0, race=0, setc=0, code=(0, 0),
      st=0, lv=(-BIG, BIG), spsum=0, nodis=0, fus=0, rit=0, zx=0):
    """setc : archétype (setcode ygopro) ; code : un ou deux noms de carte (ids ygopro) ; st : l'un de ces bits de
    statut ; spsum : peut être invoquée spécialement (IsCanBeSpecialSummoned) ; nodis : effets non annulés ;
    fus : monstre fusion invocable maintenant avec la main et le terrain (recette réalisable) ; rit : monstre rituel
    invocable maintenant par rituel en mode RIT_GREATER / RIT_EQUAL ; zx : bits ZX_* (co-lié, case principale)."""
    code = (code, code) if isinstance(code, int) else code
    return (me, op, pos, typ, not_self, atk[0], atk[1], df[0], df[1], attr, race, setc, code[0], code[1], st, lv[0], lv[1],
            spsum, nodis, fus, rit, zx)


def FILTER(r, f, within=None, tgt=False):  # ∩ registre within ; tgt : sélection avec ciblage (IsCanBeEffectTarget)
    return (OP_FILTER, r, f, (0 if within is None else within + 1) | (8 if tgt else 0))
def SELF(r): return (OP_SELF, r, 0, 0)
def REQ(r, n, maximum=False): return (OP_REQ, r, n, int(maximum))  # >= n, ou <= n
def REQ_FREE(who, zone, n): return (OP_REQ_FREE, who, zone, n)
def MINATK(dst, src): return (OP_MINATK, dst, src, 0)
def CHOOSE(dst, src, n, who=ME, upto=False, mn=None):
    """n cartes (moins s'il n'y a pas assez de candidats) ; upto : test d'activation sans minimum. mn : sélection de
    taille variable (Select*(mn, n) pendant la résolution) : le joueur s'arrête quand il veut après mn cartes (A_PASS)."""
    if mn is not None:
        return (OP_CHOOSE, dst, src, mn | (n << 4) | (who << 8) | (1 << 9))
    return (OP_CHOOSE, dst, src, (0 if upto else n) | (n << 4) | (who << 8))
def MOVE(r, zone=Z_GRAVE): return (OP_MOVE, r, zone, 0)
def DESTROY(r): return (OP_MOVE, r, Z_GRAVE, 1)  # Duel.Destroy : les indestructibles par effet restent, r = détruites
def SPSUMMON(r, pos, proc=False): return (OP_SPSUMMON, r, pos, int(proc))  # proc : invocation correcte (rituel)
def DRAW(who, n): return (OP_DRAW, who, n, 0)
def DAMAGE(who, n, per=None): return (OP_DAMAGE, who, n, 0 if per is None else per + 1)  # n × |registre per|
def RECOVER(who, n): return (OP_RECOVER, who, n, 0)
def SHUFFLE_HAND(who): return (OP_SHUFFLE_HAND, who, 0, 0)
def COND_CHAIN(typ): return (OP_COND_CHAIN, typ, 0, 0)  # le maillon précédent est une activation de carte typ
def COND_CHAIN_FIELD(): return (OP_COND_CHAIN, 0, 1, 0)  # … est un effet (pas une activation de carte) d'une carte sur le terrain
def CHAINCARD(r, zone=0): return (OP_CHAINCARD, r, zone, 0)  # sa carte, si elle est toujours dans zone (0 : magie/piège)
def NEGATE(): return (OP_NEGATE, 0, 0, 0)               # nie son activation
def PAY_LP(n): return (OP_PAY_LP, n, 0, 0)
def EVENT(r, ev, trig=False): return (OP_EVENT, r, ev, int(trig))  # cartes concernées par les événements ev (masque) ; trig : du dernier point d'événement (eg d'un déclencheur)
def COND_TURN(who): return (OP_COND_TURN, who, 0, 0)     # c'est le tour de who
def COND_EVP(ev, who): return (OP_COND_EVP, ev, who, 0)  # l'un des événements ev vient de who
def COND_SELF(zone=0, prev=0, st=0): return (OP_COND_SELF, zone, prev, st)  # la carte est dans zone / venait de prev / a l'un des statuts st
def LABEL(r): return (OP_LABEL, r, 0, 0)            # e:SetLabel(registre r non vide) à l'activation
def COND_LABEL(): return (OP_COND_LABEL, 0, 0, 0)   # à la résolution : le label est mis
def YESNO(): return (OP_YESNO, 0, 0, 0)             # Duel.SelectYesNo : non arrête l'effet
def LOCK(r): return (OP_LOCK, r, 0, 0)  # ME ne peut plus activer les effets des cartes de ce nom ce tour
def MODIFY(r, code, val=0, mode=MV_CONST, dur=TD_TURN, rd=False):
    """MC_*, valeur val ou MV_* ; durée TD_* (fin du tour par défaut) ; rd : RESET_DISABLE (effet de la carte sur
    elle-même, effacé si ses effets sont annulés)."""
    return (OP_MODIFY, r, code | (mode << 5) | (dur << 9) | (int(rd) << 13), val)
def COND_PHASE(ph): return (OP_COND_PHASE, ph, 0, 0)      # pendant l'une des phases ph (masque PH_*)
def FUSMAT(dst, src): return (OP_FUSMAT, dst, src, 0)     # matériaux du monstre fusion de src, choisis un par un
def RITMAT(dst, src, mode): return (OP_RITMAT, dst, src, mode)  # libérations du monstre rituel de src (RIT_*), une par une
def COND_ATK(target=False): return (OP_COND_ATK, int(target), 0, 0)  # la carte attaque (target : est attaquée)
def ATTACKER(r, tgt=False): return (OP_ATTACKER, r, int(tgt), 0)  # le monstre attaquant (tgt : s'il peut être ciblé)


def ritual_spell(code, mode):
    """Sort rituel (Ritual.AddProcGreaterCode / AddProcEqualCode) pour le monstre rituel code, depuis la main."""
    f = F(me=Z_HAND, code=code, rit=mode)
    return [act(target=[FILTER(0, f), REQ(0, 1)],
                resolve=[FILTER(0, f), CHOOSE(1, 0, 1), RITMAT(2, 1, mode), MOVE(2), SPSUMMON(1, P_FU_ATK | P_FU_DEF, True)])]


def ev(*codes):
    return sum(1 << c for c in codes)


# effet = (genre, on, opt, rng, coût, cible, résolution)
def act(cost=(), target=(), resolve=(), on=0, dmg=0, opt=0):
    """on : masque d'événements (ev(...)) ; 0 = activation libre. dmg : bits DMG_* (activation au damage step).
    opt : OPT_* (SetCountLimit(1, id) : une activation par tour et par nom)."""
    return (K_ACTIVATE, on, int(opt), 0, cost, target, resolve, dict(dmg=dmg))


def ignition(cost=(), target=(), resolve=(), opt=0, rng=0):
    """Effet d'ignition ; opt : OPT_* (OPT_CARD : SetCountLimit(1) sans code, par carte) ; rng : zones d'activation
    (Z_HAND | Z_MZONE | Z_GRAVE | Z_SZONE recto ; 0 = zone monstres)."""
    return (K_IGNITION, 0, int(opt), int(rng), cost, target, resolve)


def trigger(on, cost=(), target=(), resolve=(), optional=False, opt=False, rng=0, dmg=0):
    """Effet déclenché (SINGLE) par les événements `on` qui concernent la carte ; opt : une fois par tour (nom).
    rng ≠ 0 : déclencheur FIELD (TRIG_FIELD), par les événements de n'importe quelle carte quand celle-ci est dans les
    zones rng (eg : EVENT(r, on, trig=True)). dmg : DMG_STEP (EFFECT_FLAG_DAMAGE_STEP) ; seul un déclencheur FIELD
    optionnel le lit (sans lui, pas d'activation au damage step, sauf sur un événement de combat)."""
    return (K_TRIG_O if optional else K_TRIG_F, on | (TRIG_FIELD if rng else 0), int(opt), int(rng), cost, target, resolve,
            dict(dmg=dmg))


def quick(rng, cost=(), target=(), resolve=(), opt=False, on=0, dmg=0):
    """Effet rapide de monstre (vitesse 2) activable depuis les zones rng (main, zone monstres recto) ; on : seulement
    en réponse à ces événements ; opt : True (une fois par tour par nom) ou OPT_*."""
    return (K_QUICK, on, int(opt), rng, cost, target, resolve, dict(dmg=dmg))


def cont(rng, code, val, tgt=None, per=None, tp=0, ph=0, cnt=None):
    """Effet continu appliqué tant que la carte est recto dans les zones rng : code MC_*, valeur val (× nombre de
    cartes du filtre per), sur les monstres recto du filtre tgt (EFFECT_TYPE_FIELD, zones relatives au contrôleur de
    la carte) ou sur la carte elle-même (tgt=None : SINGLE_RANGE). Conditions : tour de tp (ME + 1 / OPP + 1),
    phases ph, cnt = (filtre, min, max) sur le nombre de cartes du filtre."""
    assert per is None or cnt is None  # un seul filtre de compte (filtre 1)
    mod = {M_CODE: code, M_SELF: int(tgt is None), M_VAL: val, M_PER: int(per is not None), M_TP: tp, M_PH: ph}
    if cnt is not None:
        mod.update({M_CNT: 1, M_CMIN: cnt[1], M_CMAX: cnt[2]})
    filters = [tgt or F(), per or (cnt[0] if cnt else F())]
    return (K_CONT, 0, 0, rng, (), (), (), dict(mod=mod, filters=filters))


MY_DECK = F(me=Z_DECK)


def knightmare(tgt, indes):
    """Knightmare Phoenix / Cerberus : quand il est invoqué par lien (une fois par tour, par nom) : défausser 1 carte,
    cibler une carte adverse (filtre tgt), la détruire ; s'il était co-lié à l'activation (label), piocher 1 (oui / non).
    Continu : les monstres co-liés de son contrôleur ne peuvent pas être détruits (indes : au combat / par les effets)."""
    colinked = F(me=Z_MZONE, zx=ZX_COLINK)
    return [trigger(ev(EV_SPSUMMON), optional=True, opt=True,
                    cost=[COND_SELF(st=ST_LINKSUM), FILTER(0, F(me=Z_HAND)), CHOOSE(1, 0, 1), MOVE(1)],
                    target=[FILTER(2, tgt, tgt=True), CHOOSE(3, 2, 1), SELF(0), FILTER(0, colinked, within=0), LABEL(0)],
                    resolve=[DESTROY(3), REQ(3, 1), COND_LABEL(), FILTER(0, MY_DECK), REQ(0, 1), YESNO(), DRAW(ME, 1)]),
            cont(Z_MZONE, indes, 0, tgt=colinked)]
OPP_MON = F(op=Z_MZONE)
ALL_MON = F(me=Z_MZONE, op=Z_MZONE)
OPP_FACEUP_MON = F(op=Z_MZONE, pos=P_FACEUP)
OPP_ST = F(op=Z_MZONE | Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP, not_self=1)
GRAVE_MON = F(me=Z_GRAVE, op=Z_GRAVE, typ=TYPE_MONSTER)
GRAVE_MON_SS = F(me=Z_GRAVE, op=Z_GRAVE, typ=TYPE_MONSTER, spsum=1)
BIG_FACEUP = F(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP, atk=(1000, BIG))
OPP_ATK_MON = F(op=Z_MZONE, pos=P_FU_ATK | P_FD_ATK)
ONFIELD = Z_MZONE | Z_SZONE
DARK = 0x20
SEARCH_ATK1500 = F(me=Z_DECK, typ=TYPE_MONSTER, atk=(-BIG, 1500))
SEARCH_DEF1500 = F(me=Z_DECK, typ=TYPE_MONSTER, df=(-BIG, 1500))
DECK_DARK1500 = F(me=Z_DECK, typ=TYPE_MONSTER, atk=(-BIG, 1500), attr=DARK, spsum=1)
FIELD_ST = F(me=Z_MZONE | Z_SZONE, op=Z_MZONE | Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP, not_self=1)
EARTH, WATER, WIND = 0x1, 0x2, 0x8
SET_ELEMENTAL_HERO = 0x3008
DARK_MAGICIAN, MAGICIAN_OF_BLACK_CHAOS = 46986414, 30208479
FACEUP_MON = F(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP)
RACE_INSECT = 0x800
STAT_DS = DMG_STEP | DMG_STAT  # EFFECT_FLAG_DAMAGE_STEP + aux.StatChangeDamageStepCondition
RACE_WARRIOR, RACE_DRAGON, LIGHT = 0x1, 0x2000, 0x10
ALL_ZONES = Z_DECK | Z_HAND | Z_MZONE | Z_SZONE | Z_GRAVE | Z_REMOVED | Z_EXTRA
TYPE_EFFECT = 0x20

CARDS = {
    55144522: [act(target=[FILTER(0, MY_DECK), REQ(0, 2)], resolve=[DRAW(ME, 2)])],            # Pot of Greed
    12580477: [act(target=[FILTER(0, OPP_MON), REQ(0, 1)], resolve=[FILTER(0, OPP_MON), DESTROY(0)])],  # Raigeki
    53129443: [act(target=[FILTER(0, ALL_MON), REQ(0, 1)], resolve=[FILTER(0, ALL_MON), DESTROY(0)])],  # Dark Hole
    46130346: [act(resolve=[DAMAGE(OPP, 500)])],                                                # Hinotama
    84257639: [act(resolve=[RECOVER(ME, 1000)])],                                               # Dian Keto
    70368879: [act(target=[FILTER(0, MY_DECK), REQ(0, 1)], resolve=[DRAW(ME, 1), RECOVER(OPP, 1000)])],  # Upstart
    66788016: [act(target=[FILTER(0, OPP_FACEUP_MON), REQ(0, 1)],                               # Fissure
                   resolve=[FILTER(0, OPP_FACEUP_MON), REQ(0, 1), MINATK(1, 0), CHOOSE(2, 1, 1), DESTROY(2)])],
    83764718: [act(target=[REQ_FREE(ME, Z_MZONE, 1), FILTER(0, GRAVE_MON_SS, tgt=True), CHOOSE(1, 0, 1)],  # Monster Reborn
                   resolve=[SPSUMMON(1, P_FU_ATK | P_FU_DEF)])],
    79571449: [act(target=[FILTER(0, MY_DECK), REQ(0, 3)],                                      # Graceful Charity
                   resolve=[DRAW(ME, 3), SHUFFLE_HAND(ME), FILTER(0, F(me=Z_HAND)), CHOOSE(1, 0, 2), MOVE(1)])],
    18144506: [act(target=[FILTER(0, OPP_ST), REQ(0, 1)], resolve=[FILTER(0, OPP_ST), DESTROY(0)])],  # Feather Duster
    74131780: [ignition(cost=[SELF(0), MOVE(0)],                                                # Exiled Force
                        target=[FILTER(1, F(me=Z_MZONE, op=Z_MZONE, not_self=1), tgt=True), CHOOSE(2, 1, 1)],
                        resolve=[DESTROY(2)])],
    11384280: [ignition(cost=[FILTER(0, F(me=Z_MZONE)), CHOOSE(1, 0, 1), MOVE(1)],              # Cannon Soldier
                        resolve=[DAMAGE(OPP, 500)])],
    # jalon 3a : chaînes
    83968380: [act(target=[FILTER(0, MY_DECK), REQ(0, 1)], resolve=[DRAW(ME, 1)])],               # Jar of Greed
    24068492: [act(target=[FILTER(0, OPP_MON), REQ(0, 1)],                                       # Just Desserts
                   resolve=[FILTER(0, OPP_MON), DAMAGE(OPP, 500, per=0)])],
    5318639: [act(target=[FILTER(0, FIELD_ST, tgt=True), CHOOSE(1, 0, 1)], resolve=[DESTROY(1)])],            # Mystical Space Typhoon
    77414722: [act(cost=[COND_CHAIN(TYPE_SPELL), FILTER(0, F(me=Z_HAND)), CHOOSE(1, 0, 1), MOVE(1)],  # Magic Jammer
                   resolve=[NEGATE(), CHAINCARD(0), DESTROY(0)])],
    3819470: [act(cost=[COND_CHAIN(TYPE_TRAP), PAY_LP(1000)], resolve=[NEGATE(), CHAINCARD(0), DESTROY(0)])],  # Seven Tools
    # jalon 3b : pièges sur événement
    4206964: [act(on=ev(EV_SUMMON, EV_FLIP_SUMMON),                                              # Trap Hole
                  target=[COND_EVP(ev(EV_SUMMON, EV_FLIP_SUMMON), OPP), EVENT(0, ev(EV_SUMMON, EV_FLIP_SUMMON)),
                          FILTER(1, BIG_FACEUP, within=0, tgt=True), REQ(1, 1)],
                  resolve=[FILTER(1, BIG_FACEUP, within=1), DESTROY(1)])],
    53582587: [act(on=ev(EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON),                                # Torrential Tribute
                   target=[FILTER(0, ALL_MON), REQ(0, 1)], resolve=[FILTER(0, ALL_MON), DESTROY(0)])],
    56120475: [act(on=ev(EV_ATTACK), target=[COND_TURN(OPP), EVENT(0, ev(EV_ATTACK)),            # Sakuretsu Armor
                                             FILTER(0, ALL_MON, within=0, tgt=True), REQ(0, 1)], resolve=[DESTROY(0)])],
    44095762: [act(on=ev(EV_ATTACK), target=[COND_TURN(OPP), FILTER(0, OPP_ATK_MON), REQ(0, 1)],  # Mirror Force
                   resolve=[FILTER(0, OPP_ATK_MON), DESTROY(0)])],
    26202165: [trigger(ev(EV_TO_GRAVE), opt=True, target=[COND_SELF(prev=ONFIELD)],                       # Sangan
                       resolve=[FILTER(0, SEARCH_ATK1500), CHOOSE(1, 0, 1, upto=True), MOVE(1, Z_HAND), LOCK(1)])],
    78010363: [trigger(ev(EV_TO_GRAVE), opt=True, target=[COND_SELF(prev=ONFIELD)],         # Witch of the Black Forest
                       resolve=[FILTER(0, SEARCH_DEF1500), CHOOSE(1, 0, 1, upto=True), MOVE(1, Z_HAND), LOCK(1)])],
    54652250: [trigger(ev(EV_FLIP), target=[FILTER(0, ALL_MON, tgt=True), CHOOSE(1, 0, 1, upto=True)],              # Man-Eater Bug
                       resolve=[DESTROY(1)])],
    83011277: [trigger(ev(EV_BATTLE_DESTROYED), optional=True,                                               # Mystic Tomato
                       target=[COND_SELF(zone=Z_GRAVE), REQ_FREE(ME, Z_MZONE, 1), FILTER(0, DECK_DARK1500), REQ(0, 1)],
                       resolve=[REQ_FREE(ME, Z_MZONE, 1), FILTER(0, DECK_DARK1500), CHOOSE(1, 0, 1, upto=True),
                                SPSUMMON(1, P_FU_ATK)])],
    # jalon 3b, volet 3 : effets rapides de monstre
    24508238: [quick(Z_HAND, cost=[SELF(0), MOVE(0)],                                                       # D.D. Crow
                     target=[FILTER(1, F(op=Z_GRAVE), tgt=True), CHOOSE(2, 1, 1)], resolve=[MOVE(2, Z_REMOVED)])],
    59438930: [quick(Z_HAND | Z_MZONE, opt=True, cost=[COND_CHAIN_FIELD(), SELF(0), MOVE(0)],      # Ghost Ogre & Snow Rabbit
                     target=[CHAINCARD(1, ONFIELD)], resolve=[DESTROY(1)])],
    18430390: [quick(Z_MZONE, cost=[COND_CHAIN(TYPE_TRAP)],                                                  # Winged Rhynos
                     resolve=[SELF(0), FILTER(0, F(me=Z_MZONE, pos=P_FACEUP), within=0), MOVE(0, Z_HAND)])],
    # jalon 4, volet 1 : archétypes, effets continus de stats, modificateurs temporaires
    213326: [act(target=[FILTER(0, F(me=Z_DECK, typ=TYPE_MONSTER, setc=SET_ELEMENTAL_HERO)), REQ(0, 1)],  # E - Emergency Call
                 resolve=[FILTER(0, F(me=Z_DECK, typ=TYPE_MONSTER, setc=SET_ELEMENTAL_HERO)), CHOOSE(1, 0, 1),
                          MOVE(1, Z_HAND)])],
    28470714: [cont(Z_MZONE, MC_ATK, 500, tgt=F(me=Z_MZONE, op=Z_MZONE, attr=WIND)),                       # Bladefly
               cont(Z_MZONE, MC_ATK, -400, tgt=F(me=Z_MZONE, op=Z_MZONE, attr=EARTH))],
    10012614: [act(), cont(Z_SZONE, MC_ATK, 200, tgt=F(me=Z_MZONE), tp=ME + 1, ph=PH_BATTLE)],              # Banner of Courage
    38033121: [cont(Z_MZONE, MC_ATK, 300, per=F(me=Z_GRAVE, op=Z_GRAVE,                                    # Dark Magician Girl
                                              code=(MAGICIAN_OF_BLACK_CHAOS, DARK_MAGICIAN)))],
    68833958: [act(target=[FILTER(0, FACEUP_MON, tgt=True), CHOOSE(1, 0, 1)],                                       # Back-Up Rider
                   resolve=[FILTER(2, FACEUP_MON, within=1), MODIFY(2, MC_ATK, 1500)])],
    13314457: [trigger(ev(EV_TO_GRAVE), target=[COND_SELF(prev=ONFIELD)],                                 # Deepsea Macrotrema
                       resolve=[FILTER(0, F(me=Z_MZONE, pos=P_FACEUP, attr=WATER)), MODIFY(0, MC_ATK, 500)])],
    25109950: [ignition(cost=[FILTER(0, F(me=Z_MZONE, not_self=1)), CHOOSE(1, 0, 1), MOVE(1)],         # Little Swordsman of Aile
                        target=[SELF(2)], resolve=[FILTER(3, F(me=Z_MZONE, pos=P_FACEUP), within=2), MODIFY(3, MC_ATK, 700, rd=True)])],
    # jalon 4, volet 2 : damage step, SET_ATTACK / SET_ATTACK_FINAL / SET_BASE_ATTACK
    70046172: [act(target=[FILTER(0, FACEUP_MON, tgt=True), CHOOSE(1, 0, 1)], dmg=STAT_DS,                         # Rush Recklessly
                   resolve=[FILTER(2, FACEUP_MON, within=1), MODIFY(2, MC_ATK, 700)])],
    55713623: [act(target=[FILTER(0, FACEUP_MON, tgt=True), CHOOSE(1, 0, 1)], dmg=STAT_DS,                         # Shrink
                   resolve=[FILTER(2, FACEUP_MON, within=1), MODIFY(2, MC_BASE_ATK, mode=MV_HALF_BASE_ATK)])],
    44209392: [act(target=[FILTER(0, FACEUP_MON, tgt=True), CHOOSE(1, 0, 1)], dmg=STAT_DS,                         # Castle Walls
                   resolve=[FILTER(2, FACEUP_MON, within=1), MODIFY(2, MC_DEF, 500)])],
    94156050: [act(target=[FILTER(0, FACEUP_MON), REQ(0, 1)], dmg=STAT_DS,                               # Reverse Glasses
                   resolve=[FILTER(0, FACEUP_MON), MODIFY(0, MC_FINAL_ATK, mode=MV_HALF_ATK)])],
    62340868: [quick(Z_MZONE, opt=OPT_CARD_NR, on=ev(EV_PRE_DMG), cost=[COND_TURN(OPP), COND_ATK(target=True)],  # Kazejin
                     target=[ATTACKER(0, tgt=True), REQ(0, 1)], resolve=[MODIFY(0, MC_FINAL_ATK, 0, dur=TD_DCAL)])],
    # jalon 4, volet 3a : interdictions et protections
    85742772: [act(), cont(Z_SZONE, MC_NO_ATTACK, 0, tgt=F(me=Z_MZONE, op=Z_MZONE, lv=(4, BIG)))],         # Gravity Bind
    59380081: [cont(Z_MZONE, MC_NO_ATTACK, 0, tgt=F(op=Z_MZONE, st=ST_SUMMON_TURN | ST_FLIP_TURN))],      # Big-Tusked Mammoth
    10375182: [cont(Z_MZONE, MC_NO_BTARGET, 0, cnt=(F(me=Z_MZONE), 2, BIG)),                              # Command Knight
               cont(Z_MZONE, MC_ATK, 400, tgt=F(me=Z_MZONE, race=RACE_WARRIOR))],
    52077741: [cont(Z_MZONE, MC_INDES_BATTLE, 1900)],                                                   # Obnoxious Celtic Guard
    17985575: [cont(Z_MZONE, MC_NO_ETARGET, 0, tgt=F(me=Z_MZONE, op=Z_MZONE, race=RACE_DRAGON))],         # Lord of D.
    86188410: [cont(Z_MZONE, MC_IMM_TRAP, 0)],                                                         # Elemental HERO Wildheart
    72845813: [cont(Z_MZONE, MC_NO_SS, 0, tgt=F(me=ALL_ZONES, op=ALL_ZONES, attr=LIGHT))],                # Ally of Justice Quarantine
    15960641: [cont(Z_MZONE, MC_P_NO_TRAP, 2, ph=PH_BATTLE)],                                           # Mirage Dragon
    # jalon 4, volet 3b : annulation d'effets
    82732705: [act(cost=[PAY_LP(1000)]),                                                                   # Skill Drain
               cont(Z_SZONE, MC_DISABLE, 0, tgt=F(me=Z_MZONE, op=Z_MZONE, typ=TYPE_EFFECT))],
    25789292: [act(target=[FILTER(0, FACEUP_MON, tgt=True), CHOOSE(1, 0, 1)], dmg=STAT_DS,                 # Forbidden Chalice
                   resolve=[FILTER(2, FACEUP_MON, within=1), MODIFY(2, MC_ATK, 400), MODIFY(2, MC_DISABLE)])],
    97268402: [quick(Z_HAND, cost=[COND_TURN(OPP), COND_PHASE(PH_MAIN1 | PH_MAIN2), SELF(0), MOVE(0)],         # Effect Veiler
                     target=[FILTER(1, F(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_EFFECT, nodis=1), tgt=True), CHOOSE(2, 1, 1)],
                     resolve=[FILTER(3, F(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_MONSTER, nodis=1), within=2),
                              MODIFY(3, MC_DISABLE)])],
    # jalon 5, volet a : fusion
    24094653: [act(target=[FILTER(0, F(me=Z_EXTRA, fus=1)), REQ(0, 1)],                                    # Polymerization
                   resolve=[FILTER(0, F(me=Z_EXTRA, fus=1)), CHOOSE(1, 0, 1), FUSMAT(2, 1), MOVE(2),
                            SPSUMMON(1, P_FU_ATK | P_FU_DEF)])],
    # jalon 5, volet d : Xyz ; détacher = choisir parmi ses propres matériaux (filtre Z_OVERLAY) et les envoyer au cimetière
    69610924: [ignition(opt=OPT_CARD, cost=[FILTER(0, F(me=Z_OVERLAY)), CHOOSE(1, 0, 1), MOVE(1)],      # Leviathan Dragon
                        resolve=[SELF(0), FILTER(1, F(me=Z_MZONE, pos=P_FACEUP), within=0),
                                 MODIFY(1, MC_ATK, 500, dur=TD_PERM, rd=True)]),
               cont(Z_MZONE, MC_NO_DIRECT, 0, cnt=(F(me=Z_OVERLAY), 0, 0))],              # sans matériau : pas d'attaque directe
    # jalon 5, volet b : rituel (sorts rituels : choisir le monstre en main, libérations, invocation correcte)
    55761792: ritual_spell(5405694, RIT_GREATER),                                                          # Black Luster Ritual
    8198712: ritual_spell(72426662, RIT_EQUAL),                                                            # End of the World
    72426662: [ignition(cost=[PAY_LP(2000)], target=[FILTER(0, F(me=ONFIELD, op=ONFIELD, not_self=1)), REQ(0, 1)],  # Demise
                        resolve=[FILTER(0, F(me=ONFIELD, op=ONFIELD, not_self=1)), DESTROY(0)])],
    42364374: [cont(Z_MZONE, MC_SET_ATK, 1000, cnt=(F(me=Z_MZONE, pos=P_FACEUP, race=RACE_INSECT, not_self=1), 0, 0)),  # Arsenal Bug
               cont(Z_MZONE, MC_SET_DEF, 1000, cnt=(F(me=Z_MZONE, pos=P_FACEUP, race=RACE_INSECT, not_self=1), 0, 0))],
    # jalon 5, volet e : Knightmare (co-liens)
    2857636: knightmare(F(op=ONFIELD, typ=TYPE_SPELL | TYPE_TRAP), MC_INDES_BATTLE),                      # Knightmare Phoenix
    75452921: knightmare(F(op=Z_MZONE, st=ST_SPSUM, zx=ZX_MAIN), MC_INDES_EFF),                         # Knightmare Cerberus
    # jalon 5, volet f : monstres pendule normaux (activation en zone pendule, sans effet pendule)
    17390179: [act()],                                                                                   # Flash Knight
    19474136: [act()],                                                                                   # Mandragon
    28363749: [act()],                                                                                   # Fire Opal Head
    32872239: [act()],                                                                     # Gallugallud the Endless Enigma
    83980492: [act()],                                                                                   # Dragong
}


class FusionRecipe(list):
    def __init__(self, filters, substitute=True):
        super().__init__(filters)
        if type(substitute) is bool:
            self.substitute_mask = sum(1 << j for j, f in enumerate(self) if substitute and f[core.F_CODE1] > 0)
        else:
            self.substitute_mask = int(substitute)
            assert 0 <= self.substitute_mask < (1 << len(self))


def assemble(codes, cards=CARDS, fusion=None, synchro=None, xyz=None, link=None, ritual=None):
    """codes : [C] ids → (ekind [C,E], prog [C,E,3,P,4], filt [C,E,NF,FW]) en numpy.
    fusion / synchro / xyz / link / ritual : recettes (mêmes formats que FUSION, SYNCHRO, XYZ, LINK, RITUAL) ;
    None : celles de ce module (cartes procédurales : gen/cardgen.py)."""
    FUSION_, SYNCHRO_, XYZ_, LINK_, RITUAL_ = (FUSION if fusion is None else fusion, SYNCHRO if synchro is None else synchro,
                                               XYZ if xyz is None else xyz, LINK if link is None else link,
                                               RITUAL if ritual is None else ritual)
    c = len(codes)
    subzone = np.zeros(c, np.int32)
    revive = np.zeros(c, np.int32)
    ekind = np.zeros((c, E), np.int32)
    prog = np.zeros((c, E, 3, P, 4), np.int32)
    filt = np.zeros((c, E, NF, FW), np.int32)
    etrig = np.zeros((c, E), np.int32)
    eopt = np.zeros((c, E), np.int32)
    erng = np.zeros((c, E), np.int32)
    emod = np.zeros((c, E, MW), np.int32)
    edmg = np.zeros((c, E), np.int32)
    for i, code in enumerate(codes):
        for e, (kind, on, opt, rng, *progs) in enumerate(cards.get(code, ())):
            ekind[i, e] = kind
            etrig[i, e] = on
            eopt[i, e] = opt
            erng[i, e] = rng
            extra = progs[3] if len(progs) > 3 else {}
            progs = progs[:3]
            edmg[i, e] = extra.get("dmg", 0)
            revive[i] |= extra.get("revive", 0)
            fl = list(extra.get("filters", ()))  # effets continus : filtres 0 (cibles), 1 (compte ou condition)
            for k, v in extra.get("mod", {}).items():
                emod[i, e, k] = v
            if extra.get("mod", {}).get(M_CODE) == core.MC_FUS_SUB:
                subzone[i] |= rng
            for g, pr in enumerate(progs):
                assert len(pr) < P, (code, g)  # toujours au moins un OP_END
                assert g == PG_RESOLVE or len(pr) <= PT, (code, g)  # test_ok ne lit que PT instructions
                for j, (op, a, b, cc) in enumerate(pr):
                    if op == OP_FILTER:
                        if b not in fl:
                            fl.append(b)
                        b = fl.index(b)
                    prog[i, e, g, j] = (op, a, b, cc)
            assert len(fl) <= NF, code
            for j, f in enumerate(fl):
                filt[i, e, j] = f
    fmat = np.zeros((c, FK, len(F())), np.int32)
    fmn = np.zeros(c, np.int32)
    fsub = np.zeros(c, np.int32)
    erl = np.zeros(c, np.int32)
    dflt = F()
    for i, code in enumerate(codes):
        for j, f in enumerate(FUSION_.get(code, ())):
            # recette : seuls les champs de core.MAT_FIELDS sont lus (la zone est la main et le terrain)
            assert all(f[k] == dflt[k] for k in range(len(f)) if k not in core.MAT_FIELDS + (core.F_LOC_ME,)), code
            fmat[i, j] = f
            if (getattr(FUSION_.get(code), "substitute_mask", (1 << FK) - 1) & (1 << j)) and f[core.F_CODE1] > 0:
                fsub[i] |= 1 << j
        fmn[i] = len(FUSION_.get(code, ()))
        erl[i] = int(revive[i]) or int(code in FUSION_ or code in RITUAL_ or code in SYNCHRO_ or code in XYZ_ or code in LINK_)
    smat = np.zeros((c, 2, len(F())), np.int32)
    syn = np.zeros(c, np.int32)
    for i, code in enumerate(codes):
        if code in SYNCHRO_:
            for j, f in enumerate(SYNCHRO_[code]):
                assert all(f[k] == dflt[k] for k in range(len(f)) if k not in core.MAT_FIELDS + (core.F_LOC_ME,)), code
                smat[i, j] = f
            syn[i] = 1
    xmat = np.zeros((c, len(F())), np.int32)
    xct = np.zeros(c, np.int32)
    for i, code in enumerate(codes):
        if code in XYZ_:
            f, k = XYZ_[code]
            assert all(f[j] == dflt[j] for j in range(len(f)) if j not in core.MAT_FIELDS + (core.F_LOC_ME,)), code
            xmat[i], xct[i] = f, k
    lmat = np.zeros((c, len(F())), np.int32)
    lmin, lmax, lneed, ldiff = (np.zeros(c, np.int32) for _ in range(4))
    for i, code in enumerate(codes):
        if code in LINK_:
            f, lmin[i], lmax[i], lneed[i], *dd = LINK_[code]
            ldiff[i] = dd[0] if dd else 0
            assert all(f[j] == dflt[j] for j in range(len(f)) if j not in core.MAT_FIELDS + (core.F_LOC_ME,)), code
            lmat[i] = f
    return dict(ekind=ekind, prog=prog, filt=filt, etrig=etrig, eopt=eopt, erng=erng, emod=emod, edmg=edmg,
                fmat=fmat, fmn=fmn, fsub=fsub, submat=(subzone[:, None] if subzone.any() else np.zeros((c, 0), np.int32)), erl=erl, smat=smat, syn=syn, xmat=xmat, xct=xct, lmat=lmat, lmin=lmin, lmax=lmax,
                lneed=lneed, ldiff=ldiff)


def setcodes(sc):
    """setcode 64 bits de la base (4 archétypes de 16 bits) → [NSET]."""
    return [(sc >> (16 * k)) & 0xFFFF for k in range(NSET)]


# types / stats des cartes ci-dessus (pour les bancs de test sans base de données)
SPELL_CODES = [55144522, 12580477, 53129443, 46130346, 84257639, 70368879, 66788016, 83764718, 79571449, 18144506]
ST_TYPES = {5318639: TYPE_SPELL | TYPE_QUICKPLAY, 83968380: TYPE_TRAP, 24068492: TYPE_TRAP,
            77414722: TYPE_TRAP | TYPE_COUNTER, 3819470: TYPE_TRAP | TYPE_COUNTER,
            4206964: TYPE_TRAP, 53582587: TYPE_TRAP, 56120475: TYPE_TRAP, 44095762: TYPE_TRAP}
# jalon 4 (tenues hors des anciens modes de compare.py, pour que leurs graines rejouent les mêmes decks)
J4_SPELLS = {213326: TYPE_SPELL, 10012614: TYPE_SPELL | TYPE_CONTINUOUS, 68833958: TYPE_SPELL}
J4_MONSTERS = [28470714, 38033121, 13314457, 25109950]
J4 = set(J4_SPELLS) | set(J4_MONSTERS)
# volet 2 (mode dmg de compare.py ; hors J4 pour que les graines du mode cont rejouent les mêmes decks)
J4B_SPELLS = {70046172: TYPE_SPELL | TYPE_QUICKPLAY, 55713623: TYPE_SPELL | TYPE_QUICKPLAY, 44209392: TYPE_TRAP,
              94156050: TYPE_TRAP}
J4B_MONSTERS = [62340868, 42364374]
J4B = set(J4B_SPELLS) | set(J4B_MONSTERS)
# volet 3a (mode flag de compare.py)
J4C_SPELLS = {85742772: TYPE_TRAP | TYPE_CONTINUOUS}
J4C_MONSTERS = [59380081, 10375182, 52077741, 17985575, 86188410, 72845813, 15960641]
J4C = set(J4C_SPELLS) | set(J4C_MONSTERS)
# volet 3b (mode neg de compare.py)
J4D_SPELLS = {82732705: TYPE_TRAP | TYPE_CONTINUOUS, 25789292: TYPE_SPELL | TYPE_QUICKPLAY}
J4D_MONSTERS = [97268402]
J4D = set(J4D_SPELLS) | set(J4D_MONSTERS)
# jalon 5 : recettes de fusion (un filtre par matériau, pris sur la main et le terrain) ; EnableReviveLimit pour tous
def MAT(**kw):  # filtre d'un matériau de fusion : main et terrain du joueur
    return F(me=Z_HAND | Z_MZONE, **kw)


FUSION = {
    45231177: [MAT(code=34460851), MAT(code=44287299)],      # Flame Swordsman : Flame Manipulator + Masaki
    66889139: [MAT(code=6368038), MAT(code=28279543)],       # Gaia the Dragon Champion : Gaia the Fierce Knight + Curse of Dragon
    23995346: [MAT(code=89631139)] * 3,                      # Blue-Eyes Ultimate Dragon : 3 Blue-Eyes White Dragon
}
FUSION_STATS = {45231177: (1800, 1600, 5), 66889139: (2600, 2100, 7), 23995346: (4500, 3800, 12)}
MATERIAL_STATS = {34460851: (900, 1000, 3), 44287299: (1100, 1100, 4), 6368038: (2300, 2100, 7), 28279543: (2000, 1500, 5),
                  89631139: (3000, 2500, 8)}  # matériaux (monstres normaux) : banque de test
J5A_SPELLS = {24094653: TYPE_SPELL}
J5A = set(J5A_SPELLS)  # cartes du volet 5a (mode fus de compare.py)
# volet 5b : monstres rituels (EnableReviveLimit) et leurs sorts
RITUAL = {5405694: (3000, 2500, 8), 72426662: (2400, 2000, 8)}  # Black Luster Soldier, Demise, King of Armageddon
RITUAL_TYPES = {5405694: TYPE_MONSTER | TYPE_RITUAL, 72426662: TYPE_MONSTER | TYPE_RITUAL | TYPE_EFFECT}
J5B_SPELLS = {55761792: TYPE_SPELL | TYPE_RITUAL, 8198712: TYPE_SPELL | TYPE_RITUAL}
J5B = set(J5B_SPELLS) | set(RITUAL)
# volet 5c : synchro (un Syntoniseur + un ou plusieurs non-Syntoniseurs ; filtres sur MAT_FIELDS) ; EnableReviveLimit
SYNCHRO = {
    97204936: (MAT(), MAT()),                                 # Gaia Knight, the Force of Earth (6)
    45815891: (MAT(), MAT()),                                 # Scrap Archfiend (7)
    16527176: (MAT(attr=EARTH), MAT(attr=EARTH)),             # Naturia Leodrake (9) : Syntoniseur TERRE + non-Syntoniseurs TERRE
}
SYNCHRO_STATS = {97204936: (2600, 800, 6, EARTH), 45815891: (2700, 1800, 7, EARTH), 16527176: (3000, 1800, 9, EARTH)}
TUNER_STATS = {74093656: (1600, 200, 3, EARTH), 487395: (400, 1200, 1, WATER), 21615956: (100, 2000, 1, 0x4),  # Tune Warrior, Water Spirit, Flamvell Guard
               68505803: (1400, 1200, 3, 0x20), 40155554: (1800, 1400, 5, 0x20)}  # Genex Controller, Ally Mind
J5C = set(TUNER_STATS)  # Syntoniseurs normaux (mode syn de compare.py)
# volet 5d : Xyz (matériaux : ct monstres de niveau = rang, filtre sur MAT_FIELDS) ; EnableReviveLimit
XYZ = {71594310: (MAT(), 2),                                  # Gem-Knight Pearl (rang 4)
       9053187: (MAT(), 3),                                   # Gagagigo the Risen (rang 4)
       69610924: (MAT(), 2)}                                  # Number 17: Leviathan Dragon (rang 3)
XYZ_STATS = {71594310: (2600, 1900, 4, EARTH), 9053187: (2950, 2800, 4, WATER), 69610924: (2000, 0, 3, WATER)}
J5D = {69610924}  # Xyz à effet (cartes de CARDS)
# volet 5e : Link (filtre des matériaux sur MAT_FIELDS, nombre min / max, un Link parmi eux) ; EnableReviveLimit
LINK = {77637979: (MAT(), 2, 2, 0),                           # LANphorhynchus (lien 2) : 2 monstres
        67598234: (MAT(), 2, 3, 0),                           # Gaia Saber, the Lightning Shadow (lien 3) : 2+ monstres
        41248270: (MAT(), 3, 3, 0),                           # Traffic Ghost (lien 3) : 3 monstres
        5402805: (MAT(), 2, 3, 1),                            # Berserker of the Tenyi (lien 3) : 2-3 monstres dont un Link
        # cartes fictives de test (driver.TEST_CARDS) : Extra Link
        990000001: (MAT(), 2, 2, 0),                          # Test Link TL-TR (lien 2)
        990000002: (MAT(), 2, 2, 0),                          # Test Link T-R
        990000003: (MAT(), 2, 2, 0),                          # Test Link L-T
        # (filtre, min, max, un Link parmi eux, noms différents)
        2857636: (MAT(), 2, 2, 0, 1),                         # Knightmare Phoenix (lien 2) : 2 monstres de noms différents
        75452921: (MAT(), 2, 2, 0, 1)}                        # Knightmare Cerberus (lien 2)
# (ATK, flèches, lien, attribut)
LINK_STATS = {77637979: (1200, 5, 2, 0x10), 67598234: (2600, 42, 3, EARTH), 41248270: (1800, 7, 3, 0x20),
              5402805: (3000, 388, 3, 0x20), 990000001: (1500, 0x140, 2, 0x10), 990000002: (1400, 0xA0, 2, 0x10),
              990000003: (1400, 0x88, 2, 0x10), 2857636: (1900, 0xA0, 2, 0x4), 75452921: (1600, 0x88, 2, EARTH)}
J5E = {2857636, 75452921}  # Link à effet (cartes de CARDS, mode kn de compare.py)
# monstres pendule normaux : (ATK, DEF, niveau, échelle gauche, échelle droite) ; mode pen de compare.py
PENDULUM = {17390179: (1800, 600, 4, 7, 7), 19474136: (2500, 1000, 5, 2, 2), 28363749: (2500, 1000, 6, 0, 0),
            32872239: (2000, 2000, 6, 9, 9), 83980492: (500, 2100, 4, 7, 7)}
J5F = set(PENDULUM)
EFFECT_MONSTERS = {74131780: (1000, 1000, 2), 11384280: (1400, 1300, 4),  # Exiled Force, Cannon Soldier
                   26202165: (1000, 600, 3), 78010363: (1100, 1200, 4),    # Sangan, Witch of the Black Forest
                   54652250: (450, 600, 2), 83011277: (1400, 1100, 4),     # Man-Eater Bug, Mystic Tomato
                   24508238: (100, 100, 1), 59438930: (0, 1800, 3),        # D.D. Crow, Ghost Ogre & Snow Rabbit
                   18430390: (1800, 500, 4),                               # Winged Rhynos
                   28470714: (600, 700, 2), 38033121: (2000, 1700, 6),     # Bladefly, Dark Magician Girl
                   13314457: (600, 100, 1), 25109950: (800, 1300, 3),      # Deepsea Macrotrema, Little Swordsman of Aile
                   62340868: (2400, 2200, 7), 42364374: (2000, 2000, 3),   # Kazejin, Arsenal Bug
                   59380081: (2000, 1000, 5), 10375182: (1200, 1900, 4),   # Big-Tusked Mammoth, Command Knight
                   52077741: (1400, 1200, 4), 17985575: (1200, 1100, 4),   # Obnoxious Celtic Guard, Lord of D.
                   86188410: (1500, 1600, 4), 72845813: (1700, 1200, 4),   # Wildheart, Ally of Justice Quarantine
                   15960641: (1600, 600, 4), 97268402: (0, 0, 1)}          # Mirage Dragon, Effect Veiler


def test_bank(key, c):
    """c monstres normaux aléatoires (codes 0..c-1) + les cartes de CARDS (indices c..)."""
    import jax
    import core
    st = SPELL_CODES + list(ST_TYPES) + list(J4_SPELLS) + list(J4B_SPELLS) + list(J4C_SPELLS) + list(J4D_SPELLS) + list(J5A_SPELLS) \
        + list(J5B_SPELLS)
    codes = list(range(c)) + st + list(EFFECT_MONSTERS) + list(MATERIAL_STATS) + list(FUSION) + list(RITUAL) \
        + list(TUNER_STATS) + list(SYNCHRO) + list(XYZ) + list(LINK) + list(PENDULUM)
    xs = list(MATERIAL_STATS.values()) + [FUSION_STATS[k] for k in FUSION] + list(RITUAL.values()) \
        + list(TUNER_STATS.values()) + [SYNCHRO_STATS[k] for k in SYNCHRO] + [XYZ_STATS[k] for k in XYZ] \
        + [LINK_STATS[k] for k in LINK] + [v[:3] for v in PENDULUM.values()]
    atk = np.concatenate([np.asarray(jax.random.randint(key, (c,), 0, 30)) * 100, np.zeros(len(st), int),
                          [v[0] for v in EFFECT_MONSTERS.values()], [v[0] for v in xs]])
    df = np.concatenate([np.asarray(jax.random.randint(jax.random.fold_in(key, 1), (c,), 0, 30)) * 100,
                         np.zeros(len(st), int), [v[1] for v in EFFECT_MONSTERS.values()], [v[1] for v in xs]])
    lv = np.concatenate([np.asarray(jax.random.randint(jax.random.fold_in(key, 2), (c,), 1, 13)),
                         np.zeros(len(st), int), [v[2] for v in EFFECT_MONSTERS.values()], [v[2] for v in xs]])
    ctype = [core.TYPE_MONSTER | core.TYPE_NORMAL] * c + [core.TYPE_SPELL] * len(SPELL_CODES) \
        + list(ST_TYPES.values()) + list(J4_SPELLS.values()) + list(J4B_SPELLS.values()) + list(J4C_SPELLS.values()) + list(J4D_SPELLS.values()) + list(J5A_SPELLS.values()) + list(J5B_SPELLS.values()) + [core.TYPE_MONSTER | 0x20] * len(EFFECT_MONSTERS) \
        + [core.TYPE_MONSTER | core.TYPE_NORMAL] * len(MATERIAL_STATS) + [core.TYPE_MONSTER | core.TYPE_FUSION] * len(FUSION) \
        + list(RITUAL_TYPES.values()) + [core.TYPE_MONSTER | core.TYPE_NORMAL | core.TYPE_TUNER] * len(TUNER_STATS) \
        + [core.TYPE_MONSTER | core.TYPE_SYNCHRO] * len(SYNCHRO) + [core.TYPE_MONSTER | core.TYPE_XYZ] * len(XYZ) \
        + [core.TYPE_MONSTER | core.TYPE_LINK] * len(LINK) \
        + [core.TYPE_MONSTER | core.TYPE_NORMAL | core.TYPE_PENDULUM] * len(PENDULUM)
    # attributs aléatoires (les effets continus de Bladefly, Deepsea Macrotrema en dépendent)
    attr = np.concatenate([1 << np.asarray(jax.random.randint(jax.random.fold_in(key, 3), (c,), 0, 6)),
                           np.zeros(len(st), int), [WIND if k in (28470714, 62340868) else WATER for k in EFFECT_MONSTERS],
                           [v[3] if len(v) > 3 else 0 for v in xs]])
    sc = lambda j: [PENDULUM[k][j] if k in PENDULUM else 0 for k in codes]  # échelles
    return core.make_bank(np.array(codes), atk, df, lv, ctype=ctype, attr=attr, lsc=sc(3), rsc=sc(4), **assemble(codes))


def test_decks(key, bank, b, d, c, eff, x):
    """Decks de test [b, 2, d] (eff : part de cartes à effet, 15 % de matériaux de fusion et de Syntoniseurs, le reste
    des monstres normaux aléatoires) et Extra Decks [b, 2, x] de monstres fusion et synchro, pour test_bank(key, c)."""
    import jax
    import jax.numpy as jnp
    code = np.asarray(bank.code)
    extra_codes = set(FUSION) | set(SYNCHRO) | set(XYZ) | set(LINK)
    effs = np.array([i for i in range(c, len(code)) if int(code[i]) not in MATERIAL_STATS and int(code[i]) not in extra_codes
                     and int(code[i]) not in TUNER_STATS])
    mats = np.array([i for i in range(len(code)) if int(code[i]) in MATERIAL_STATS or int(code[i]) in TUNER_STATS])
    fus = np.array([i for i in range(len(code)) if int(code[i]) in extra_codes])
    k = jax.random.split(key, 6)
    u = jax.random.uniform(k[0], (b, 2, d))
    decks = jnp.where(u < eff, jnp.asarray(effs)[jax.random.randint(k[1], (b, 2, d), 0, len(effs))],
                      jnp.where(u < eff + 0.15, jnp.asarray(mats)[jax.random.randint(k[2], (b, 2, d), 0, len(mats))],
                                jax.random.randint(k[3], (b, 2, d), 0, c)))
    extra = jnp.asarray(fus)[jax.random.randint(k[4], (b, 2, x), 0, len(fus))]
    return decks, extra
