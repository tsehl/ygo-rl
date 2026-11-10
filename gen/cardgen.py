"""Générateur procédural de cartes et de decks (jalon 6, étape 1 ; conception dans ETAPE3.md).

Une carte générée est d'abord une spécification (dict) : un cadre (type de carte, stats, recette d'invocation), puis
0 à 2 effets décrits par gabarit : moment d'activation (genre, événements, une fois par tour), conditions, coût,
clauses. Une clause est soit un verbe appliqué à une sélection de cartes (cible à l'activation, toutes les cartes
d'un filtre, choix à la résolution, cartes de l'événement, la carte elle-même), soit une action sans carte (piocher,
dommages, gain de LP, nier une activation). Une deuxième clause suit la première (« puis ») et ne s'applique que si
la première a touché au moins une carte.

La spécification se traduit :
- en briques du moteur (`bricks`, puis `cards.assemble` dans `to_bank`) ;
- plus tard en script Lua, pour vérifier les cartes générées contre ygopro-core (étape 2).
Les gabarits reprennent les formes des vraies cartes vérifiées contre l'oracle (cards.CARDS) ; ce sont leurs
combinaisons qui sont nouvelles.

Budget de puissance (`value`, en « cartes d'avantage ») : grossier, sert seulement de bornes pour écarter les cartes
absurdes. À recalibrer sur les taux de victoire en self-play (ETAPE3.md).

Toutes les cartes générées ont des codes >= CODE0 : elles ne se mélangent jamais aux vraies cartes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

import cards  # noqa: E402
import core  # noqa: E402
from cards import (F, FILTER, SELF, REQ, REQ_FREE, CHOOSE, MOVE, DESTROY, SPSUMMON, DRAW, FIRST,  # noqa: E402
                   DAMAGE, RECOVER, COND_CHAIN, COND_CHAIN_FIELD, CHAINCARD, NEGATE, PAY_LP, EVENT, COND_TURN,
                   COND_EVP, COND_SELF, MODIFY, COND_PHASE, LOCK, MINATK, act, ignition, trigger, quick, cont, ev, ritual_spell,
                   ME, OPP)
from core import (BIG, E, P, PT, NF, NSET, Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED, Z_EXTRA,  # noqa: E402
                  Z_OVERLAY, P_FU_ATK, P_FU_DEF, P_FD_ATK, P_FACEUP, TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL,
                  TYPE_QUICKPLAY, TYPE_CONTINUOUS, TYPE_COUNTER, TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ,
                  TYPE_LINK, TYPE_TUNER, TYPE_PENDULUM, EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON, EV_ATTACK,
                  EV_TO_GRAVE, EV_BATTLE_DESTROYED, EV_FLIP, OPT_NONE, OPT_NAME, OPT_CARD, PH_MAIN1, PH_MAIN2,
                  PH_BATTLE, MC_ATK, MC_DEF, MC_DISABLE, MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP,
                  MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP, MC_INDES_EFF, MC_NO_DIRECT, DMG_STEP, DMG_STAT,
                  MC_FINAL_ATK, MC_BASE_ATK, MV_HALF_BASE_ATK, MV_HALF_ATK, TD_PERM,
                  RIT_GREATER, RIT_EQUAL, ST_SUMMON_TURN, ST_FLIP_TURN, ST_LINKSUM, LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T,
                  LM_TR)

CODE0 = 600_000_000
TYPE_EFFECT, TYPE_FLIP = 0x20, 0x200000
ONFIELD = Z_MZONE | Z_SZONE
ATTRS = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20]          # TERRE, EAU, FEU, VENT, LUMIÈRE, TÉNÈBRES
RACES = [1 << i for i in range(20)]
SETS = [0x7A0 + i for i in range(8)]             # archétypes procéduraux (étape 4 : vrais archétypes structurés)
ARROWS = [LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T, LM_TR]
STAT_BUDGET = {1: 1200, 2: 1900, 3: 2500, 4: 3200, 5: 3700, 6: 4200, 7: 4700, 8: 5200, 9: 5500, 10: 5800,
               11: 6000, 12: 6200}                # ATK + DEF d'un monstre normal de ce niveau (≈ 1000 par « carte »)
LV_W = np.array([5, 8, 10, 24, 8, 8, 8, 9, 4, 4, 1, 1], float)  # niveaux des monstres du deck principal
CARD_CAP = 2.6                                    # valeur maximale d'une carte (magie / piège : avantage net + 1)


def _r100(x):
    return int(max(0, round(x / 100)) * 100)


class Gen:
    """Générateur (numpy, côté hôte) : une graine → une banque de cartes procédurales (`make_set`)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ tirages de base
    def p(self, x):
        return self.rng.random() < x

    def pick(self, xs, w=None):
        w = None if w is None else np.asarray(w, float) / np.sum(w)
        return xs[self.rng.choice(len(xs), p=w)]

    def qual(self, kw, n=1, stats=True, lv=True):
        """Ajoute n qualificatifs au filtre de monstres kw (ATK, attribut, type, niveau, archétype)."""
        for _ in range(n):
            q = self.pick(["atk_lo", "atk_hi", "attr", "race", "lv", "setc"], [2, 3, 3, 2, 2 if lv else 0, 1.5])
            if q == "atk_lo" and stats:
                kw["atk"] = (self.pick([1000, 1500, 1800, 2000, 2400]), BIG)
            elif q == "atk_hi" and stats:
                kw["atk"] = (-BIG, self.pick([500, 1000, 1500, 1800]))
            elif q == "attr":
                kw["attr"] = self.pick(ATTRS)
            elif q == "race":
                kw["race"] = self.pick(RACES)
            elif q == "lv":
                kw["lv"] = self.pick([(-BIG, 4), (5, BIG), (-BIG, 3), (7, BIG)])
            elif q == "setc":
                kw["setc"] = self.pick(SETS)
        return kw

    def mon(self, nq=None, **kw):
        kw.setdefault("typ", TYPE_MONSTER)
        if kw["typ"] == TYPE_MONSTER and self.p(0.15):
            kw["not_typ"] = kw.get("not_typ", 0) | TYPE_EFFECT
        nq = self.pick([0, 1, 1, 2]) if nq is None else nq
        if (kw.get("me", 0) | kw.get("op", 0)) == Z_GRAVE and self.p(0.15):
            kw["reason"] = self.pick([core.REASON_EFFECT, core.REASON_DESTROY, core.REASON_BATTLE,
                                      core.REASON_COST, -(core.REASON_EFFECT | core.REASON_DESTROY)])
        return self.qual(kw, nq)

    # ------------------------------------------------------------------ clauses
    def card_clause(self, ctx):
        """Clause « verbe sur une sélection ». ctx : genre d'effet, événement, zone de la carte à l'activation."""
        kind, evs, zone = ctx["kind"], ctx.get("on", 0), ctx.get("zone")
        opts = ["destroy", "destroy", "banish", "bounce", "tohand", "tohand", "ss", "atk", "disable", "tograve", "indes_battle"]
        verb = self.pick(opts)
        if evs:  # piège / effet rapide sur un événement : le verbe s'applique aux cartes de l'événement
            if self.p(0.75):
                verb = self.pick(["destroy", "destroy", "banish", "bounce", "atk", "disable"])
                f = dict(op=Z_MZONE, pos=P_FACEUP) if not evs & ev(EV_ATTACK) else dict(op=Z_MZONE)
                if verb == "disable":
                    f = dict(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_EFFECT, nodis=1)
                elif verb == "atk":
                    f = dict(op=Z_MZONE, pos=P_FACEUP)
                elif self.p(0.4):
                    f = self.mon(1, **f)
                return dict(verb=verb, sel="event", f=f, k=1, v=self.neg_atk() if verb == "atk" else 0)
        if verb in ("destroy", "banish", "bounce", "tograve"):
            where = self.pick(["opp_mon", "opp_st", "any_mon", "opp_card", "opp_gy"],
                              [4, 2, 2, 1, 1.5 if verb == "banish" else 0])
            f = {"opp_mon": lambda: self.mon(op=Z_MZONE), "any_mon": lambda: self.mon(me=Z_MZONE, op=Z_MZONE, not_self=1),
                 "opp_st": lambda: dict(op=Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP),
                 "opp_card": lambda: dict(op=ONFIELD), "opp_gy": lambda: dict(op=Z_GRAVE)}[where]()
            if verb == "bounce" and where in ("opp_gy", "opp_st"):
                f = self.mon(op=Z_MZONE)
            if where == "opp_mon" and verb != "tograve" and self.p(0.15):  # Fissure : le monstre adverse au plus faible ATK
                return dict(verb=verb, sel="min", f=self.mon(op=Z_MZONE, pos=P_FACEUP), k=1, v=0)
            sel = "target" if self.p(0.6) else "all"
            if sel == "all" and where in ("opp_card", "any_mon") and verb != "destroy":
                sel = "target"
            k = 2 if sel == "target" and self.p(0.15) else 1
            clause = dict(verb=verb, sel=sel, f=f, k=k, v=0)
            if sel == "target" and k > 1 and self.p(0.5):
                clause["first_only"] = True
            return clause
        if verb == "tohand":
            if self.p(0.6):  # recherche dans le deck
                f = self.mon(self.pick([1, 1, 2]), me=Z_DECK) if self.p(0.7) else dict(
                    me=Z_DECK, typ=self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_SPELL | TYPE_TRAP]))
                return dict(verb="tohand", sel="pick", f=f, k=1, v=0, lock=self.p(0.3))  # lock : Sangan
            f = self.mon(me=Z_GRAVE)  # récupération dans le cimetière
            return dict(verb="tohand", sel="target" if self.p(0.6) else "pick", f=f, k=1, v=0)
        if verb == "ss":
            pos = self.pos()
            sp = 1 if pos & P_FU_ATK else 2  # en défense seulement : jamais un Link (IsCanBeSpecialSummoned)
            if zone in (Z_GRAVE, Z_HAND) and self.p(0.5):  # la carte s'invoque elle-même
                return dict(verb="ss", sel="self", f=dict(me=zone, spsum=sp), k=1, v=0, pos=pos)
            src = self.pick(["gy", "gy_any", "deck", "hand", "removed"], [3, 1, 2, 2, 1])
            base = {"gy": dict(me=Z_GRAVE), "gy_any": dict(me=Z_GRAVE, op=Z_GRAVE), "deck": dict(me=Z_DECK),
                    "hand": dict(me=Z_HAND, not_self=1), "removed": dict(me=Z_REMOVED)}[src]
            f = self.mon(self.pick([1, 1, 2]), spsum=sp, **base)
            if src in ("deck", "hand") and "lv" not in f and "atk" not in f:
                f["lv"] = (-BIG, 4)
            sel = "target" if src in ("gy", "gy_any", "removed") and self.p(0.7) else "pick"
            return dict(verb="ss", sel=sel, f=f, k=1, v=0, pos=pos)
        if verb == "atk":
            r = self.rng.random()
            if zone == Z_MZONE and r < 0.4:  # sur soi ; rd : effacé si ses effets sont annulés (Little Swordsman),
                rd = self.p(0.5)              # perm : tant qu'elle reste sur le terrain (Number 17)
                return dict(verb="atk", sel="self", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pos_atk(), rd=rd,
                            perm=rd and self.p(0.4))
            r2 = self.rng.random()
            if r2 < 0.15:  # Castle Walls
                return dict(verb="def", sel="target", f=dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP), k=1,
                            v=self.pos_atk())
            if r2 < 0.27:  # Shrink : ATK de base / 2
                return dict(verb="halve_base", sel="target", f=self.mon(1, op=Z_MZONE, pos=P_FACEUP), k=1, v=0)
            if r2 < 0.35:  # Reverse Glasses : ATK / 2 (tous les monstres adverses, ou de tout le terrain)
                f = dict(op=Z_MZONE, pos=P_FACEUP) if self.p(0.6) else dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP)
                return dict(verb="halve", sel="all", f=f, k=1, v=0)
            if r < 0.6:
                if self.p(0.5):
                    return dict(verb="atk", sel="target", f=dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP), k=1,
                                v=self.pos_atk() if self.p(0.6) else self.neg_atk())
                return dict(verb="atk", sel="target", f=self.mon(1, op=Z_MZONE, pos=P_FACEUP), k=1, v=self.neg_atk())
            if self.p(0.6):
                return dict(verb="atk", sel="all", f=self.mon(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pos_atk())
            return dict(verb="atk", sel="all", f=self.mon(op=Z_MZONE, pos=P_FACEUP), k=1, v=self.neg_atk())
        if verb == "indes_battle":
            return dict(verb=verb, sel="target", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=0)
        if verb == "disable":
            return dict(verb="disable", sel="target", f=dict(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_EFFECT, nodis=1),
                        k=1, v=0)
        # tograve depuis le deck (meule)
        return dict(verb="tograve", sel="pick", f=self.mon(me=Z_DECK), k=self.pick([1, 1, 2]), v=0)

    def pos(self):
        return self.pick([P_FU_ATK | P_FU_DEF, P_FU_ATK | P_FU_DEF, P_FU_DEF, P_FU_ATK])

    def pos_atk(self):
        return self.pick([300, 500, 500, 700, 800, 1000, 1500])

    def neg_atk(self):
        return -self.pick([300, 500, 700, 1000, 1500])

    def plain_clause(self):
        """Clause sans sélection de carte."""
        verb = self.pick(["draw", "damage", "recover", "burn_per"], [4, 3, 1.5, 1])
        if verb == "draw":
            return dict(verb="draw", n=self.pick([1, 1, 1, 2]))
        if verb == "damage":
            return dict(verb="damage", v=self.pick([300, 500, 800, 1000, 1500]))
        if verb == "recover":
            return dict(verb="recover", v=self.pick([500, 1000, 1500, 2000]))
        f = self.pick([dict(op=Z_MZONE), dict(op=Z_GRAVE, typ=TYPE_MONSTER), dict(me=Z_GRAVE, typ=TYPE_MONSTER),
                       dict(op=ONFIELD)])
        return dict(verb="burn_per", f=f, v=self.pick([100, 200, 300, 500]))

    def then_clause(self):
        """Seconde clause (« puis ») : sans cible, résolue sans vérification à l'activation (sauf la pioche)."""
        r = self.pick(["draw", "damage", "recover", "atk_all", "mill"], [3, 3, 1, 1.5, 1])
        if r == "draw":
            return dict(verb="draw", n=1)
        if r == "damage":
            return dict(verb="damage", v=self.pick([300, 500, 800]))
        if r == "recover":
            return dict(verb="recover", v=self.pick([500, 1000]))
        if r == "atk_all":
            return dict(verb="atk", sel="all", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pick([300, 500]))
        return dict(verb="tograve", sel="pick", f=self.mon(1, me=Z_DECK), k=1, v=0)

    def clauses(self, ctx):
        c1 = self.card_clause(ctx) if self.p(0.75) else self.plain_clause()
        out = [c1]
        if c1["verb"] == "ss" and self.p(0.5):
            c1["tag"] = self.pick([1, 2, 3, 7, 15, 31, 63, 127, 255])
        if self.p(0.25) and c1["verb"] not in ("burn_per",):
            out.append(self.then_clause())
        return out

    # ------------------------------------------------------------------ coûts et effets
    def cost(self, choices, w):
        c = self.pick(choices, w)
        if c is None:
            return None
        if c == "lp":
            return ("lp", self.pick([500, 800, 1000, 1500, 2000]))
        if c == "detach":
            return ("detach", 1)
        return (c,)

    def effect(self, kind, card):
        """Un effet de genre kind pour la carte card (spécification, sans briques)."""
        mon = card["ctype"] & TYPE_MONSTER
        e = dict(kind=kind, on=0, opt=OPT_NONE, optional=False, rng=0, dmg=0, conds=[], cost=None, clauses=[])
        if kind in ("act", "ignition", "quick") and self.p(0.12):
            e["related"] = True  # l’effet dépend encore de sa source à la résolution
        if kind == "act":  # magie / piège
            ct = card["ctype"]
            if ct & TYPE_COUNTER:
                typ = self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_MONSTER, TYPE_SPELL | TYPE_TRAP])
                e["cost"] = self.cost([None, "lp", "discard"], [2, 2, 1])
                if typ == TYPE_MONSTER:
                    e["conds"] = [("chain_field",)]
                    e["clauses"] = [dict(verb="chain_destroy")]
                else:
                    e["conds"] = [("chain", typ)]
                    e["clauses"] = [dict(verb="negate")]
                return e
            if ct & TYPE_TRAP and self.p(0.35):
                if self.p(0.6):
                    e["on"] = self.pick([ev(EV_SUMMON, EV_FLIP_SUMMON), ev(EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON),
                                         ev(EV_SPSUMMON)])
                    e["conds"] = [("evp", e["on"], OPP)]
                else:
                    e["on"] = ev(EV_ATTACK)
                    e["conds"] = [("turn", OPP)]
            e["clauses"] = self.clauses(dict(kind="act", on=e["on"]))
            if e["on"] == ev(EV_ATTACK) and e["clauses"][0].get("sel") != "event" and self.p(0.5):
                e["clauses"][0] = dict(verb="destroy", sel="all", f=dict(op=Z_MZONE, pos=P_FU_ATK | P_FD_ATK), k=1, v=0)
            e["cost"] = self.cost([None, "lp", "discard"], [6, 2, 1])
            c1 = e["clauses"][0]  # modificateur de stats : activable pendant le damage step (Shrink, Reverse Glasses)
            if (ct & (TYPE_QUICKPLAY | TYPE_TRAP)) and len(e["clauses"]) == 1 and self.p(0.5) and \
                    c1["verb"] in ("atk", "def", "halve_base", "halve") and c1["sel"] in ("target", "all"):
                e["dmg"] = DMG_STEP | DMG_STAT
            return e
        if kind == "hand_proc":
            e["cost"] = self.cost([None, "lp", "discard", "banish_gy"], [4, 1, 2, 1])
            e["rng"] = Z_HAND
            e["opt"] = self.pick([OPT_NONE, OPT_NAME])
            mode = self.pick(["own_empty", "op_empty", "cyber", "ally"])
            if mode == "ally":
                f = dict(me=Z_MZONE, pos=P_FACEUP)
                self.qual(f, stats=False)
                e["counts"] = [(f, 1, BIG)]
            else:
                e["counts"] = [(dict(**({"op": Z_MZONE} if mode == "op_empty" else {"me": Z_MZONE})),0,0)]
                if mode == "cyber":
                    e["counts"].append((dict(op=Z_MZONE),1,BIG))
            return e
        if kind == "ignition":
            e["cost"] = self.cost([None, "lp", "discard", "self", "tribute", "banish_gy"] +
                                  (["detach"] if card["ctype"] & TYPE_XYZ else []),
                                  [4, 1.5, 1.5, 1, 1, 1] + ([4] if card["ctype"] & TYPE_XYZ else []))
            # effet répétable sans coût qui consomme une carte : une fois par tour
            if e["cost"] is None or e["cost"][0] == "lp" or self.p(0.5):
                e["opt"] = self.pick([OPT_NAME, OPT_NAME, OPT_CARD])
            e["clauses"] = self.clauses(dict(kind=kind, zone=None if e["cost"] == ("self",) else Z_MZONE))
            return e
        if kind == "trigger":
            trig = self.pick(["summon", "spsummon", "flip", "to_gy_field", "to_gy", "battle_destroyed"],
                             [3, 1.5, 1.5 if not card.get("extra") else 0, 2, 1, 1.5])
            e["on"], zone = {"summon": (ev(EV_SUMMON), Z_MZONE), "spsummon": (ev(EV_SPSUMMON), Z_MZONE),
                             "flip": (ev(EV_FLIP), Z_MZONE), "to_gy_field": (ev(EV_TO_GRAVE), Z_GRAVE),
                             "to_gy": (ev(EV_TO_GRAVE), Z_GRAVE),
                             "battle_destroyed": (ev(EV_BATTLE_DESTROYED), Z_GRAVE)}[trig]
            if trig == "flip":
                card["flip"] = True
            if trig == "to_gy_field":
                e["conds"] = [("self", 0, ONFIELD, 0)]
            elif trig == "battle_destroyed":
                e["conds"] = [("self", Z_GRAVE, 0, 0)]
            elif trig == "spsummon" and self.p(0.5):
                typ = core.SUMMON_SPECIAL
                for flag, value in ((TYPE_FUSION, core.SUMMON_FUSION), (TYPE_RITUAL, core.SUMMON_RITUAL),
                                    (TYPE_SYNCHRO, core.SUMMON_SYNCHRO), (TYPE_XYZ, core.SUMMON_XYZ),
                                    (TYPE_LINK, core.SUMMON_LINK)):
                    if card["ctype"] & flag:
                        typ = value
                e["conds"] = [("self", 0, 0, core.summon_mask(typ))]
            if trig in ("to_gy_field", "to_gy", "battle_destroyed") and self.p(0.35):
                mask = self.pick([core.REASON_EFFECT, core.REASON_DESTROY, core.REASON_BATTLE,
                                  core.REASON_COST, core.REASON_COST | core.REASON_EFFECT])
                e["conds"].append(("self", 0, 0, -((1 << 30) | mask)))
            e["optional"] = self.p(0.6)
            e["opt"] = OPT_NAME if self.p(0.5) else OPT_NONE
            e["cost"] = self.cost([None, "lp"], [6, 1]) if e["optional"] else None
            e["clauses"] = self.clauses(dict(kind=kind, zone=zone))
            return e
        if kind == "quick":
            hand = mon and not card.get("extra") and self.p(0.5)
            e["rng"] = Z_HAND if hand else Z_MZONE
            if hand:
                e["cost"] = ("self",)
            else:
                e["opt"] = self.pick([OPT_NAME, OPT_NAME, OPT_CARD])
                e["cost"] = self.cost([None, "lp", "discard", "tribute"] + (["detach"] if card["ctype"] & TYPE_XYZ else []),
                                      [4, 1, 1, 0.5] + ([3] if card["ctype"] & TYPE_XYZ else []))
            if self.p(0.5):
                e["conds"].append(("turn", OPP))
            if self.p(0.2):
                e["conds"].append(("phase", PH_MAIN1 | PH_MAIN2))
            if self.p(0.15):  # contre-effet : nier une activation de magie / piège, ou détruire la carte d'un effet
                if self.p(0.5):
                    e["conds"] = [("chain", self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_SPELL | TYPE_TRAP]))]
                    e["clauses"] = [dict(verb="negate")]
                else:
                    e["conds"] = [("chain_field",)]
                    e["clauses"] = [dict(verb="chain_destroy")]
                return e
            e["clauses"] = self.clauses(dict(kind=kind, zone=None if hand else Z_MZONE))
            return e
        if kind == "cont":
            e["rng"] = Z_MZONE if mon else Z_SZONE
            e["cont"] = self.cont_spec(mon)
            return e
        raise ValueError(kind)

    def cont_spec(self, mon):
        """Effet continu : modificateur de stats, interdiction ou protection."""
        r = self.pick(["atk_field", "atk_self", "debuff", "no_attack", "no_btarget", "indes_battle", "no_etarget",
                       "imm_trap", "no_ss", "p_no_trap", "disable", "indes_eff", "no_direct"],
                      [4, 2 if mon else 0, 2, 1.5, 1 if mon else 0, 1 if mon else 0, 1, 1 if mon else 0, 1, 0.7,
                       0.5 if not mon else 0, 1, 0.5])
        c = dict(code=MC_ATK, val=0, tgt=None, per=None, cnt=None, tp=0, ph=0)
        # modificateurs de stats : pas de filtre sur l'ATK / la DEF des cibles (ygopro évalue la cible pendant le calcul
        # de l'ATK de la carte, GetAttack y rend la valeur partielle temp.attack : pas modélisé par le moteur)
        if r == "atk_field":
            c.update(code=self.pick([MC_ATK, MC_ATK, MC_DEF]), val=self.pick([200, 300, 400, 500]),
                     tgt=self.qual(dict(me=Z_MZONE, typ=TYPE_MONSTER), 1, stats=False))
            if self.p(0.2):
                c["tp"], c["ph"] = ME + 1, PH_BATTLE
        elif r == "atk_self":
            c.update(code=MC_ATK, val=self.pick([100, 200, 300, 500]))
            if self.p(0.5):
                c["per"] = self.pick([dict(me=Z_GRAVE, typ=TYPE_MONSTER), dict(me=Z_MZONE, not_self=1),
                                      self.mon(1, me=Z_GRAVE, op=Z_GRAVE)])
            else:
                c["val"] = self.pick([500, 800, 1000])
                c["cnt"] = (self.pick([dict(me=Z_MZONE, not_self=1), dict(op=Z_MZONE)]), self.pick([1, 2]), BIG)
        elif r == "debuff":
            c.update(code=self.pick([MC_ATK, MC_DEF]), val=-self.pick([200, 300, 500]),
                     tgt=self.qual(dict(op=Z_MZONE, typ=TYPE_MONSTER), 1, stats=False))
        elif r == "no_attack":
            c.update(code=MC_NO_ATTACK, tgt=self.pick([dict(op=Z_MZONE, lv=(self.pick([4, 5, 7]), BIG)),
                                                      dict(op=Z_MZONE, st=ST_SUMMON_TURN | ST_FLIP_TURN),
                                                      dict(op=Z_MZONE, atk=(self.pick([1500, 2000]), BIG))]))
        elif r == "no_btarget":
            c.update(code=MC_NO_BTARGET, cnt=(dict(me=Z_MZONE), 2, BIG))
        elif r == "indes_battle":
            c.update(code=MC_INDES_BATTLE, val=self.pick([0, 1900, 2000]))
        elif r == "no_etarget":
            c.update(code=MC_NO_ETARGET, tgt=self.mon(1, me=Z_MZONE))
        elif r == "imm_trap":
            c.update(code=MC_IMM_TRAP)
        elif r == "no_ss":
            c.update(code=MC_NO_SS, tgt=self.qual(dict(me=Z_DECK | Z_HAND | Z_GRAVE | Z_REMOVED | Z_EXTRA,
                                                       op=Z_DECK | Z_HAND | Z_GRAVE | Z_REMOVED | Z_EXTRA),
                                                  1, stats=False, lv=False))
        elif r == "p_no_trap":
            c.update(code=MC_P_NO_TRAP, val=2, ph=PH_BATTLE)
        elif r == "disable":
            c.update(code=MC_DISABLE, tgt=dict(op=Z_MZONE, typ=TYPE_EFFECT))
        elif r == "indes_eff":
            c.update(code=MC_INDES_EFF, tgt=None if mon and self.p(0.5) else self.mon(1, me=Z_MZONE))
        elif r == "no_direct":
            c.update(code=MC_NO_DIRECT, tgt=dict(op=Z_MZONE))
        return c

    # ------------------------------------------------------------------ cartes
    def stats(self, lv, val):
        tot = max(0, STAT_BUDGET[lv] - 900 * val)
        a = self.rng.uniform(0.25, 0.8)
        return _r100(tot * a), _r100(tot * (1 - a))

    def base(self, ctype, lv=0):
        return dict(ctype=ctype, atk=0, df=0, lv=lv, attr=0, race=0, setc=[], lsc=0, rsc=0, effects=[],
                    recipe=None, flip=False, extra=False)

    def monster_frame(self, extra=False, lv=None):
        lv = int(self.rng.choice(12, p=LV_W / LV_W.sum())) + 1 if lv is None else lv
        c = self.base(TYPE_MONSTER, lv)
        c.update(attr=self.pick(ATTRS), race=self.pick(RACES), extra=extra)
        if self.p(0.3):
            c["setc"] = [self.pick(SETS)]
        return c

    def with_effects(self, c, kinds_w, n=None):
        """Ajoute 1 à 2 effets (genres tirés selon kinds_w), en respectant le budget de la carte."""
        n = (1 if self.p(0.65) else 2) if n is None else n
        kinds = list(kinds_w)
        for _ in range(20):
            effs, used = [], []
            for _ in range(n):
                # au plus un déclencheur par carte : le moteur ne garde qu'un effet déclenché en attente par carte
                # (tg / tg_e), ygopro les propose tous (deux événements simultanés : cimetière + détruit au combat)
                k = self.pick(kinds, [kinds_w[x] * (0 if x in ("trigger", "hand_proc") and x in used else 0.3 if x in used else 1)
                                      for x in kinds])
                used.append(k)
                effs.append(self.effect(k, c))
            v = sum(value(e) for e in effs)
            if v <= CARD_CAP and all(fits(e) for e in effs):
                c["effects"] = effs
                return c, v
        c["effects"] = []
        return c, 0.0

    def effect_monster(self, lv=None):
        c = self.monster_frame(lv=lv)
        c["ctype"] |= TYPE_EFFECT
        c, v = self.with_effects(c, dict(ignition=3, trigger=4, quick=2, cont=2, hand_proc=1))
        c["atk"], c["df"] = self.stats(c["lv"], v)
        if not c["effects"]:
            c["ctype"] = (c["ctype"] & ~TYPE_EFFECT) | TYPE_NORMAL
        return c

    def normal_monster(self, tuner=False, pend=False):
        c = self.monster_frame(lv=self.pick([1, 2, 3, 4]) if tuner else None)
        c["ctype"] |= TYPE_NORMAL | (TYPE_TUNER if tuner else 0) | (TYPE_PENDULUM if pend else 0)
        c["atk"], c["df"] = self.stats(c["lv"], 0.4 if tuner else 0.0)
        if pend:
            c["lsc"] = c["rsc"] = int(self.rng.integers(0, 11))
            c["effects"] = [dict(kind="pend")]
        return c

    def spell(self, sub):
        ct = TYPE_SPELL | {"normal": 0, "quick": TYPE_QUICKPLAY, "cont": TYPE_CONTINUOUS}[sub]
        c = self.base(ct)
        if sub == "cont":
            for _ in range(20):
                e = self.effect("cont", c)
                if value(e) <= CARD_CAP:
                    break
            c["effects"] = [dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], cost=None,
                                 clauses=[]), e]
            return c
        return self.with_effects(c, dict(act=1), n=1)[0]

    def trap(self, sub):
        ct = TYPE_TRAP | {"normal": 0, "cont": TYPE_CONTINUOUS, "counter": TYPE_COUNTER}[sub]
        c = self.base(ct)
        if sub == "cont":
            if self.p(0.25):
                for _ in range(20):
                    e = dict(kind="quick", on=0, opt=self.pick([OPT_NAME, OPT_CARD]), optional=False,
                             rng=Z_SZONE, dmg=0, conds=[],
                             cost=self.cost([None, "lp", "discard", "tribute"], [4, 1, 1, 1]),
                             clauses=self.clauses(dict(kind="quick", zone=Z_SZONE)))
                    e['refund'] = e['opt'] == OPT_NAME and self.p(0.5)
                    if fits(e) and value(e) <= CARD_CAP:
                        break
                a = dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0,
                         conds=[], cost=None, clauses=[], inline=2)
                c["effects"] = [a, e]
                return c
            e = self.effect("cont", c)
            a = dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], clauses=[],
                     cost=("lp", 1000) if e["cont"]["code"] == MC_DISABLE else None)
            c["effects"] = [a, e]
            return c
        return self.with_effects(c, dict(act=1), n=1)[0]

    def extra_monster(self, kind, main):
        """Monstre de l'Extra Deck ; main : monstres du deck principal (codes nommés par les recettes de fusion)."""
        mats = lambda: self.qual(dict(), 1, stats=False, lv=False) if self.p(0.4) else dict()
        if kind == "fusion":
            lv = self.pick([4, 5, 6, 7, 8, 9, 10])
        elif kind == "synchro":
            lv = self.pick([3, 4, 5, 6, 7, 8, 9, 10])
        elif kind == "xyz":
            lv = self.pick([2, 3, 4, 4, 4, 5, 6, 7, 8])
        else:
            lv = self.pick([1, 2, 2, 3, 3, 4])
        c = self.monster_frame(extra=True, lv=lv)
        c["ctype"] |= {"fusion": TYPE_FUSION, "synchro": TYPE_SYNCHRO, "xyz": TYPE_XYZ, "link": TYPE_LINK}[kind]
        if kind == "fusion":
            n = self.pick([2, 2, 2, 3])
            rec = []
            for _ in range(n):
                if self.p(0.35) and main:
                    rec.append(dict(code=self.pick(main)))
                else:
                    rec.append(self.qual(dict(), 1, stats=False))
            c["recipe"] = ("fusion", rec)
        elif kind == "synchro":
            c["recipe"] = ("synchro", (mats(), mats()))
        elif kind == "xyz":
            c["recipe"] = ("xyz", (mats(), self.pick([2, 2, 2, 3])))
        else:
            arrows = self.rng.choice(8, size=lv, replace=False)
            c["df"] = int(sum(ARROWS[i] for i in arrows))
            c["recipe"] = ("link", (mats(), 1 if lv == 1 else 2, max(2, lv) if lv > 1 else 1, 0))
        v = 0.0
        if self.p(0.7):
            c["ctype"] |= TYPE_EFFECT
            c, v = self.with_effects(c, dict(ignition=3, trigger=3, quick=2, cont=2))
        bonus = {"fusion": 400, "synchro": 500, "xyz": 500, "link": 600}[kind]
        tot = STAT_BUDGET[max(1, min(12, lv if kind != "link" else 2 * lv + 1))] + bonus - 900 * v
        if kind == "link":
            c["atk"] = _r100(max(0, tot * self.rng.uniform(0.4, 0.55)))
        else:
            c["atk"], c["df"] = self.stats(lv, 0)
            s = (tot / max(1, c["atk"] + c["df"]))
            c["atk"], c["df"] = _r100(c["atk"] * s), _r100(c["df"] * s)
        return c

    def ritual_pair(self):
        """Monstre rituel + son sort rituel (Greater ou Equal) ; le code du monstre est fixé par make_set."""
        m = self.monster_frame(lv=self.pick([4, 5, 6, 7, 8, 8, 10]))
        m["ctype"] |= TYPE_RITUAL
        v = 0.0
        if self.p(0.6):
            m["ctype"] |= TYPE_EFFECT
            m, v = self.with_effects(m, dict(ignition=3, trigger=3, quick=2, cont=2))
        m["atk"], m["df"] = self.stats(m["lv"], v - 0.6)
        s = self.base(TYPE_SPELL | TYPE_RITUAL)
        s["effects"] = [dict(kind="ritual", mode=self.pick([RIT_GREATER, RIT_EQUAL]))]
        m["recipe"] = ("ritual", None)
        return m, s

    def fusion_spell(self):
        s = self.base(TYPE_SPELL)
        f = dict(me=Z_EXTRA, fus=1)
        if self.p(0.3):
            f["setc"] = self.pick(SETS)
        s["effects"] = [dict(kind="fusion_spell", f=f)]
        return s

    def make_set(self, n_main=400, n_extra=80):
        """Banque procédurale : n_main cartes du deck principal (index 0..), puis n_extra de l'Extra Deck."""
        main = []
        kinds = ["normal", "tuner", "pend", "effect", "spell", "quick", "cspell", "fusion_spell", "ritual",
                 "trap", "ctrap", "counter"]
        w = [14, 4, 2, 32, 12, 5, 3, 2, 2, 12, 3, 2]
        while len(main) < n_main:
            k = self.pick(kinds, w)
            if k == "normal":
                main.append(self.normal_monster())
            elif k == "tuner":
                main.append(self.effect_monster(lv=self.pick([1, 2, 3, 4])) if self.p(0.5) else self.normal_monster(True))
                main[-1]["ctype"] |= TYPE_TUNER
            elif k == "pend":
                main.append(self.normal_monster(pend=True))
            elif k == "effect":
                main.append(self.effect_monster())
            elif k in ("spell", "quick", "cspell"):
                main.append(self.spell({"spell": "normal", "quick": "quick", "cspell": "cont"}[k]))
            elif k == "fusion_spell":
                main.append(self.fusion_spell())
            elif k == "ritual" and len(main) + 2 <= n_main:
                m, s = self.ritual_pair()
                main += [m, s]
            elif k in ("trap", "ctrap", "counter"):
                main.append(self.trap({"trap": "normal", "ctrap": "cont", "counter": "counter"}[k]))
        for i, c in enumerate(main):
            c["code"] = CODE0 + i
        for i, c in enumerate(main):  # sort rituel : code du monstre rituel qui le précède
            if c["effects"] and c["effects"][0]["kind"] == "ritual":
                c["effects"][0]["code"] = main[i - 1]["code"]
        mon_codes = [c["code"] for c in main if c["ctype"] & TYPE_MONSTER and not c["ctype"] & TYPE_RITUAL]
        extra = [self.extra_monster(self.pick(["fusion", "synchro", "xyz", "link"]), mon_codes) for _ in range(n_extra)]
        for i, c in enumerate(extra):
            c["code"] = CODE0 + n_main + i
        return main + extra


# ---------------------------------------------------------------------- valeur (budget grossier)

def clause_value(c):
    v, k = c["verb"], c.get("k", 1)
    sel = c.get("sel")
    if v == "draw":
        return 1.0 * c["n"]
    if v == "damage":
        return c["v"] / 1500
    if v == "recover":
        return c["v"] / 3000
    if v == "burn_per":
        return 2.5 * c["v"] / 1500
    if v in ("negate", "chain_destroy"):
        return 1.3
    mult = {"target": 1.0, "pick": 1.0, "event": 0.9, "self": 0.8, "all": 1.8, "min": 0.9}[sel]
    base = {"destroy": 1.0, "banish": 1.15, "bounce": 0.8, "tograve": 0.9, "tohand": 0.9, "ss": 1.0,
            "atk": abs(c.get("v", 0)) / 1500 * (1.6 if c.get("perm") else 1.0), "def": c.get("v", 0) / 3000,
            "halve_base": 0.6, "halve": 0.5, "disable": 0.7, "indes_battle": 0.6}[v]
    if c.get("lock"):
        base -= 0.1
    if v == "tograve" and c["f"].get("me") == Z_DECK:
        base = 0.3
    if v in ("tohand", "ss") and sel == "all":
        mult = 1.0
    return base * mult * k


def cost_value(cost):
    if cost is None:
        return 0.0
    return {"lp": cost[1] / 2500 if cost[0] == "lp" else 0, "discard": 0.9, "self": 0.6, "tribute": 0.9,
            "banish_gy": 0.3, "detach": 0.4}[cost[0]]


def value(e):
    """Valeur grossière d'un effet, en cartes d'avantage (avant la valeur de la carte elle-même)."""
    k = e["kind"]
    if k == "hand_proc":
        return 0.7
    if k == "cont":
        c = e["cont"]
        if c["code"] in (MC_ATK, MC_DEF):
            return abs(c["val"]) / 1000 * (2.0 if c["tgt"] is not None else 1.0) * (2.5 if c["per"] else 1.0)
        return {MC_DISABLE: 1.8, MC_NO_ATTACK: 1.0, MC_NO_SS: 0.8, MC_P_NO_TRAP: 0.6}.get(c["code"], 0.6)
    if k in ("pend", "ritual", "fusion_spell"):
        return 0.0
    v = sum(clause_value(c) for c in e["clauses"])
    speed = {"act": 1.0, "ignition": 1.0, "trigger": 0.9, "quick": 1.25}[k]
    if k == "act" and e["on"]:
        speed = 0.9
    if e["opt"] == OPT_NONE and k in ("ignition", "quick") and e["cost"] not in (("self",),):
        speed *= 1.3  # répétable
    return max(0.0, v * speed - cost_value(e["cost"]) - 0.15 * len(e["conds"]))


# ---------------------------------------------------------------------- traduction en briques

def _F(kw):
    return F(**kw)


COST_REGS = (2, 3)


def cost_bricks(cost):
    a, b = COST_REGS
    if cost is None:
        return []
    t = cost[0]
    if t == "lp":
        return [PAY_LP(cost[1])]
    if t == "discard":
        return [FILTER(a, F(me=Z_HAND, not_self=1)), CHOOSE(b, a, 1), MOVE(b, reason=core.REASON_COST | core.REASON_DISCARD)]
    if t == "self":
        return [SELF(a), MOVE(a, reason=core.REASON_COST, release_if_field=True)]
    if t == "tribute":
        return [FILTER(a, F(me=Z_MZONE, not_self=1)), CHOOSE(b, a, 1), MOVE(b, reason=core.REASON_COST | core.REASON_RELEASE)]
    if t == "banish_gy":
        return [FILTER(a, F(me=Z_GRAVE, typ=TYPE_MONSTER)), CHOOSE(b, a, 1), MOVE(b, Z_REMOVED, reason=core.REASON_COST)]
    if t == "detach":
        return [FILTER(a, F(me=Z_OVERLAY)), CHOOSE(b, a, cost[1]), MOVE(b, reason=core.REASON_COST)]
    raise ValueError(cost)


def cond_bricks(conds):
    out = []
    for c in conds:
        if c[0] == "turn":
            out.append(COND_TURN(c[1]))
        elif c[0] == "phase":
            out.append(COND_PHASE(c[1]))
        elif c[0] == "self":
            out.append(COND_SELF(zone=c[1], prev=c[2], st=c[3]))
        elif c[0] == "evp":
            out.append(COND_EVP(c[1], c[2]))
        elif c[0] == "chain":
            out.append(COND_CHAIN(c[1]))
        elif c[0] == "chain_field":
            out.append(COND_CHAIN_FIELD())
        else:
            raise ValueError(c)
    return out


def verb_bricks(c, r):
    v = c["verb"]
    if v == "destroy":
        return [DESTROY(r)]
    if v == "banish":
        return [MOVE(r, Z_REMOVED)]
    if v in ("bounce", "tohand"):
        return [MOVE(r, Z_HAND)] + ([LOCK(r)] if c.get("lock") else [])
    if v == "tograve":
        return [MOVE(r, Z_GRAVE)]
    if v == "ss":
        return [SPSUMMON(r, c["pos"], tag=c.get("tag", 0))]
    if v == "atk":
        return [MODIFY(r, MC_ATK, c["v"], dur=TD_PERM if c.get("perm") else core.TD_TURN, rd=bool(c.get("rd")))]
    if v == "def":
        return [MODIFY(r, MC_DEF, c["v"])]
    if v == "halve_base":
        return [MODIFY(r, MC_BASE_ATK, mode=MV_HALF_BASE_ATK)]
    if v == "halve":
        return [MODIFY(r, MC_FINAL_ATK, mode=MV_HALF_ATK)]
    if v == "disable":
        return [MODIFY(r, MC_DISABLE)]
    if v == "indes_battle":
        return [MODIFY(r, MC_INDES_BATTLE, 0)]
    raise ValueError(v)


def clause_bricks(c, on, first):
    """(cible, résolution, registre du résultat ou None) d'une clause. La première clause peut cibler (registres 0-2 ;
    la cible garde le registre 1 jusqu'à la résolution) ; la seconde n'utilise que le registre 3 à l'activation."""
    v = c["verb"]
    if v == "draw":
        rr = 0 if first else 3
        return [FILTER(rr, F(me=Z_DECK)), REQ(rr, c["n"])], [DRAW(ME, c["n"])], None
    if v == "damage":
        return [], [DAMAGE(OPP, c["v"])], None
    if v == "recover":
        return [], [RECOVER(ME, c["v"])], None
    if v == "burn_per":
        return [FILTER(0, _F(c["f"])), REQ(0, 1)], [FILTER(0, _F(c["f"])), DAMAGE(OPP, c["v"], per=0)], 0
    if v == "negate":
        return [], [NEGATE(), CHAINCARD(0), DESTROY(0)], None
    if v == "chain_destroy":
        return [CHAINCARD(1, ONFIELD)], [DESTROY(1)], 1
    f, sel, k = c["f"], c["sel"], c["k"]
    ss = v == "ss"
    pre_t = [REQ_FREE(ME, Z_MZONE, 1)] if ss else []
    pre_r = [REQ_FREE(ME, Z_MZONE, 1)] if ss else []
    if not first:  # « puis » : sélection à la résolution seulement
        assert sel in ("all", "pick")
        if sel == "all":
            return [], [FILTER(0, _F(f))] + verb_bricks(c, 0), 0
        return [], pre_r + [FILTER(0, _F(f)), CHOOSE(1, 0, k)] + verb_bricks(c, 1), 1
    if sel == "target":
        first = [FIRST(2, 1, original=True), REQ(2, 1, related=True)] if c.get("first_only") else []
        source = 2 if c.get("first_only") else 1
        return (pre_t + [FILTER(0, _F(f), tgt=True), CHOOSE(1, 0, k, tgt=True)],
                pre_r + first + [FILTER(2, _F(f), within=source)] + verb_bricks(c, 2), 2)
    if sel == "all":
        return [FILTER(0, _F(f)), REQ(0, 1)], [FILTER(2, _F(f))] + verb_bricks(c, 2), 2
    if sel == "pick":
        return (pre_t + [FILTER(0, _F(f)), REQ(0, k)],
                pre_r + [FILTER(2, _F(f)), CHOOSE(1, 2, k)] + verb_bricks(c, 1), 1)
    if sel == "min":  # Fissure
        return ([FILTER(0, _F(f)), REQ(0, 1)],
                [FILTER(0, _F(f)), REQ(0, 1), MINATK(1, 0), CHOOSE(2, 1, 1)] + verb_bricks(c, 2), 2)
    if sel == "event":
        return ([EVENT(0, on), FILTER(1, _F(f), within=0, tgt=True), REQ(1, 1), cards.LABEL(1, target=True)],
                [FILTER(2, _F(f), within=1)] + verb_bricks(c, 2), 2)
    if sel == "self":
        return (pre_t + [SELF(0), FILTER(1, _F(f), within=0), REQ(1, 1)],
                pre_r + [SELF(0), FILTER(2, _F(f), within=0)] + verb_bricks(c, 2), 2)
    raise ValueError(sel)


def bricks(e):
    """Effet (spécification) → tuple d'effet de cards.py."""
    k = e["kind"]
    if k == "hand_proc":
        program = [REQ_FREE(ME,Z_MZONE,1)]
        for r,(f,mn,mx) in enumerate(e["counts"]):
            program.append(FILTER(r,_F(f)))
            if mn > 0: program.append(REQ(r,mn))
            if mx < BIG: program.append(REQ(r,mx,maximum=True))
        return (core.K_PROC,0,e["opt"],Z_HAND,program,cost_bricks(e.get("cost")),[],
                {"refund": int(e["opt"] == OPT_NAME)})
    if k == "pend":
        return act()
    if k == "cont":
        c = e["cont"]
        return cont(e["rng"], c["code"], c["val"], tgt=None if c["tgt"] is None else _F(c["tgt"]),
                    per=None if c["per"] is None else _F(c["per"]),
                    cnt=None if c["cnt"] is None else (_F(c["cnt"][0]), c["cnt"][1], c["cnt"][2]), tp=c["tp"], ph=c["ph"])
    if k == "fusion_spell":
        f = _F(e["f"])
        return act(target=[FILTER(0, f), REQ(0, 1)],
                   resolve=[FILTER(0, f), CHOOSE(1, 0, 1), cards.FUSMAT(2, 1), MOVE(2, reason=core.REASON_EFFECT | core.REASON_MATERIAL | core.REASON_FUSION), SPSUMMON(1, P_FU_ATK | P_FU_DEF, fusion=True)])
    if k == "ritual":
        return ritual_spell(e["code"], e["mode"])[0]
    cost = cond_bricks(e["conds"]) + cost_bricks(e["cost"])
    target, resolve = [], [SELF(0, related=True), REQ(0, 1)] if e.get("related") else []
    first = True
    for c in e["clauses"]:
        t, r, out = clause_bricks(c, e["on"], first)
        target += t
        resolve += r
        if first and out is not None and len(e["clauses"]) > 1:
            resolve.append(REQ(out, 1))  # « puis » : seulement si la première clause a touché une carte
        first = False
    if k == "act":
        result = act(cost=cost, target=target, resolve=resolve, on=e["on"], dmg=e["dmg"], opt=e["opt"])
        if e.get("inline"):
            result = (*result[:7], dict(result[7], inline=e["inline"]))
    elif k == "ignition":
        result = ignition(cost=cost, target=target, resolve=resolve, opt=e["opt"])
    elif k == "trigger":
        result = trigger(e["on"], cost=cost, target=target, resolve=resolve, optional=e["optional"], opt=e["opt"] == OPT_NAME)
    elif k == "quick":
        result = quick(e["rng"], cost=cost, target=target, resolve=resolve, opt=e["opt"] if e["opt"] else False, on=e["on"])
    else:
        raise ValueError(k)
    if e.get('refund'):
        result = (*result[:7], dict(result[7], refund=1))
    return result


def fits(e):
    """L'effet tient dans les tableaux du moteur (PT instructions de coût et de cible, P - 1 de résolution, NF
    filtres)."""
    b = bricks(e)
    progs = b[4:7]
    fl = set()
    for pr in progs:
        for ins in pr:
            if ins[0] == core.OP_FILTER:
                fl.add(ins[2])
    extra = b[7] if len(b) > 7 else {}
    for f in extra.get("filters", ()):
        fl.add(f)
    return len(progs[0]) <= PT and len(progs[1]) <= PT and len(progs[2]) < P and len(fl) <= NF


# ---------------------------------------------------------------------- banque et decks

def to_bank(cset):
    """Liste de cartes (spécifications) → core.Bank. tprog est complété jusqu'à 2 × PT instructions : toutes les
    banques procédurales ont alors les mêmes formes (une seule compilation pour toutes les graines)."""
    import jax.numpy as jnp
    codes = [c["code"] for c in cset]
    effects = {c["code"]: [bricks(e) for e in c["effects"]] for c in cset}
    rec = dict(fusion={}, synchro={}, xyz={}, link={}, ritual={})
    for c in cset:
        if c["recipe"] is None:
            continue
        kind, r = c["recipe"]
        mat = lambda kw: cards.MAT(**kw)
        if kind == "fusion":
            # Seuls les noms bruts permettent un substitut dans Fusion.AddProcMix.
            mask = 0
            for j, material in enumerate(r):
                code = material.get("code")
                same_code = isinstance(code, (int, np.integer)) or isinstance(code, (list, tuple)) and len(code) == 2 and code[0] == code[1]
                if code and same_code and not any(material.get(k) for k in ("attr", "race", "setc", "lv", "typ", "not_typ")):
                    mask |= 1 << j
            rec["fusion"][c["code"]] = cards.FusionRecipe([mat(x) for x in r], mask)
        elif kind == "synchro":
            rec["synchro"][c["code"]] = (mat(r[0]), mat(r[1]))
        elif kind == "xyz":
            rec["xyz"][c["code"]] = (mat(r[0]), r[1])
        elif kind == "link":
            rec["link"][c["code"]] = (mat(r[0]), r[1], r[2], r[3])
        elif kind == "ritual":
            rec["ritual"][c["code"]] = (c["atk"], c["df"], c["lv"])
    asm = cards.assemble(codes, cards=effects, **rec)
    # Forme stable même si cette banque n'a tiré aucun piège à activation combinée.
    if not asm['einline'].shape[1]:
        asm['einline'] = np.zeros((len(cset), E), np.int32)
    sc = np.zeros((len(cset), NSET), np.int32)
    for i, c in enumerate(cset):
        sc[i, :len(c["setc"])] = c["setc"]
    bank = core.make_bank(np.array(codes), [c["atk"] for c in cset], [c["df"] for c in cset],
                          [c["lv"] for c in cset], ctype=[c["ctype"] for c in cset], attr=[c["attr"] for c in cset],
                          race=[c["race"] for c in cset], setcode=sc, lsc=[c["lsc"] for c in cset],
                          rsc=[c["rsc"] for c in cset], **asm)
    tp = bank.tprog
    pad = 2 * PT - tp.shape[2]
    return bank._replace(tprog=jnp.pad(tp, ((0, 0), (0, 0), (0, pad), (0, 0))),
                         erefund=bank.erefund if bank.erefund.shape[1] else jnp.zeros_like(bank.ekind))


def is_extra(c):
    return bool(c["ctype"] & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK))


def random_decks(cset, rng, b, d=40, x=15):
    """Decks aléatoires (sans structure) : [b, 2, d] et Extra Decks [b, 2, x], index dans la banque."""
    main = np.array([i for i, c in enumerate(cset) if not is_extra(c)])
    ext = np.array([i for i, c in enumerate(cset) if is_extra(c)])
    return main[rng.integers(0, len(main), (b, 2, d))], ext[rng.integers(0, len(ext), (b, 2, x))]


def structured_decks(cset, rng, b, d=40, x=15, p_rand=0.2):
    """Decks structurés [b, 2, d] / [b, 2, x] : cartes en 1 à 3 exemplaires (comme de vrais decks) et, avec une
    certaine probabilité, des paquets de mécaniques qui demandent des cartes assorties (decks aléatoires : quasi jamais
    d'invocation pendule, 3-5 fois moins de rituels / Xyz que les vrais decks, voir bench/coverage.py) : pendules,
    paires rituelles, fusions avec leurs matériaux nommés, monstres d'un même niveau avec des Xyz de ce rang,
    Syntoniseurs. Une part p_rand reste tirée au hasard."""
    idx = np.arange(len(cset))
    ext_m = np.array([is_extra(c) for c in cset])
    main, ext = idx[~ext_m], idx[ext_m]
    ct = np.array([c["ctype"] for c in cset])
    k0 = lambda c: c["effects"][0]["kind"] if c["effects"] else ""
    pend = idx[~ext_m & (ct & TYPE_PENDULUM != 0)]
    tuner = idx[~ext_m & (ct & TYPE_TUNER != 0) & (ct & TYPE_MONSTER != 0)]
    rit = [(i, c["effects"][0]["code"] - CODE0) for i, c in enumerate(cset) if k0(c) == "ritual"]
    fspell = idx[[k0(c) == "fusion_spell" for c in cset]]
    fus = [(i, [r["code"] - CODE0 for r in c["recipe"][1] if "code" in r]) for i, c in enumerate(cset)
           if c.get("recipe") and c["recipe"][0] == "fusion"]
    fus = [(i, m) for i, m in fus if m]
    lvl = np.array([c.get("lv", 0) for c in cset])
    xyz = idx[ext_m & (ct & TYPE_XYZ != 0)]
    nmon = idx[~ext_m & (ct & TYPE_MONSTER != 0) & (ct & TYPE_RITUAL == 0)]
    xyz_lv = [L for L in sorted(set(lvl[xyz])) if (lvl[nmon] == L).sum() >= 3]
    cop = lambda: int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))

    def one():
        if rng.random() < p_rand:
            return main[rng.integers(0, len(main), d)], ext[rng.integers(0, len(ext), x)]
        cards, extra = [], []
        if len(pend) >= 2 and rng.random() < 0.25:
            for c in rng.choice(pend, min(len(pend), int(rng.integers(3, 6))), replace=False):
                cards += [c] * int(rng.integers(2, 4))
        if rit and rng.random() < 0.25:
            for j in rng.choice(len(rit), min(len(rit), int(rng.integers(1, 3))), replace=False):
                cards += [rit[j][0]] * 2 + [rit[j][1]] * int(rng.integers(1, 3))
        if fus and len(fspell) and rng.random() < 0.3:
            for j in rng.choice(len(fus), min(len(fus), int(rng.integers(1, 3))), replace=False):
                extra += [fus[j][0]] * 2
                for mt in fus[j][1]:
                    cards += [mt] * 2
            cards += [rng.choice(fspell)] * int(rng.integers(2, 4))
        if xyz_lv and rng.random() < 0.3:          # paquet Xyz : monstres d'un même niveau + Xyz de ce rang
            L = rng.choice(xyz_lv)
            same = nmon[lvl[nmon] == L]
            for c in rng.choice(same, min(len(same), int(rng.integers(3, 5))), replace=False):
                cards += [c] * 2
            xs = xyz[lvl[xyz] == L]
            extra += list(rng.choice(xs, min(len(xs), 3), replace=False))
        if len(tuner) and rng.random() < 0.3:
            for c in rng.choice(tuner, min(len(tuner), int(rng.integers(2, 4))), replace=False):
                cards += [c] * 2
        cards = cards[:d - 10]                     # au moins 10 cartes de remplissage
        while len(cards) < d:
            cards += [rng.choice(main)] * cop()
        extra = extra[:x]
        while len(extra) < x:
            extra += [rng.choice(ext)] * cop()
        return np.array(cards[:d]), np.array(extra[:x])

    out = [[one() for _ in range(2)] for _ in range(b)]
    return (np.array([[p[0] for p in r] for r in out], np.int64), np.array([[p[1] for p in r] for r in out], np.int64))


# ---------------------------------------------------------------------- texte (débogage)

def describe(c):
    t = c["ctype"]
    names = [(TYPE_FUSION, "fusion"), (TYPE_SYNCHRO, "synchro"), (TYPE_XYZ, "xyz"), (TYPE_LINK, "link"),
             (TYPE_RITUAL, "rituel"), (TYPE_PENDULUM, "pendule"), (TYPE_TUNER, "syntoniseur"), (TYPE_QUICKPLAY, "jeu-rapide"),
             (TYPE_CONTINUOUS, "continu"), (TYPE_COUNTER, "contre"), (TYPE_EFFECT, "effet"), (TYPE_NORMAL, "normal")]
    kind = "monstre" if t & TYPE_MONSTER else "magie" if t & TYPE_SPELL else "piège"
    tags = " ".join(n for b, n in names if t & b)
    head = f"{c['code']} {kind} {tags}"
    if t & TYPE_MONSTER:
        head += f" niv {c['lv']} {c['atk']}/{c['df']} attr {c['attr']:#x} race {c['race']:#x}"
        if c["setc"]:
            head += f" set {c['setc'][0]:#x}"
    if c["recipe"]:
        head += f" recette {c['recipe']}"
    lines = [head] + [f"  - {e['kind']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k != "kind" and v)
                      for e in c["effects"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tire une banque procédurale et l'affiche")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main", type=int, default=40)
    ap.add_argument("--extra", type=int, default=10)
    a = ap.parse_args()
    cs = Gen(a.seed).make_set(a.main, a.extra)
    for c in cs:
        print(describe(c))
    vals = [sum(value(e) for e in c["effects"] if e["kind"] not in ("pend",)) for c in cs]
    print(f"{len(cs)} cartes ; valeur moyenne des effets {np.mean(vals):.2f}, max {np.max(vals):.2f}")
