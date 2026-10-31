"""Moteur JAX, jalons 1-4 : monstres normaux, magies (normales, jeu-rapide, continues), pièges (normaux, contre-pièges),
effets d'ignition, chaînes et fenêtres de réponse (règles MR5).

État en struct-of-arrays (une partie = N slots de cartes). `step(bank, s, a)` applique l'action puis avance
jusqu'à la prochaine décision qui a au moins deux options : les décisions à option unique sont jouées
automatiquement (comme le harness le fait côté oracle), et les effets s'exécutent instruction par instruction
(état D_RUN) jusqu'au prochain choix. Tout est vmappable (pas de Python sur les valeurs).

Les codes de zone, de position et de type reprennent les valeurs d'ygopro-core (LOCATION_*, POS_*, TYPE_*)
pour simplifier la comparaison avec l'oracle.

Effets : chaque carte a jusqu'à E effets, chacun = un genre (activation de magie, ignition) + 3 programmes de
bytecode (coût, cible, résolution) de P instructions (op, a, b, c) + une table de filtres. Machine à registres :
R registres ensembles (masques [N]). Les cartes sont écrites dans `cards.py`.

Chaîne : pile de CH maillons (carte, effet, joueur, registres, nié). Fenêtres de réponse (décision D_CHAIN) :
W_QUICK = QuickEffect d'ygopro-core (priorité alternée, deux passes de suite ferment la fenêtre), W_PHASE =
PhaseEvent (fenêtre de début/fin de phase). À la fermeture, s'il y a une chaîne elle se résout (du haut vers le
bas), puis une nouvelle fenêtre s'ouvre pour le joueur du tour ; sinon on reprend selon `ret` (R_*).
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# cache de compilation persistant : on ne recompile pas à chaque lancement tant que le code ne change pas
jax.config.update("jax_compilation_cache_dir", os.environ.get("YGO_JAX_CACHE", os.path.expanduser("~/.cache/ygo-jax")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

# zones (= LOCATION_* d'ygopro-core)
Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED, Z_EXTRA = 1, 2, 4, 8, 16, 32, 64
# positions (= POS_*) ; main = 10 et cimetière = 5, comme les renvoie l'oracle
P_FU_ATK, P_FD_ATK, P_FU_DEF, P_FD_DEF = 1, 2, 4, 8
P_HAND, P_GRAVE = 10, 5
P_SFU, P_SFD = 5, 10  # magie/piège recto (POS_FACEUP) / verso (POS_FACEDOWN)
P_FACEUP = P_FU_ATK | P_FU_DEF
# types (= TYPE_*)
TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL = 1, 2, 4, 0x10
TYPE_QUICKPLAY, TYPE_CONTINUOUS, TYPE_COUNTER = 0x10000, 0x20000, 0x100000
TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ, TYPE_LINK = 0x40, 0x80, 0x2000, 0x800000, 0x4000000
TYPE_EXTRA = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK  # monstres de l'Extra Deck
# statuts (bits)
ST_SUMMON_TURN, ST_ATTACKED, ST_FORM_CHANGED, ST_SEL = 1, 2, 4, 8  # ST_SEL : choisie dans la sélection en cours
ST_SET_TURN, ST_LEAVE = 16, 32  # posée ce tour (magie/piège) ; activée, ira au cimetière en fin de chaîne
ST_BDES = 64  # détruite au combat, part au cimetière à la fin du damage step (si elle est encore sur le terrain)
ST_ON = 128  # magie/piège continue résolue : ses effets continus s'appliquent (STATUS_EFFECT_ENABLED)
ST_CNT = 256  # ST_CNT << e : effet e déjà utilisé ce tour (SetCountLimit(1) sans code : par carte, effacé si elle bouge)
ST_CNTN = 1024  # ST_CNTN << e : idem, sans remise à zéro au tour suivant (EFFECT_FLAG_NO_TURN_RESET)
ST_FLIP_TURN = 4096  # invoquée par flip ce tour (STATUS_FLIP_SUMMON_TURN)
ST_TURN_RESET = ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN | ST_CNT | (ST_CNT << 1) | ST_FLIP_TURN
# phases (= PHASE_* d'ygopro-core, telles que vues aux décisions)
PH_DRAW, PH_STANDBY, PH_MAIN1, PH_BATTLE, PH_MAIN2, PH_END = 0x1, 0x2, 0x4, 0x8, 0x100, 0x200
# décisions en attente (D_RUN : effet en cours d'exécution, interne au moteur)
D_NONE, D_IDLE, D_BATTLE, D_TRIBUTE, D_ZONE, D_TARGET, D_DISCARD, D_SELECT, D_POS, D_RUN, D_CHAIN, D_TRIG = range(12)
# D_TRIG (interne) : déclencheurs en attente à tester, traité par settle hors de la boucle d'exécution
# à quoi sert le choix de case en cours
ZC_SUMMON, ZC_SSET, ZC_ACT, ZC_SPSUM = range(4)
# fenêtres de réponse ; W_TRIG : mise en chaîne des effets déclenchés (SEGOC)
W_QUICK, W_PHASE, W_TRIG = range(3)
# événements (EVENT_* d'ygopro-core) : disponibles pendant la fenêtre qui suit et la chaîne bâtie dessus ;
# ceux qui surviennent pendant la résolution d'un maillon remplacent les précédents (seul le dernier maillon compte)
# Les mêmes événements alimentent aussi `tev` (cumul depuis le dernier point d'événement) pour les déclencheurs.
EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON, EV_ATTACK, EV_TO_GRAVE, EV_BATTLE_DESTROYED, EV_FLIP, EV_PRE_DMG = range(8)
NEV = 8  # EV_PRE_DMG : EVENT_PRE_DAMAGE_CALCULATE (attaquant et cible, au début de la fenêtre du calcul des dommages)
# suite d'une fenêtre W_QUICK fermée sans chaîne (`chained` : une chaîne s'est résolue depuis l'ouverture)
R_IDLE, R_BATTLE, R_TO_BP, R_TO_EP, R_TO_M2B, R_TO_EPB, R_DP, R_PHASE, R_ATK1, R_ATK2, R_DMG1, R_DMG2, \
    R_DS1, R_DS2, R_DS3 = range(15)
# damage step (BattleCommand d'ygopro, MR5) : R_DS1 début (étape 20), R_DS2 après le retournement de la cible (22),
# R_DS3 calcul des dommages (24, PHASE_DAMAGE_CAL), puis R_DMG1 après le calcul (32) et R_DMG2 après l'envoi au
# cimetière (34). dstep : DS_BEFORE (fenêtres 1-2), DS_CAL (3), DS_AFTER (après le calcul)
DS_NONE, DS_BEFORE, DS_CAL, DS_AFTER = range(4)
# activation au damage step (bank.edmg) : EFFECT_FLAG_DAMAGE_STEP, EFFECT_FLAG_DAMAGE_CAL,
# condition aux.StatChangeDamageStepCondition (pas après le calcul des dommages)
DMG_STEP, DMG_CAL, DMG_STAT = 1, 2, 4
# limites (bank.eopt) : une fois par tour par nom, une fois par tour par carte, une fois tant que la carte reste
# (SetCountLimit(1) + EFFECT_FLAG_NO_TURN_RESET)
OPT_NONE, OPT_NAME, OPT_CARD, OPT_CARD_NR = range(4)
# fenêtres de phase (PhaseEvent)
WP_DRAW, WP_STANDBY, WP_BSTART, WP_BEND, WP_END = range(5)

HAND_LIMIT = 6
N_MAIN_ZONES = 5
KO = 4  # registres « par nom » (une fois par tour, activations interdites) par joueur

# effets
E = 2           # effets par carte
P = 12          # instructions par programme
PT = 6          # instructions testées par programme de coût / cible (test_ok)
R = 4           # registres ensembles
NF = 4          # filtres par effet
KC = 24         # (carte, effet) candidats à l'activation testés par legal (main + terrain du joueur qui décide)
K_NONE, K_ACTIVATE, K_IGNITION, K_TRIG_F, K_TRIG_O, K_QUICK, K_CONT = range(7)  # déclencheur obligatoire / optionnel ;
# K_QUICK : effet rapide de monstre (vitesse 2), activable depuis les zones erng ; K_CONT : effet continu, appliqué
# tant que sa carte est recto dans les zones erng (et résolue, pour une magie/piège)
PG_COST, PG_TARGET, PG_RESOLVE = range(3)
CH = 12          # maillons de chaîne (au plus 1 ignition + 2 × 5 cases magie/piège)
(OP_END, OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_CHOOSE, OP_MOVE, OP_SPSUMMON, OP_DRAW,
 OP_DAMAGE, OP_RECOVER, OP_SHUFFLE_HAND, OP_COND_CHAIN, OP_CHAINCARD, OP_NEGATE, OP_PAY_LP, OP_EVENT, OP_COND_TURN,
 OP_COND_EVP, OP_COND_SELF, OP_LOCK, OP_MODIFY, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE, OP_FUSMAT, OP_RITMAT) = range(28)
N_OPS = 28
# champs d'un filtre ; les zones sont relatives au joueur de l'effet (moi / adversaire)
F_LOC_ME, F_LOC_OP, F_POS, F_TYPE, F_NOT_SELF, F_ATK_LO, F_ATK_HI, F_DEF_LO, F_DEF_HI, F_ATTR, F_RACE, F_SET, \
    F_CODE1, F_CODE2, F_ST, F_LV_LO, F_LV_HI, F_SPSUM, F_NODIS, F_FUS, F_RIT = range(21)
FW = 21  # F_RIT : monstre rituel invocable maintenant (cache rz_c) en mode RIT_GREATER ou RIT_EQUAL  # F_FUS : monstre fusion de l'Extra Deck invocable par fusion maintenant (cache fz_c)  # F_ST : l'un des bits de statut ; F_SPSUM : peut être invoquée spécialement (pas de FL_NO_SS) ;
# F_NODIS : effets non annulés (not IsDisabled)
NSET = 4  # archétypes (setcode 16 bits) par carte
# modificateurs de stats (EFFECT_*), effets continus (bank.emod) et temporaires (table tm_* de l'état)
# UPDATE_*, SET_*, SET_*_FINAL, SET_BASE_* pour l'ATK puis la DEF
MC_NONE, MC_ATK, MC_SET_ATK, MC_FINAL_ATK, MC_BASE_ATK, MC_DEF, MC_SET_DEF, MC_FINAL_DEF, MC_BASE_DEF = range(9)
# interdictions et protections (effets continus) : EFFECT_CANNOT_ATTACK, CANNOT_BE_BATTLE_TARGET, CANNOT_BE_EFFECT_TARGET,
# IMMUNE_EFFECT (effets de piège), CANNOT_SPECIAL_SUMMON (cartes visées, toutes zones), INDESTRUCTABLE_BATTLE (valeur :
# ATK minimale du monstre adverse, 0 = toujours), CANNOT_ACTIVATE (cartes piège, joueurs M_VAL : 1 = moi, 2 = adversaire)
MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP, MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP = range(9, 16)
MC_DISABLE = 16  # EFFECT_DISABLE : effets de la carte annulés (continus inactifs, pas d'activation, maillons annulés)
FL_NO_ATTACK, FL_NO_BTARGET, FL_NO_ETARGET, FL_IMM_TRAP, FL_NO_SS, FL_DISABLED = 1, 2, 4, 8, 16, 32  # bits de fl_c
FL_ONF = 64  # recto sur le terrain, effets actifs (STATUS_EFFECT_ENABLED) au calcul précédent
PF_NO_TRAP = 1  # bits de pf_c
# card::add_effect : un modificateur temporaire de ce code efface ceux de ces codes déjà sur la carte (masque de bits)
_b = lambda *cs: sum(1 << c for c in cs)
MC_KILL = [0] * 32
for _set, _fin, _upd, _base in ((MC_SET_ATK, MC_FINAL_ATK, MC_ATK, MC_BASE_ATK), (MC_SET_DEF, MC_FINAL_DEF, MC_DEF, MC_BASE_DEF)):
    MC_KILL[_set] = MC_KILL[_base] = _b(_set, _fin, _base)
    MC_KILL[_fin] = _b(_upd, _set, _fin)
# valeur d'un modificateur temporaire (OP_MODIFY) : constante, moitié de l'ATK de base / de l'ATK courante de la cible
MV_CONST, MV_HALF_BASE_ATK, MV_HALF_ATK = range(3)
# durée : fin du tour (RESETS_STANDARD_PHASE_END), fin du calcul des dommages (RESET_PHASE + PHASE_DAMAGE_CAL)
TD_TURN, TD_DCAL = range(2)
# champs de emod : code, la carte elle-même (SINGLE_RANGE) ou le filtre 0 (FIELD), valeur, × nombre de cartes du
# filtre 1, condition : joueur du tour (1 : moi, 2 : adversaire), phases (masque PH_*, 0 : toutes), nombre de cartes
# du filtre 1 dans [M_CMIN, M_CMAX] (si M_CNT ; M_PER et M_CNT s'excluent)
M_CODE, M_SELF, M_VAL, M_PER, M_TP, M_PH, M_CNT, M_CMIN, M_CMAX = range(9)
MW = 9
KM = 16         # effets continus actifs pris en compte (compaction, comme KC)
TM = 16         # modificateurs temporaires par tour
FK = 4          # matériaux de fusion au plus (un filtre par matériau)
# rituel : somme des niveaux des libérations >= niveau sans libération superflue (Greater), ou égale (Equal)
RIT_GREATER, RIT_EQUAL = 1, 2
RIT_MON = TYPE_MONSTER | TYPE_RITUAL  # les sorts rituels ont aussi TYPE_RITUAL
LVMAX, SMAX = 12, 26  # niveaux des matériaux ; sommes suivies (masques de bits) jusqu'à SMAX
EXK = 32        # monstres fusion de l'Extra Deck évalués par refresh (fz_c)
BIG = 1 << 20

# actions : verbe × slot, puis actions spéciales
V_SUMMON, V_MSET, V_REPOS, V_ATTACK, V_SELECT, V_SSET, V_ACT0 = range(7)
NV = V_ACT0 + E
A_TO_BP, A_TO_M2, A_TO_EP, A_ZONE0, A_POS_ATK, A_POS_DEF, A_PASS = 0, 1, 2, 3, 10, 11, 12  # A_ZONE0 + z
N_SPECIAL = 13


def n_actions(n):
    return NV * n + N_SPECIAL


def special(n, k):
    return NV * n + k


class Bank(NamedTuple):
    """Banque de cartes globale (statique, partagée par toutes les parties)."""
    code: jnp.ndarray   # [C] id ygopro (pour la comparaison avec l'oracle)
    atk: jnp.ndarray
    df: jnp.ndarray
    level: jnp.ndarray
    ctype: jnp.ndarray  # [C] bits TYPE_*
    attr: jnp.ndarray   # [C] bits ATTRIBUTE_*
    race: jnp.ndarray   # [C] bits RACE_*
    ekind: jnp.ndarray  # [C, E] K_*
    prog: jnp.ndarray   # [C, E, 3, P, 4] (op, a, b, c)
    tprog: jnp.ndarray  # [C, E, PT2, 4] coût puis cible sans les OP_END (mode test) ; PT2 = le plus long de la banque
    filt: jnp.ndarray   # [C, E, NF, FW]
    etrig: jnp.ndarray  # [C, E] bits 1 << EV_* : activation seulement en réponse à ces événements (0 : libre)
    eopt: jnp.ndarray   # [C, E] une fois par tour et par nom (SetCountLimit(1, id))
    erng: jnp.ndarray   # [C, E] zones d'où un effet rapide de monstre s'active (Z_HAND | Z_MZONE) ; zones d'un effet continu
    setcode: jnp.ndarray  # [C, NSET] archétypes (0 : aucun)
    emod: jnp.ndarray   # [C, E, MW] effet continu (K_CONT) : M_*
    edmg: jnp.ndarray   # [C, E] activation au damage step : bits DMG_*
    fmat: jnp.ndarray   # [C, FK, FW] matériaux de fusion (un filtre par matériau, sur la main et le terrain)
    fmn: jnp.ndarray    # [C] nombre de matériaux (0 : pas de recette)
    erl: jnp.ndarray    # [C] EnableReviveLimit : invocable spécialement seulement si correctement invoquée avant


def make_bank(code, atk, df, level, ctype=None, ekind=None, prog=None, filt=None, etrig=None, eopt=None, attr=None,
              race=None, erng=None, setcode=None, emod=None, edmg=None, fmat=None, fmn=None, erl=None):
    c = code.shape[0]
    i = lambda x, shape: jnp.zeros(shape, I32) if x is None else jnp.asarray(x, I32)
    return Bank(code=jnp.asarray(code, I32), atk=jnp.asarray(atk, I32), df=jnp.asarray(df, I32), tprog=test_prog(prog, c),
                level=jnp.asarray(level, I32),
                ctype=jnp.full(c, TYPE_MONSTER | TYPE_NORMAL, I32) if ctype is None else jnp.asarray(ctype, I32),
                attr=i(attr, (c,)), race=i(race, (c,)),
                ekind=i(ekind, (c, E)), prog=i(prog, (c, E, 3, P, 4)), filt=i(filt, (c, E, NF, FW)),
                etrig=i(etrig, (c, E)), eopt=i(eopt, (c, E)), erng=i(erng, (c, E)), setcode=i(setcode, (c, NSET)),
                emod=i(emod, (c, E, MW)), edmg=i(edmg, (c, E)), fmat=i(fmat, (c, FK, FW)), fmn=i(fmn, (c,)),
                erl=i(erl, (c,)))


def test_prog(prog, c):
    """Programmes de test compacts (numpy, à la construction de la banque) : coût puis cible, sans les OP_END de
    remplissage (sans effet en mode test). test_ok ne parcourt que la longueur du plus long, pas 2 × PT."""
    import numpy as np
    pr = np.zeros((c, E, 3, P, 4), np.int32) if prog is None else np.asarray(prog, np.int32)
    ins = pr[:, :, :PG_RESOLVE, :PT].reshape(c, E, 2 * PT, 4)
    keep = ins[..., 0] != OP_END
    n = max(1, int(keep.sum(-1).max()))
    order = np.argsort(~keep, axis=-1, kind="stable")[..., :n]  # instructions gardées d'abord, dans l'ordre
    out = np.take_along_axis(ins, order[..., None], axis=2)
    out[~np.take_along_axis(keep, order, axis=-1)] = 0
    return jnp.asarray(out)


class State(NamedTuple):
    cid: jnp.ndarray     # [N] index dans la banque
    zone: jnp.ndarray    # [N]
    seq: jnp.ndarray     # [N] case (terrain) ou rang dans la zone (deck : dessus = seq max)
    ctrl: jnp.ndarray    # [N]
    owner: jnp.ndarray   # [N]
    pos: jnp.ndarray     # [N]
    st: jnp.ndarray      # [N] bits ST_*
    lp: jnp.ndarray      # [2]
    turn: jnp.ndarray
    tp: jnp.ndarray      # joueur du tour
    phase: jnp.ndarray
    ns_used: jnp.ndarray
    dec: jnp.ndarray     # D_*
    dec_player: jnp.ndarray
    pend: jnp.ndarray    # slot en cours (monstre à invoquer, attaquant, magie à poser/activer), -1 sinon
    pend_set: jnp.ndarray  # l'invocation en cours est une pose
    need: jnp.ndarray    # sélections restantes (tributs, défausses, choix d'effet)
    picks: jnp.ndarray   # sélections déjà faites dans la décision en cours
    zctx: jnp.ndarray    # ZC_* : usage du choix de case en cours
    zsel: jnp.ndarray    # case choisie pour l'invocation spéciale en cours
    ch_src: jnp.ndarray  # effet en cours (activation ou résolution) : carte source (-1 si aucun)
    ch_eff: jnp.ndarray  # index d'effet
    ch_p: jnp.ndarray    # joueur qui active
    ch_k: jnp.ndarray    # index de son maillon (= n_chain pendant l'activation)
    ch_dis: jnp.ndarray  # activé par une carte sous EFFECT_DISABLE_EFFECT : le maillon sera annulé (DISABLE_CHAIN)
    pg: jnp.ndarray      # programme en cours (PG_*)
    pc: jnp.ndarray      # instruction en cours
    reg: jnp.ndarray     # [R, N] registres ensembles
    done: jnp.ndarray
    odraw: jnp.ndarray   # pioche de règle impossible : défaite au prochain « adjust » (activation, fin de fenêtre de phase)
    winner: jnp.ndarray  # -1 si pas fini
    # pile de chaîne
    lk_src: jnp.ndarray  # [CH] carte
    lk_eff: jnp.ndarray  # [CH]
    lk_p: jnp.ndarray    # [CH]
    lk_reg: jnp.ndarray  # [CH, R, N] registres au moment de l'activation (une carte qui bouge en sort)
    lk_neg: jnp.ndarray  # [CH] activation niée
    lk_t: jnp.ndarray    # [CH] horloge de déplacement à l'activation
    mv: jnp.ndarray      # [N] horloge du dernier déplacement de la carte
    clk: jnp.ndarray     # horloge de déplacement : une carte déplacée après l'activation sort de lk_reg (sans le recopier)
    en_t: jnp.ndarray    # [N] horloge à l'activation des effets continus de la carte (enable_field_effect : arrivée sur
    # le terrain, retournement recto, résolution) ; ordonne les modificateurs comme les id d'effet d'ygopro
    cav: jnp.ndarray     # [N, E] condition de l'effet continu remplie au dernier calcul (EFFECT_STATUS_AVAILABLE)
    cav_t: jnp.ndarray   # [N, E] horloge où elle l'est redevenue (is_available donne alors un nouvel id à l'effet)
    n_chain: jnp.ndarray
    # fenêtres
    wk: jnp.ndarray      # W_*
    cw_passed: jnp.ndarray  # l'autre joueur a passé
    ret: jnp.ndarray     # R_* : suite de la fenêtre W_QUICK
    chained: jnp.ndarray
    ph_restart: jnp.ndarray  # joueur qui reprend la fenêtre de phase après une chaîne
    wph: jnp.ndarray     # WP_* : fenêtre de phase en cours
    bend_ep: jnp.ndarray  # fin de phase de combat vers la End Phase (sinon Main 2)
    atk: jnp.ndarray     # attaquant en cours
    atk_tgt: jnp.ndarray  # cible (-1 : attaque directe)
    # événements récents
    ev_g: jnp.ndarray    # [NEV, N] cartes concernées (eg)
    ev_p: jnp.ndarray    # [NEV] joueur à l'origine (ep)
    # déclencheurs
    pzone: jnp.ndarray   # [N] zone précédente (IsPreviousLocation)
    tev: jnp.ndarray     # [NEV, N] événements subis depuis le dernier point d'événement
    tg: jnp.ndarray      # [N] effet déclenché en attente
    tg_e: jnp.ndarray    # [N] son index d'effet
    tst: jnp.ndarray     # étape SEGOC : 0/1 obligatoires (tour, adversaire), 2/3 optionnels
    pt_p: jnp.ndarray    # fenêtre à ouvrir si aucun déclencheur n'est activé : joueur, autre joueur a passé
    pt_passed: jnp.ndarray
    dstep: jnp.ndarray   # DS_* : où en est le damage step
    ds_t: jnp.ndarray    # horloge au début du damage step (attaquant ou cible déplacés depuis : attaque annulée)
    ds_flip: jnp.ndarray  # la cible a été retournée au début du damage step (événement FLIP au calcul des dommages)
    opt_cid: jnp.ndarray   # [2, KO] cid + 1 des effets « une fois par tour » déjà utilisés
    lock_cid: jnp.ndarray  # [2, KO] cid + 1 des cartes dont le joueur ne peut plus activer les effets ce tour
    # stats courantes (cache recalculé par refresh après chaque micro-pas) et modificateurs temporaires
    atk_c: jnp.ndarray   # [N] ATK courante (hors zone monstres : ATK de base)
    df_c: jnp.ndarray    # [N]
    batk_c: jnp.ndarray  # [N] ATK de base courante (GetBaseAttack)
    tm_code: jnp.ndarray  # [TM] MC_* (dans l'ordre de création ; table vidée en fin de tour)
    tm_m: jnp.ndarray    # [TM, N] cartes concernées (une carte déplacée depuis tm_t en sort, RESETS_STANDARD)
    tm_v: jnp.ndarray    # [TM, N] valeur par carte
    tm_t: jnp.ndarray    # [TM] horloge de déplacement à la création
    tm_dur: jnp.ndarray  # [TM] TD_*
    tm_rd: jnp.ndarray   # [TM] RESET_DISABLE (effet de la carte sur elle-même) : effacé quand la carte est annulée
    tm_n: jnp.ndarray
    fl_c: jnp.ndarray    # [N] interdictions / protections courantes (FL_*), calculées par refresh
    ib_c: jnp.ndarray    # [N] indestructible au combat contre un monstre d'ATK >= ib_c (BIG : non)
    pf_c: jnp.ndarray    # [2] interdictions des joueurs (PF_*)
    proc: jnp.ndarray    # [N] correctement invoquée (STATUS_PROC_COMPLETE : invocation fusion...) ; effacé en main / deck
    fz_c: jnp.ndarray    # [N] monstre fusion de l'Extra Deck réalisable avec la main et le terrain de son propriétaire
    fsel: jnp.ndarray    # sélection de matériaux en cours (OP_FUSMAT) : jamais jouée d'office s'il reste un choix
    rz_c: jnp.ndarray    # [N] monstre rituel en main invocable : bit 0 en mode Greater, bit 1 en mode Equal
    rsel: jnp.ndarray    # sélection de libérations rituelles en cours (OP_RITMAT) : mode RIT_*, 0 sinon
    ovf: jnp.ndarray     # débordement des tables (bits) : 1 = plus de KM effets continus actifs, 2 = plus de TM modificateurs


I32 = jnp.int32


def _merge(i, outs):
    """Sélection par i parmi les sorties des branches, champ par champ. Un champ qu'aucune branche ne modifie (même
    objet que l'entrée) passe tel quel, et seules les valeurs distinctes entrent dans la sélection."""
    flat = [jax.tree.flatten(o) for o in outs]
    res = []
    for vals in zip(*[f[0] for f in flat]):
        uniq, pos = [], []
        for v in vals:
            j = next((j for j, u in enumerate(uniq) if u is v), None)
            if j is None:
                j = len(uniq)
                uniq.append(v)
            pos.append(j)
        if len(uniq) == 1:
            res.append(uniq[0])
        else:
            dt = jnp.result_type(uniq[0])
            w = i if pos == list(range(len(vals))) else jnp.asarray(pos, I32)[i]
            res.append(lax.select_n(w, *[jnp.asarray(u, dt) for u in uniq]))
    return flat[0][1].unflatten(res)


def _switch(i, fs, *ops):
    """Remplace lax.switch : calcule toutes les branches (ce que fait déjà lax.switch sous vmap) mais ne sélectionne
    que les champs modifiés. Sous vmap, lax.switch sélectionne l'état entier pour chaque branche."""
    i = jnp.clip(jnp.asarray(i), 0, len(fs) - 1).astype(I32)
    return _merge(i, [f(*ops) for f in fs])


def _cond(p, f, g, *ops):
    return _switch(jnp.asarray(p).astype(I32), [g, f], *ops)


def _i(x):
    return jnp.asarray(x, I32)


def rel(p, r):
    """Joueur relatif (0 = moi, 1 = adversaire) → absolu."""
    return jnp.where(r == 0, p, 1 - p).astype(I32)


# ---------------------------------------------------------------- primitives O(N)

def count(s, zone, p):
    return ((s.zone == zone) & (s.ctrl == p)).sum().astype(I32)


def zone_used(s, p, zone=Z_MZONE):
    """[8] : cases occupées du joueur p dans la zone (monstres 0-6, magies/pièges 0-7)."""
    m = (s.zone == zone) & (s.ctrl == p)
    return jnp.zeros(8, bool).at[jnp.where(m, s.seq, 8)].set(True, mode="drop")


def free_main(s, p, zone=Z_MZONE):
    return (~zone_used(s, p, zone)[:N_MAIN_ZONES]).sum().astype(I32)


def leave_hand(s, c):
    """Retire c de la main : les cartes suivantes se décalent."""
    p, sq = s.ctrl[c], s.seq[c]
    shift = (s.zone == Z_HAND) & (s.ctrl == p) & (s.seq > sq)
    return s._replace(seq=s.seq - shift.astype(I32))


ORDERED = (Z_DECK, Z_HAND, Z_GRAVE, Z_REMOVED, Z_EXTRA)


def compact(s, leaving):
    """Les cartes `leaving` vont quitter leur zone : resserre les seq des zones ordonnées (deck, main,
    cimetière, bannies). O(N)."""
    n = s.cid.shape[0]
    oz = jnp.full(n, -1, I32)
    for i, z in enumerate(ORDERED):
        oz = jnp.where(s.zone == z, i, oz)
    grp = s.ctrl * len(ORDERED) + jnp.maximum(oz, 0)
    hit = leaving & (oz >= 0)
    bins = jnp.zeros((2 * len(ORDERED), n), I32).at[grp, s.seq].add(hit.astype(I32))
    before = jnp.cumsum(bins, axis=1) - bins       # partants de seq strictement inférieure
    shift = before[grp, s.seq]
    return s._replace(seq=jnp.where((oz >= 0) & ~leaving, s.seq - shift, s.seq))


def move(s, c, zone, seq, ctrl, pos):
    return s._replace(pzone=s.pzone.at[c].set(s.zone[c]), zone=s.zone.at[c].set(zone), seq=s.seq.at[c].set(seq), ctrl=s.ctrl.at[c].set(ctrl),
                      pos=s.pos.at[c].set(pos), st=s.st.at[c].set(0), mv=s.mv.at[c].set(s.clk + 1),
                      en_t=s.en_t.at[c].set(s.clk + 1), clk=s.clk + 1)


def enable(s, mask):
    """Les effets continus des cartes mask s'activent maintenant (nouvel id d'effet dans ygopro)."""
    return s._replace(en_t=jnp.where(mask, s.clk + 1, s.en_t), clk=s.clk + 1)


LOC_RANK = (Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED)


def group_send(s, mask, zone, pos, zone2=None, pos2=None, alt=None):
    """Envoie les cartes `mask` dans la zone ordonnée `zone` (cimetière, main) de leur propriétaire, d'un coup.
    Comme ygopro-core (card_operation_sort), elles y arrivent triées par contrôleur (joueur du tour d'abord),
    zone d'origine, puis seq (croissante sur le terrain et en main, décroissante pour deck/cimetière/bannies). O(N).
    alt [N] : ces cartes vont plutôt dans zone2 / pos2 (un monstre de l'Extra Deck renvoyé en main), dans le même
    passage."""
    n = s.cid.shape[0]
    if alt is not None:
        dest = jnp.where(alt, zone2, zone)
        pos = jnp.where(alt, pos2, pos)
    else:
        dest = zone
    zr = jnp.zeros(n, I32)
    for i, z in enumerate(LOC_RANK):
        zr = jnp.where(s.zone == z, i, zr)
    desc = (s.zone == Z_DECK) | (s.zone == Z_GRAVE) | (s.zone == Z_REMOVED)
    key = ((s.ctrl != s.tp) * len(LOC_RANK) + zr) * n + jnp.where(desc, n - 1 - s.seq, s.seq)
    nk = 2 * len(LOC_RANK) * n
    if alt is not None:
        key = key + jnp.where(alt, nk, 0)  # rang compté séparément pour chaque destination
        nk *= 2
    seq = s.seq
    for o in (0, 1):
        mo = mask & (s.owner == o)
        bins = jnp.zeros(nk, I32).at[key].add(mo.astype(I32))
        rank = (jnp.cumsum(bins) - bins)[key]
        if alt is not None:
            rank = rank - jnp.where(alt, (mo & ~alt).sum(), 0)
            base = jnp.where(alt, count(s, zone2, o), count(s, zone, o))
        else:
            base = count(s, zone, o)
        seq = jnp.where(mo, base + rank, seq)
    s = compact(s, mask)
    out = (dest == Z_HAND) | (dest == Z_DECK) | (dest == Z_EXTRA)  # STATUS_PROC_COMPLETE perdu
    s = s._replace(proc=jnp.where(mask & out, False, s.proc))
    return s._replace(pzone=jnp.where(mask, s.zone, s.pzone), zone=jnp.where(mask, dest, s.zone),
                      seq=jnp.where(mask, seq, s.seq), ctrl=jnp.where(mask, s.owner, s.ctrl),
                      pos=jnp.where(mask, pos, s.pos), st=jnp.where(mask, 0, s.st), mv=jnp.where(mask, s.clk + 1, s.mv),
                      en_t=jnp.where(mask, s.clk + 1, s.en_t),
                      clk=s.clk + 1)


def group_to_hand(bank, s, mask):
    """Renvoi en main ; un monstre de l'Extra Deck retourne dans l'Extra Deck (verso)."""
    ex = mask & ((bank.ctype[s.cid] & TYPE_EXTRA) != 0)
    return group_send(s, mask, Z_HAND, P_HAND, Z_EXTRA, P_FD_DEF, alt=ex)


def group_to_grave(s, mask):
    return raise_ev(group_send(s, mask, Z_GRAVE, P_GRAVE), EV_TO_GRAVE, mask, s.tp)


def selected_to_grave(s):
    return group_to_grave(s, (s.st & ST_SEL) != 0)._replace(picks=_i(0))


def lose(s, p, cond):
    """Si cond : p perd (sans écraser une victoire déjà acquise)."""
    hit = cond & ~s.done
    return s._replace(done=s.done | hit, winner=jnp.where(hit, 1 - p, s.winner))


def damage(s, p, amount):
    lp = s.lp.at[p].set(jnp.maximum(0, s.lp[p] - amount))
    s = s._replace(lp=lp)
    return lose(s, p, lp[p] <= 0)


def draw(s, p):
    m = (s.zone == Z_DECK) & (s.ctrl == p)
    c = jnp.argmax(jnp.where(m, s.seq, -1))
    empty = ~m.any()
    s2 = move(s, c, Z_HAND, count(s, Z_HAND, p), p, P_HAND)
    s = jax.tree.map(lambda a, b: jnp.where(empty, a, b), s, s2)
    return lose(s, p, empty)


def rule_draw(s, p):
    """Pioche de la Draw Phase : ygopro-core ne vérifie la défaite qu'au prochain adjust."""
    empty = ~((s.zone == Z_DECK) & (s.ctrl == p)).any()
    return _cond(empty, lambda s: s._replace(odraw=jnp.bool_(True)), lambda s: draw(s, p), s)


def adjust(s):
    return lose(s, s.tp, s.odraw)._replace(odraw=jnp.bool_(False))


def raise_ev(s, ev, mask, p):
    return s._replace(ev_g=s.ev_g.at[ev].set(s.ev_g[ev] | mask), ev_p=s.ev_p.at[ev].set(p),
                      tev=s.tev.at[ev].set(s.tev[ev] | mask))


def name_in(table, cid):
    """cid [...] présent dans le registre par nom `table` [KO] ?"""
    return (table == (cid + 1)[..., None]).any(-1)


def name_add(table, cid):
    """Ajoute cid au registre [KO] (première case libre ; la dernière est écrasée si tout est plein)."""
    i = jnp.where((table == 0).any(), jnp.argmax(table == 0), KO - 1)
    return table.at[i].set(cid + 1)


def clear_ev(s):
    return s._replace(ev_g=jnp.zeros_like(s.ev_g))


def ev_bits(s):
    return (s.ev_g.any(axis=1) * (1 << jnp.arange(NEV))).sum().astype(I32)


def tributes_needed(bank, s, c):
    lv = bank.level[s.cid[c]]
    return jnp.where(lv >= 7, 2, jnp.where(lv >= 5, 1, 0)).astype(I32)


# ---------------------------------------------------------------- fenêtres et déroulement

def open_quick(s, p, ret=None, passed=False, chained=None):
    """Fenêtre W_QUICK (QuickEffect) : p a la priorité ; `passed` : l'autre joueur a déjà passé."""
    return s._replace(dec=_i(D_CHAIN), dec_player=_i(p), wk=_i(W_QUICK), cw_passed=jnp.bool_(passed),
                      ret=s.ret if ret is None else _i(ret),
                      chained=s.chained if chained is None else jnp.bool_(chained))


def open_phase(s, wph, phase):
    """Fenêtre W_PHASE (PhaseEvent) : le joueur du tour d'abord."""
    return s._replace(phase=_i(phase), dec=_i(D_CHAIN), dec_player=s.tp, wk=_i(W_PHASE), cw_passed=jnp.bool_(False),
                      wph=_i(wph))


def act_ok(bank, s, h, e, p):
    """L'effet e de h est activable par p : coût + cible exécutables, pas d'interdiction ni de limite par nom."""
    cid = s.cid[h]
    # une carte aux effets annulés peut encore les activer (is_activateable ne teste pas STATUS_DISABLED) : ils sont
    # annulés à la résolution (load_link)
    return (test_ok(bank, s, h, e, p) & ~name_in(s.lock_cid[p], cid)
            & ~((bank.eopt[cid, e] == OPT_NAME) & name_in(s.opt_cid[p], cid))
            & ~((bank.eopt[cid, e] >= OPT_CARD) & ((s.st[h] & ((ST_CNT | ST_CNTN) << e)) != 0)))


def trig_next(bank, s):
    """Diffère trig_eval à settle : sous vmap, l'appeler ici le calculerait à chaque instruction exécutée."""
    return s._replace(dec=_i(D_TRIG))


def trig_eval(bank, s):
    """Prochaine décision de mise en chaîne des déclencheurs (à partir de l'étape s.tst) ; s'il n'y en a plus,
    fenêtre de réponse (après la chaîne bâtie : l'adversaire du dernier maillon a la priorité)."""
    # étape SEGOC de chaque déclencheur en attente ; activabilité testée une seule fois pour les 4 étapes
    stage = (s.ctrl != s.tp) + 2 * (bank.ekind[s.cid, s.tg_e] == K_TRIG_O)
    hs = jnp.nonzero(s.tg, size=KC, fill_value=0)[0]
    okc = jax.vmap(lambda h: act_ok(bank, s, h, s.tg_e[h], s.ctrl[h]))(hs) & (jnp.arange(KC) < s.tg.sum())
    ok = s.tg & (jnp.zeros_like(s.seq).at[hs].add(okc.astype(I32)) > 0)
    avail = (ok[None, :] & (stage[None, :] == jnp.arange(4)[:, None])).any(1) & (jnp.arange(4) >= s.tst)
    nxt = jnp.argmax(avail).astype(I32)

    def go(s):
        return s._replace(tst=nxt, dec=_i(D_CHAIN), dec_player=jnp.where(nxt % 2 == 0, s.tp, 1 - s.tp).astype(I32),
                          wk=_i(W_TRIG), cw_passed=jnp.bool_(False))

    def finish(s):
        s = s._replace(tg=jnp.zeros_like(s.tg), wk=_i(W_QUICK), dec=_i(D_CHAIN))
        top = jnp.clip(s.n_chain - 1, 0, CH - 1)
        return _cond(s.n_chain > 0, lambda s: s._replace(dec_player=1 - s.lk_p[top], cw_passed=jnp.bool_(False)),
                        lambda s: s._replace(dec_player=s.pt_p, cw_passed=s.pt_passed), s)
    return _cond(avail.any(), go, finish, s)


def trig_point(bank, s):
    """Point d'événement, juste après l'ouverture d'une fenêtre W_QUICK (chaîne vide) : les effets déclenchés par
    les événements cumulés dans `tev` passent d'abord (SEGOC), la fenêtre ne s'ouvre qu'ensuite."""
    bits = (s.tev * (1 << jnp.arange(NEV))[:, None]).sum(0)
    tg, tg_e = s.tg, s.tg_e
    for e in range(E):
        k = bank.ekind[s.cid, e]
        hit = ((k == K_TRIG_F) | (k == K_TRIG_O)) & ((bank.etrig[s.cid, e] & bits) != 0) & ~tg
        tg_e, tg = jnp.where(hit, e, tg_e).astype(I32), tg | hit
    s = s._replace(tg=tg, tg_e=tg_e, tev=jnp.zeros_like(s.tev), pt_p=s.dec_player, pt_passed=s.cw_passed, tst=_i(0))
    return _cond(tg.any(), lambda s: trig_next(bank, s), lambda s: s, s)


def start_turn(s):
    tp = jnp.where(s.turn == 0, 0, 1 - s.tp).astype(I32)
    s = s._replace(turn=s.turn + 1, tp=tp, ns_used=jnp.bool_(False), st=s.st & ~ST_TURN_RESET, phase=_i(PH_DRAW),
                   tev=jnp.zeros_like(s.tev), opt_cid=jnp.zeros_like(s.opt_cid), lock_cid=jnp.zeros_like(s.lock_cid),
                   tm_code=jnp.zeros_like(s.tm_code), tm_m=jnp.zeros_like(s.tm_m), tm_n=_i(0))
    # pioche (sauf au premier tour) puis fenêtre ; ensuite fenêtre de phase
    return _cond(s.turn > 1, lambda s: open_quick(rule_draw(s, tp), tp, R_DP, chained=False),
                        lambda s: open_phase(s, WP_DRAW, PH_DRAW), s)


def end_phase(s):
    over = count(s, Z_HAND, s.tp) - HAND_LIMIT
    return _cond(over > 0,
                        lambda s: s._replace(phase=_i(PH_END), dec=_i(D_DISCARD), dec_player=s.tp, need=over,
                                             picks=_i(0)),
                        start_turn, s)


def to_idle(s):
    return s._replace(dec=_i(D_IDLE), dec_player=s.tp)


def to_battle(s):
    return s._replace(dec=_i(D_BATTLE), dec_player=s.tp)


def atk_canceled(s):
    """Attaque annulée avant le damage step (adjust, étape 14 : STATUS_ATTACK_CANCELED) : l'attaquant a quitté le
    terrain, a bougé depuis la déclaration (ds_t) ou n'est plus en position d'attaque recto."""
    a = s.atk
    return (s.zone[a] != Z_MZONE) | (s.mv[a] > s.ds_t) | (s.pos[a] != P_FU_ATK)


def damage_step(bank, s):
    """Début du damage step (étapes 19-20) : première fenêtre."""
    start = lambda s: trig_point(bank, open_quick(s._replace(dstep=_i(DS_BEFORE), ds_t=s.clk, ds_flip=jnp.bool_(False)),
                                                  s.tp, R_DS1, chained=False))
    return _cond(atk_canceled(s), battle_over, start, s)  # étape 11 : attaque annulée, pas de combat


def ds_canceled(s):
    """Attaque annulée pendant le damage step (adjust, étape 14) : attaquant ou cible déplacés (ou plus sur le
    terrain), attaquant en défense."""
    a, t = s.atk, jnp.maximum(s.atk_tgt, 0)
    gone = lambda c: (s.zone[c] != Z_MZONE) | (s.mv[c] > s.ds_t)
    return gone(a) | ((s.pos[a] & (P_FU_DEF | P_FD_DEF)) != 0) | ((s.atk_tgt >= 0) & gone(t))


def end_dcal(s):
    """reset_phase(PHASE_DAMAGE_CAL) : fin des modificateurs « jusqu'à la fin du calcul des dommages »."""
    return s._replace(tm_m=s.tm_m & (s.tm_dur != TD_DCAL)[:, None])


def ds_next(bank, s, f):
    """Suite du damage step après une fenêtre : f, ou directement la fin (étape 33) si l'attaque est annulée."""
    return _cond(ds_canceled(s), lambda s: dmg_send(bank, end_dcal(s)._replace(dstep=_i(DS_AFTER))), f, s)


def ds_flip(bank, s):
    """Étapes 21-22 : la cible verso est retournée (ses effets FLIP attendent la fin du calcul), deuxième fenêtre."""
    t = jnp.maximum(s.atk_tgt, 0)
    fd = (s.atk_tgt >= 0) & (s.pos[t] == P_FD_DEF)
    s = enable(s._replace(pos=s.pos.at[t].set(jnp.where(fd, P_FU_DEF, s.pos[t])), ds_flip=fd),
               fd & (jnp.arange(s.cid.shape[0]) == t))
    return trig_point(bank, open_quick(s, s.tp, R_DS2, chained=False))


def ds_cal(bank, s):
    """Étapes 23-24 : PHASE_DAMAGE_CAL, événement PRE_DAMAGE_CALCULATE, troisième fenêtre."""
    n = s.cid.shape[0]
    m = (jnp.arange(n) == s.atk) | ((s.atk_tgt >= 0) & (jnp.arange(n) == s.atk_tgt))
    s = raise_ev(s._replace(dstep=_i(DS_CAL)), EV_PRE_DMG, m, s.tp)
    return trig_point(bank, open_quick(s, s.tp, R_DS3, chained=False))


def ds_damage(bank, s):
    """Étapes 26-32 : calcul des dommages, puis point d'événement (effets FLIP) ; les monstres détruits restent sur
    le terrain jusqu'à `dmg_send`."""
    # pas de refresh ici : l'état est à jour en entrant (chaque pas de settle et chaque instruction finissent par un
    # refresh), et settle en refait un après. Sous vmap, cette branche de quick_end est calculée à chaque
    # pass_priority et à chaque apply : un refresh ici doublait presque le coût du moteur.
    direct = lambda s: damage(s, 1 - s.tp, s.atk_c[s.atk])
    s = s._replace(dstep=_i(DS_AFTER))
    s = _cond(s.atk_tgt < 0, direct, lambda s: battle(bank, s, s.atk, s.atk_tgt), s)
    return trig_point(bank, open_quick(end_dcal(s), s.tp, R_DMG1, chained=False))


def dmg_send(bank, s):
    """Fin du damage step : les monstres détruits au combat encore sur le terrain vont au cimetière, puis point
    d'événement (« détruit au combat », « envoyé au cimetière »)."""
    m = ((s.st & ST_BDES) != 0) & (s.zone == Z_MZONE)
    s = raise_ev(group_to_grave(s, m), EV_BATTLE_DESTROYED, m, s.tp)
    return trig_point(bank, open_quick(s, s.tp, R_DMG2, chained=False))


def battle_over(s):
    return to_battle(s._replace(atk=_i(-1), atk_tgt=_i(-1), dstep=_i(DS_NONE)))


def phase_end(s):
    """Fin d'une fenêtre de phase (deux passes de suite)."""
    def bend(s):
        # vers la End Phase : on passe quand même par Main 2 (ygopro-core simule « to_ep » depuis l'idle)
        s = s._replace(phase=_i(PH_MAIN2))
        return _cond(s.bend_ep, lambda s: open_quick(s, 1 - s.tp, R_TO_EP, passed=True, chained=False), to_idle, s)
    return _switch(s.wph, [
        lambda s: open_phase(s, WP_STANDBY, PH_STANDBY),
        lambda s: to_idle(s._replace(phase=_i(PH_MAIN1))),
        to_battle, bend, end_phase], clear_ev(adjust(s)))


def quick_end(bank, s):
    """Fin d'une fenêtre W_QUICK sans chaîne."""
    ch = s.chained
    sel = lambda a, b: lambda s: _cond(ch, a, b, s)
    return _switch(s.ret, [
        to_idle,                                                                    # R_IDLE
        to_battle,                                                                  # R_BATTLE
        sel(to_idle, lambda s: open_phase(s, WP_BSTART, PH_BATTLE)),                # R_TO_BP
        sel(to_idle, lambda s: open_phase(s, WP_END, PH_END)),                      # R_TO_EP
        sel(to_battle, lambda s: open_phase(s._replace(bend_ep=jnp.bool_(False)), WP_BEND, PH_BATTLE)),
        sel(to_battle, lambda s: open_phase(s._replace(bend_ep=jnp.bool_(True)), WP_BEND, PH_BATTLE)),
        lambda s: open_phase(s, WP_DRAW, PH_DRAW),                                  # R_DP
        lambda s: s._replace(dec=_i(D_CHAIN), dec_player=s.ph_restart, wk=_i(W_PHASE),
                             cw_passed=jnp.bool_(False)),                           # R_PHASE
        lambda s: _cond(atk_canceled(s), battle_over,                               # R_ATK1 (étape 9)
                        lambda s: open_quick(s, s.tp, R_ATK2, chained=False), s),
        lambda s: damage_step(bank, s),  # R_ATK2 (après une chaîne : la fenêtre de SolveChain, puis le damage step)
        lambda s: dmg_send(bank, s),                                                # R_DMG1
        battle_over,                                                                # R_DMG2
        lambda s: ds_next(bank, s, lambda s: ds_flip(bank, s)),                     # R_DS1
        lambda s: ds_next(bank, s, lambda s: ds_cal(bank, s)),                      # R_DS2
        lambda s: ds_next(bank, s, lambda s: ds_damage(bank, s)),                   # R_DS3
    ], clear_ev(s))


def pass_priority(bank, s):
    p = s.dec_player
    other = lambda s: s._replace(cw_passed=jnp.bool_(True), dec_player=1 - p)

    def close(s):
        return _cond(s.wk == W_PHASE, phase_end,
                        lambda s: _cond(s.n_chain > 0, lambda s: load_link(s, s.n_chain - 1),
                                           lambda s: quick_end(bank, s), s), s)
    return _cond(s.cw_passed, close, other, s)


def place_pending(bank, s, z):
    c = s.pend
    s = leave_hand(s, c)
    pos = jnp.where(s.pend_set, P_FD_DEF, P_FU_ATK)
    s = move(s, c, Z_MZONE, z, s.tp, pos)
    s = s._replace(st=s.st.at[c].set(ST_SUMMON_TURN), ns_used=jnp.bool_(True), pend=_i(-1))
    s = _cond(s.pend_set, lambda s: s, lambda s: raise_ev(s, EV_SUMMON, jnp.arange(s.cid.shape[0]) == c, s.tp), s)
    return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))


def battle(bank, s, a, t):
    """Combat a contre t (déjà retourné par ds_flip). Les détruits sont marqués ST_BDES."""
    idx = jnp.arange(s.cid.shape[0])
    s = raise_ev(s, EV_FLIP, (idx == t) & s.ds_flip, s.tp)
    A, T, D = s.atk_c[a], s.atk_c[t], s.df_c[t]
    pa, pt = s.ctrl[a], s.ctrl[t]
    atk_pos = s.pos[t] == P_FU_ATK
    kill_t = jnp.where(atk_pos, (A > T) | ((A == T) & (A > 0)), A > D)
    kill_a = atk_pos & ((T > A) | ((A == T) & (A > 0)))
    s = damage(s, pt, jnp.where(atk_pos & (A > T), A - T, 0))
    s = damage(s, pa, jnp.where(atk_pos, jnp.maximum(T - A, 0), jnp.maximum(D - A, 0)))
    kill_t &= A < s.ib_c[t]  # INDESTRUCTABLE_BATTLE (valeur : ATK de l'adversaire)
    kill_a &= T < s.ib_c[a]
    kill = ((idx == t) & kill_t) | ((idx == a) & kill_a)
    return s._replace(st=jnp.where(kill, s.st | ST_BDES, s.st))


# ---------------------------------------------------------------- interpréteur d'effets

def speed(bank, cid, e):
    """Vitesse de sort : 1 (magie normale, ignition), 2 (jeu-rapide, piège, effet rapide de monstre), 3 (contre-piège)."""
    ct = bank.ctype[cid]
    act = bank.ekind[cid, e] == K_ACTIVATE
    fast = ((ct & TYPE_TRAP) != 0) | ((ct & TYPE_QUICKPLAY) != 0)
    quick = bank.ekind[cid, e] == K_QUICK
    return jnp.where(act & fast, jnp.where((ct & TYPE_COUNTER) != 0, 3, 2), jnp.where(quick, 2, 1)).astype(I32)


def set_match(bank, cid, sc):
    """La carte cid est de l'archétype sc (règle d'ygopro : même archétype, sc ⊂ sous-archétype de la carte)."""
    cs = bank.setcode[cid]
    return ((cs != 0) & ((cs & 0xFFF) == (sc & 0xFFF)) & ((cs & sc) == sc)).any(-1)


def filter_mask(bank, s, f, p, h, bdes=True, targ=False):
    """Évalue un filtre (enregistrement fixe) pour le joueur p, carte source h. O(N).
    bdes : exclut les monstres détruits au combat (sélections) ; False pour les cibles des effets continus.
    targ : sélection avec ciblage (IsCanBeEffectTarget) : exclut les cartes qui ne peuvent pas être ciblées."""
    n = s.cid.shape[0]
    loc = jnp.where(s.ctrl == p, f[F_LOC_ME], f[F_LOC_OP])
    m = (s.zone & loc) != 0
    if bdes:
        m &= (s.st & ST_BDES) == 0  # détruite au combat : plus ciblable (ygopro is_capable_be_effect_target)
    m &= (f[F_POS] == 0) | ((s.pos & f[F_POS]) != 0)
    m &= (f[F_TYPE] == 0) | ((bank.ctype[s.cid] & f[F_TYPE]) != 0)
    m &= ~((f[F_NOT_SELF] != 0) & (jnp.arange(n) == h))
    atk, df = s.atk_c, s.df_c
    m &= (f[F_ATTR] == 0) | ((bank.attr[s.cid] & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((bank.race[s.cid] & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match(bank, s.cid, f[F_SET])
    code = bank.code[s.cid]
    m &= (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2])
    m &= (f[F_ST] == 0) | ((s.st & f[F_ST]) != 0)
    lv = bank.level[s.cid]
    m &= (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI])
    # IsCanBeSpecialSummoned : pas d'interdiction, et une carte à réanimation limitée doit avoir été bien invoquée
    m &= (f[F_SPSUM] == 0) | (((s.fl_c & FL_NO_SS) == 0) & ((bank.erl[s.cid] == 0) | s.proc))
    m &= (f[F_FUS] == 0) | s.fz_c
    m &= (f[F_RIT] == 0) | (((s.rz_c >> jnp.maximum(f[F_RIT] - 1, 0)) & 1) != 0)
    m &= (f[F_NODIS] == 0) | ((s.fl_c & FL_DISABLED) == 0)
    if targ:
        m &= ~untargetable(bank, s, h)
    return m & (atk >= f[F_ATK_LO]) & (atk <= f[F_ATK_HI]) & (df >= f[F_DEF_LO]) & (df <= f[F_DEF_HI])


def untargetable(bank, s, h):
    """[N] : cartes que l'effet de la carte h ne peut pas cibler (is_capable_be_effect_target : CANNOT_BE_EFFECT_TARGET ;
    l'immunité n'empêche pas le ciblage, l'effet ne s'applique simplement pas à la résolution)."""
    return (s.fl_c & FL_NO_ETARGET) != 0


def is_trap(bank, s, h):
    """L'effet de la carte h est un effet de piège (IsTrapEffect)."""
    return (bank.ctype[s.cid[h]] & TYPE_TRAP) != 0


def stats(bank, s):
    """ATK / DEF courantes, ATK de base courante (card::get_attack / get_defense d'ygopro) : les modificateurs sont
    appliqués dans l'ordre des id d'effet d'ygopro. Un effet continu reçoit un id à l'activation de sa carte (en_t) et
    quand sa condition redevient vraie (cav_t) ; un modificateur temporaire à sa création (tm_t). Hors zone monstres :
    stats de base. O((KM + TM) × N). Rend aussi (cav, cav_t) mis à jour, l'horloge et le débordement de KM."""
    n = s.cid.shape[0]
    idx = jnp.arange(n)
    onf = s.zone == Z_MZONE
    faceup = (s.pos & P_FACEUP) != 0
    # effets continus en place (effect::is_available avant la condition) : carte recto dans ses zones, magie/piège
    # résolue ; un effet FIELD d'un monstre détruit au combat ne s'applique plus
    k = bank.ekind[s.cid]                                          # [N, E]
    md = bank.emod[s.cid]                                          # [N, E, MW]
    srcok = faceup & ((s.zone == Z_MZONE) | ((s.zone == Z_SZONE) & ((s.st & ST_ON) != 0)))
    bdok = (md[..., M_SELF] != 0) | ((s.st & ST_BDES) == 0)[:, None]
    act = (k == K_CONT) & ((bank.erng[s.cid] & s.zone[:, None]) != 0) & srcok[:, None] & bdok
    flat = jnp.nonzero(act.reshape(-1), size=KM, fill_value=0)[0]  # ordre carte puis effet
    hs, es = flat // E, flat % E
    live = jnp.arange(KM) < act.sum()

    def one(h, e, ok):
        f = bank.filt[s.cid[h], e]
        m = bank.emod[s.cid[h], e]
        p = s.ctrl[h]
        cnt = filter_mask(bank, s, f[1], p, h, bdes=False).sum()  # filtre 1 : compte (M_PER) ou condition (M_CNT)
        cond = (((m[M_TP] == 0) | (s.tp == jnp.where(m[M_TP] == 1, p, 1 - p)))
                & ((m[M_PH] == 0) | ((m[M_PH] & s.phase) != 0))
                & ((m[M_CNT] == 0) | ((cnt >= m[M_CMIN]) & (cnt <= m[M_CMAX]))))
        ok &= cond
        raw = jnp.where(m[M_SELF] != 0, idx == h, filter_mask(bank, s, f[0], p, h, bdes=False))
        tgt = raw & (faceup | (m[M_SELF] != 0))
        val = m[M_VAL] * jnp.where(m[M_PER] != 0, cnt, 1)
        return (tgt & onf & ok, jnp.where(ok, m[M_CODE], MC_NONE), val, m[M_SELF] != 0, cond, raw & ok, p,
                is_trap(bank, s, h))
    ctgt, ccode, cval, cself, ccond, craw, cp, ctrap = jax.vmap(one)(hs, es, live)
    ttgt = s.tm_m & (s.mv <= s.tm_t[:, None]) & onf
    # annulations d'abord (EFFECT_DISABLE) : temporaires, puis continues (Skill Drain) des cartes non annulées, qui ne
    # touchent pas un monstre immunisé ; une carte annulée n'applique plus ses effets continus
    dis_t = (ttgt & (s.tm_code == MC_DISABLE)[:, None]).any(0)
    # l'immunité (SINGLE_RANGE) d'une carte qui vient d'arriver ou qui était annulée ne compte pas encore
    # (refresh_disable_status avant STATUS_EFFECT_ENABLED, et is_available exclut une carte annulée) : historique
    prev_ok = ((s.fl_c & FL_ONF) != 0) & ((s.fl_c & FL_DISABLED) == 0)
    imm0 = (ctgt & ((ccode == MC_IMM_TRAP) & ~dis_t[hs] & prev_ok[hs])[:, None]).any(0)
    dis = dis_t | (ctgt & ((ccode == MC_DISABLE) & ~dis_t[hs])[:, None] & ~(ctrap[:, None] & imm0[None, :])).any(0)
    off = dis[hs] & (ccode != MC_DISABLE)
    ctgt &= ~off[:, None]
    craw &= ~off[:, None]
    ccode = jnp.where(off, MC_NONE, ccode)
    tm_m = s.tm_m & ~(s.tm_rd[:, None] & dis[None, :])  # RESET_DISABLE
    ttgt &= tm_m
    # immunité : les effets continus de piège ne s'appliquent pas aux monstres immunisés
    imm = (ctgt & (ccode == MC_IMM_TRAP)[:, None]).any(0)
    ctgt &= ~(ctrap[:, None] & imm[None, :] & (ccode != MC_IMM_TRAP)[:, None] & (ccode != MC_DISABLE)[:, None])
    craw &= ~(ctrap[:, None] & imm[None, :] & onf[None, :])
    fl = (jnp.where(dis, FL_DISABLED, 0) | jnp.where(onf & faceup, FL_ONF, 0)).astype(I32)
    for c, b in ((MC_NO_ATTACK, FL_NO_ATTACK), (MC_NO_BTARGET, FL_NO_BTARGET), (MC_NO_ETARGET, FL_NO_ETARGET),
                 (MC_IMM_TRAP, FL_IMM_TRAP)):
        fl |= jnp.where((ctgt & (ccode == c)[:, None]).any(0), b, 0)
    fl |= jnp.where((craw & (ccode == MC_NO_SS)[:, None]).any(0), FL_NO_SS, 0)  # carte visée, où qu'elle soit
    ib = jnp.where(ctgt & (ccode == MC_INDES_BATTLE)[:, None], cval[:, None], BIG).min(0)
    nt = ccode == MC_P_NO_TRAP
    pf = jnp.stack([(nt & ((cval & jnp.where(cp == q, 1, 2)) != 0)).any() for q in (0, 1)]) * PF_NO_TRAP
    # condition redevenue vraie : nouvel id (horloge avancée d'un cran pour tous ceux de ce calcul)
    hw = jnp.where(live, hs, n)
    new = live & ccond & ~s.cav[hs, es]
    clk = s.clk + new.any()
    cav = s.cav.at[hw, es].set(ccond, mode="drop")
    cav_t = s.cav_t.at[jnp.where(new, hs, n), es].set(clk, mode="drop")
    # entrées triées comme les id d'effet : (cibles, code, valeur [N], type SINGLE, SINGLE sans SINGLE_RANGE)
    key = jnp.concatenate([jnp.maximum(s.en_t[hs], cav_t[hs, es]) * 4 + es, s.tm_t * 4 + 3])
    o = jnp.argsort(key)
    tgt = jnp.concatenate([ctgt, ttgt])[o]
    code = jnp.concatenate([ccode, s.tm_code])[o]
    val = jnp.concatenate([jnp.broadcast_to(cval[:, None], (KM, n)), s.tm_v])[o]
    single = jnp.concatenate([cself, jnp.ones(TM, bool)])[o]
    sgl = jnp.concatenate([jnp.zeros(KM, bool), jnp.ones(TM, bool)])[o]
    K = KM + TM

    jj = jnp.arange(K)[:, None]
    sg2 = sgl[:, None]

    def last(m):  # [K, N] -> [N] : index de la dernière entrée vraie (-1 : aucune)
        return jnp.where(m.any(0), K - 1 - jnp.argmax(m[::-1], axis=0), -1)

    def at(i):  # valeur de l'entrée i pour chaque carte
        return val[jnp.maximum(i, 0), idx]

    def fold(base, upd, set_, fin, sbase):
        """Pli de card::get_attack sous forme close (parallèle sur les entrées, pas de boucle séquentielle) :
        SET / SET_FINAL temporaire / SET_BASE temporaire fixent la valeur (SET_BASE : la base) ; un SET FIELD remet
        les UPDATE temporaires à zéro, un SET_FINAL temporaire tous les UPDATE ; SET_FINAL continus appliqués à la fin ;
        SET_BASE continus avant tout."""
        t = tgt
        is_ = lambda c: t & (code == c)[:, None]
        sb_s, sb_c = is_(sbase) & sg2, is_(sbase) & ~sg2
        fin_s, fin_c = is_(fin) & sg2, is_(fin) & ~sg2
        set_m = is_(set_)
        lc = last(sb_c)
        b0 = jnp.where(lc >= 0, at(lc), jnp.maximum(base, 0))
        lb = last(sb_s)
        b = jnp.where(lb >= 0, at(lb), b0)
        lx = last(set_m | fin_s | sb_s)
        x = jnp.where((lx < 0) | sb_s[jnp.maximum(lx, 0), idx], -1, at(lx))
        r_up = last((set_m & ~single[:, None]) | fin_s)
        r_upc = last(fin_s)
        u = is_(upd)
        up = jnp.where(u & sg2 & (jj > r_up), val, 0).sum(0)
        upc = jnp.where(u & ~sg2 & (jj > r_upc), val, 0).sum(0)
        ll = last(fin_c)
        cur = jnp.maximum(jnp.where(x < 0, b, x) + up + upc, 0)
        return jnp.where(onf, jnp.maximum(jnp.where(ll >= 0, at(ll), cur), 0), base), jnp.where(onf, b, base)
    atk, batk = fold(bank.atk[s.cid], MC_ATK, MC_SET_ATK, MC_FINAL_ATK, MC_BASE_ATK)
    df, _ = fold(bank.df[s.cid], MC_DEF, MC_SET_DEF, MC_FINAL_DEF, MC_BASE_DEF)
    return (atk.astype(I32), df.astype(I32), batk.astype(I32), fl, ib.astype(I32), pf.astype(I32), cav, cav_t, clk,
            tm_m, act.sum() > KM)


def stats_key(s):
    """Entrées de stats() qui changent pendant la partie (hors ses propres sorties) : si elles n'ont pas bougé, refresh
    est inutile (il est idempotent sur un état inchangé)."""
    return (s.zone, s.pos, s.st, s.ctrl, s.phase, s.tp, s.turn, s.clk, s.tm_n, s.tm_m)


def changed(prev, s):
    """[...] : les entrées de stats() ont changé entre prev et s (par partie si les états sont en lot)."""
    nb = s.done.shape
    d = jnp.zeros(nb, bool)
    for x, y in zip(stats_key(prev), stats_key(s)):
        d |= (x != y).reshape(nb + (-1,)).any(-1)
    return d


def refresh_if(bank, prev, s):
    """refresh seulement si l'état a changé depuis prev (une partie ; sous vmap, le cond calcule les deux branches)."""
    return lax.cond(changed(prev, s), lambda s: refresh(bank, s), lambda s: s, s)


def refresh_b(bank, prev, s, frac=8):
    """Version lot : seules les parties dont l'état a changé sont recalculées, regroupées dans un sous-lot de B/frac
    (en pratique quelques % des parties changent à chaque pas) ; s'il y en a plus, tout le lot est recalculé."""
    nb = s.done.shape[0]
    d = changed(prev, s)
    k = max(1, nb // frac)

    def ref(s):  # refresh d'un lot ; fusion_ok seulement si une partie du lot a des monstres fusion à l'Extra Deck
        fus = jax.vmap(lambda x: has_fusion(bank, x))(s).any()
        s = lax.cond(fus, lambda s: jax.vmap(lambda x: refresh_fus(bank, x))(s), lambda s: s, s)
        rit = jax.vmap(lambda x: has_ritual(bank, x))(s).any()
        s = lax.cond(rit, lambda s: jax.vmap(lambda x: refresh_rit(bank, x))(s), lambda s: s._replace(rz_c=s.rz_c * 0), s)
        return jax.vmap(lambda x: refresh(bank, x, fus=False))(s)

    def sub(s):  # tout l'état est réécrit : ne réécrire que les champs de refresh est plus lent (mesuré sur CPU)
        idx = jnp.nonzero(d, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda x: x[jnp.minimum(idx, nb - 1)], s))
        return jax.tree.map(lambda x, y: x.at[idx].set(y, mode="drop"), s, part)
    return lax.cond(d.sum() > k, ref, lambda s: lax.cond(d.any(), sub, lambda s: s, s), s)


SUBS = [[(q >> j) & 1 for j in range(FK)] for q in range(1 << FK)]  # sous-ensembles des emplacements de matériaux


MAT_FIELDS = (F_TYPE, F_ATTR, F_RACE, F_SET, F_CODE1, F_CODE2, F_LV_LO, F_LV_HI)  # champs permis dans une recette


def mat_mask(bank, s, f):
    """Filtre de matériau allégé (seulement les champs MAT_FIELDS, sans la zone) : moins cher que filter_mask."""
    ct, code, lv = bank.ctype[s.cid], bank.code[s.cid], bank.level[s.cid]
    m = (f[F_TYPE] == 0) | ((ct & f[F_TYPE]) != 0)
    m &= (f[F_ATTR] == 0) | ((bank.attr[s.cid] & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((bank.race[s.cid] & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match(bank, s.cid, f[F_SET])
    m &= (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2])
    return m & (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI])


def fus_cands(bank, s, c):
    """[FK, N] : cartes utilisables pour chaque matériau du monstre fusion c (main et terrain de son propriétaire,
    monstres, pas détruits au combat)."""
    p = s.owner[c]
    mats = (((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & (s.ctrl == p) & ((bank.ctype[s.cid] & TYPE_MONSTER) != 0)
            & ((s.st & ST_BDES) == 0))
    k = bank.fmn[s.cid[c]]
    return jnp.stack([mat_mask(bank, s, bank.fmat[s.cid[c], j]) & mats & (j < k) for j in range(FK)])


def fusion_ok(bank, s):
    """[N] fz_c : monstres fusion de l'Extra Deck réalisables (matériaux distincts sur la main et le terrain, et une case
    pour l'invocation : une case libre, ou un matériau sur le terrain qui en libérera une). Chaque carte est résumée
    par le masque des emplacements qu'elle peut remplir ; le test de Hall se fait sur l'histogramme de ces masques
    (main / terrain), en O(2^FK × 2^FK) une fois l'histogramme fait. O(EXK × (FK + 1) × N)."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (bank.fmn[s.cid] > 0)
    idx = jnp.nonzero(want, size=EXK, fill_value=0)[0]
    live = jnp.arange(EXK) < want.sum()
    nq = 1 << FK
    q = jnp.arange(nq)
    qsize = jnp.asarray([bin(x).count("1") for x in range(nq)])
    field = s.zone == Z_MZONE

    def one(c, ok):
        cand = fus_cands(bank, s, c)                               # [FK, N]
        bits = (cand * (1 << jnp.arange(FK))[:, None]).sum(0)      # [N] emplacements possibles de chaque carte
        hist = jnp.zeros(2 * nq, I32).at[bits + nq * field].add(1)  # [main | terrain]
        hh, hf = hist[:nq], hist[nq:]
        k = bank.fmn[s.cid[c]]
        use = (1 << k) - 1
        inter = (q[:, None] & q[None, :]) != 0                     # [valeur, Q] : la carte sert à un emplacement de Q

        def hall_h(hh, hf):  # tout Q ⊆ use a au moins |Q| cartes candidates
            cnt = ((hh + hf)[:, None] * inter).sum(0)              # [Q]
            return ((q & ~use) != 0) | (cnt >= qsize)
        ok_all = hall_h(hh, hf).all()
        # sans case libre : un emplacement j doit être tenu par un monstre du terrain (les cartes de la main perdent j)
        with_field = jnp.zeros((), bool)
        for j in range(FK):
            hj = jnp.zeros(nq, I32).at[q & ~(1 << j)].add(hh)
            with_field |= (j < k) & hall_h(hj, hf).all()
        return ok & ok_all & (ex_zones(s, s.owner[c]).any() | with_field)
    oks = jax.vmap(one)(idx, live)
    return jnp.zeros(n, bool).at[jnp.where(live, idx, n)].set(oks, mode="drop"), want.sum() > EXK


PERMS = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)]


def fus_picks(bank, s, f):
    """[N] : matériaux qu'on peut encore choisir pour le monstre fusion f, les cartes ST_SEL étant déjà choisies : il
    existe une affectation des matériaux choisis et du candidat à des emplacements distincts, et les emplacements
    restants ont des matériaux distincts parmi les autres cartes (Hall). Toutes les permutations des FK emplacements
    sont essayées (FK! = 24). O(FK! × 2^FK × N)."""
    n = s.cid.shape[0]
    k = bank.fmn[s.cid[f]]
    cand = fus_cands(bank, s, f)                                   # [FK, N]
    sel = (s.st & ST_SEL) != 0
    t = sel.sum()
    chosen = jnp.nonzero(sel, size=FK, fill_value=0)[0]            # cartes déjà choisies, dans l'ordre des slots
    free = cand & ~sel[None, :]
    sub = jnp.asarray(SUBS, bool)                                  # [Q, FK]
    field = s.zone == Z_MZONE

    def hallq(fr):  # [Q, N] : Hall pour Q avec les candidats fr, sans la carte c
        unionm = (sub[:, :, None] & fr[None]).any(1)
        return (unionm.sum(-1)[:, None] - unionm) >= sub.sum(1)[:, None]
    hq = hallq(free)
    hqf = [hallq(free.at[j].set(free[j] & field)) for j in range(FK)]  # l'emplacement j pris sur le terrain
    # pas de case libre pour le monstre fusion : un matériau au moins doit venir du terrain
    room = ex_zones(s, s.owner[f]).any() | (sel & field).any()
    out = jnp.zeros(n, bool)
    for pm in PERMS:
        pm_ = jnp.asarray(pm)
        pos = jnp.arange(FK)
        ok_s = ((pos >= t) | cand[pm_, chosen]).all()              # les choisis vont aux premiers emplacements
        slot_c = pm_[jnp.minimum(t, FK - 1)]
        rest = jnp.zeros(FK, bool).at[pm_].set((pos > t) & (pm_ < k))
        need_q = ~(sub & ~rest[None, :]).any(1)                    # Q ⊆ emplacements restants
        ok_h = (~need_q[:, None] | hq).all(0)
        ok_f = jnp.stack([rest[j] & (~need_q[:, None] | hqf[j]).all(0) for j in range(FK)]).any(0)
        okc = ok_s & (t < k) & (slot_c < k) & cand[slot_c] & jnp.where(room | field, ok_h, ok_f)
        out |= okc
    return out & ~sel


def ex_zones(s, p):
    """[7] : cases où p peut invoquer un monstre de l'Extra Deck (MR5 sans Link : les 5 cases principales libres ; les
    zones extra 5 et 6 si p n'en occupe aucune et que l'adversaire n'occupe pas la case symétrique)."""
    me, op = zone_used(s, p, Z_MZONE), zone_used(s, 1 - p, Z_MZONE)
    emz = ~me[5] & ~me[6]
    return jnp.concatenate([~me[:5], jnp.stack([emz & ~op[6], emz & ~op[5]])])


def has_fusion(bank, s):
    """Des monstres fusion attendent dans un Extra Deck (sinon fusion_ok est inutile)."""
    return ((s.zone == Z_EXTRA) & (bank.fmn[s.cid] > 0)).any()


def refresh_fus(bank, s):
    fz, fover = fusion_ok(bank, s)
    return s._replace(fz_c=fz, ovf=s.ovf | jnp.where(fover, 4, 0).astype(I32))


def refresh_rit(bank, s):
    return s._replace(rz_c=ritual_ok(bank, s))


def rit_mats(bank, s, p):
    """[N] : matériaux de rituel de p (monstres de sa main et de son terrain, pas détruits au combat)."""
    return (((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & (s.ctrl == p) & ((bank.ctype[s.cid] & TYPE_MONSTER) != 0)
            & ((s.st & ST_BDES) == 0) & (bank.level[s.cid] > 0))


def lv_hist(bank, s, mask):
    """Histogrammes [LVMAX + 1] des niveaux des cartes mask : main, terrain."""
    lv = jnp.clip(bank.level[s.cid], 0, LVMAX)
    field = s.zone == Z_MZONE
    oh = lv[None, :] == jnp.arange(LVMAX + 1)[:, None]  # one-hot (un scatter est lent sous vmap)
    hh = (oh & (mask & ~field)[None, :]).sum(1, dtype=I32)
    hf = (oh & (mask & field)[None, :]).sum(1, dtype=I32)
    return hh, hf


def sum_ok(hh, hf, req, u, mu, has_f):
    """Les cartes déjà prises (somme u, plus petit niveau mu, une du terrain si has_f) se complètent-elles par des
    cartes des histogrammes hh (main) / hf (terrain) en un ensemble valide ? Renvoie (Greater, Equal). Greater :
    somme >= req et somme - plus petit < req ; Equal : somme == req. Avec has_f < 0 : un matériau doit venir du terrain
    (pas de case libre). Sommes atteignables suivies par masques de bits (A : sans carte du terrain, B : avec), niveaux
    de LVMAX à 1 : avec les niveaux >= v, le plus petit ajouté est >= v. Les effectifs sont pris par paquets 1, 2, 4...
    (découpage binaire). O(LVMAX × log SMAX), indépendant de N."""
    need_f = has_f < 0
    hasf0 = has_f > 0
    full = (1 << SMAX) - 1
    A, B = jnp.int32(1), jnp.int32(0)
    need = req - u
    fok = ~need_f | hasf0
    # T vide : ce qui est pris suffit déjà
    g = (need <= 0) & (u - mu < req) & fok
    e = (need == 0) & fok
    sig = jnp.arange(SMAX)
    for L in range(LVMAX, 0, -1):
        cmax = SMAX // L
        nb = cmax.bit_length()  # paquets 1, 2, ..., 2^(nb-1) : couvrent 0..cmax
        ch, cf = jnp.minimum(hh[L], cmax), jnp.minimum(hf[L], cmax)
        for i in range(nb):
            k = jnp.clip(ch - (2 ** i - 1), 0, 2 ** i)  # paquet i de la main (0 : rien)
            A = A | ((A << (k * L)) & full)
            B = B | ((B << (k * L)) & full)
        for i in range(nb):
            k = jnp.clip(cf - (2 ** i - 1), 0, 2 ** i)  # paquet i du terrain : passe en B
            B = jnp.where(k > 0, B | (((A | B) << (k * L)) & full), B)
        reach = (jnp.where(need_f & ~hasf0, B, A | B) >> sig) & 1
        win = jnp.minimum(mu, L)                                   # le plus petit de l'ensemble est min(mu, >= L)
        g |= ((reach != 0) & (sig >= need) & (sig < need + win) & (need > 0)).any()
        e |= ((reach != 0) & (sig == need) & (need > 0)).any()
    return g, e


def ritual_ok(bank, s):
    """[N] rz_c : monstres rituels en main invocables par rituel (matériaux : main et terrain, sans le monstre
    lui-même ; niveau requis : son niveau). Bit 0 : Greater ; bit 1 : Equal."""
    n = s.cid.shape[0]
    out = jnp.zeros(n, I32)
    lvc = jnp.clip(bank.level[s.cid], 0, LVMAX)
    for p in (0, 1):
        mats = rit_mats(bank, s, p)
        hh, hf = lv_hist(bank, s, mats)
        room = ex_zones(s, p)[:N_MAIN_ZONES].any()

        def cls(lv):  # le monstre rituel (niveau lv, en main) ne se libère pas lui-même
            h2 = hh - (jnp.arange(LVMAX + 1) == lv)
            g, e = sum_ok(h2, hf, lv, 0, LVMAX + 1, jnp.where(room, 0, -1))
            return g.astype(I32) | (e.astype(I32) << 1)
        res = jax.vmap(cls)(jnp.arange(LVMAX + 1))
        mine = (s.zone == Z_HAND) & (s.ctrl == p) & ((bank.ctype[s.cid] & RIT_MON) == RIT_MON)
        out = jnp.where(mine, res[lvc], out)
    return out


def rit_picks(bank, s, r, mode):
    """[N] : libérations qu'on peut encore choisir pour le monstre rituel r : il existe un ensemble valide qui
    contient les cartes déjà choisies (ST_SEL) et celle-ci. Calcul par classe (niveau, main / terrain)."""
    p = s.ctrl[r]
    sel = (s.st & ST_SEL) != 0
    mats = rit_mats(bank, s, p) & (jnp.arange(s.cid.shape[0]) != r)
    lv = jnp.clip(bank.level[s.cid], 0, LVMAX)
    field = s.zone == Z_MZONE
    hh, hf = lv_hist(bank, s, mats & ~sel)
    u0 = jnp.where(sel, lv, 0).sum()
    mu0 = jnp.where(sel, lv, LVMAX + 1).min()
    f0 = (sel & field).any()
    room = ex_zones(s, p)[:N_MAIN_ZONES].any()
    req = bank.level[s.cid[r]]

    def cls(l, isf):
        one = jnp.arange(LVMAX + 1) == l
        h2 = hh - (one & ~isf)
        f2 = hf - (one & isf)
        hasf = f0 | isf
        g, e = sum_ok(h2, f2, req, u0 + l, jnp.minimum(mu0, l), jnp.where(room | hasf, 1, -1))
        return jnp.where(mode == RIT_GREATER, g, e)
    tab = jax.vmap(lambda isf: jax.vmap(lambda l: cls(l, isf))(jnp.arange(LVMAX + 1)))(jnp.array([False, True]))
    return mats & ~sel & tab[field.astype(I32), lv]


def rit_done(bank, s, r, mode):
    """La sélection rituelle est complète (somme atteinte)."""
    sel = (s.st & ST_SEL) != 0
    return jnp.where(sel, jnp.clip(bank.level[s.cid], 0, LVMAX), 0).sum() >= bank.level[s.cid[r]]


def has_ritual(bank, s):
    return ((s.zone == Z_HAND) & ((bank.ctype[s.cid] & RIT_MON) == RIT_MON)).any()


def refresh(bank, s, fus=True):
    """Recalcule les caches (stats, drapeaux, fusions réalisables). fus=False : sans fz_c (refresh_b le fait au
    niveau du lot, seulement si une partie a des monstres fusion dans son Extra Deck)."""
    if fus:
        s = lax.cond(has_fusion(bank, s), lambda s: refresh_fus(bank, s), lambda s: s, s)
        s = lax.cond(has_ritual(bank, s), lambda s: refresh_rit(bank, s), lambda s: s._replace(rz_c=s.rz_c * 0), s)
    atk, df, batk, fl, ib, pf, cav, cav_t, clk, tm_m, over = stats(bank, s)
    return s._replace(atk_c=atk, df_c=df, batk_c=batk, fl_c=fl, ib_c=ib, pf_c=pf, cav=cav, cav_t=cav_t, clk=clk, tm_m=tm_m,
                      ovf=s.ovf | jnp.where(over, 1, 0).astype(I32))


def _pure(bank, s, reg, ok, h, e, p, k, ins):
    """Instructions sans effet de bord (requêtes et conditions), communes au mode test et à l'exécution.
    k : index du maillon de l'effet (le maillon auquel il répond est k-1).
    En mode test, CHOOSE vérifie seulement qu'il y a assez de candidats (le registre reçoit tous les candidats)."""
    op, a, b, c = ins[0], ins[1], ins[2], ins[3]
    n = s.cid.shape[0]
    ra, rb = jnp.clip(a, 0, R - 1), jnp.clip(b, 0, R - 1)
    lk = jnp.clip(k - 1, 0, CH - 1)
    lc = s.cid[jnp.maximum(s.lk_src[lk], 0)]
    # chaque branche rend (ligne [n] écrite dans reg[a] si wr, condition) : sous vmap, le switch calcule toutes
    # les branches, donc on évite qu'elles rendent chacune le registre entier
    none = jnp.zeros(n, bool)
    put = lambda row: (row, jnp.bool_(True), jnp.bool_(True))
    cond = lambda x: (none, jnp.bool_(False), x)
    branches = [lambda: cond(jnp.bool_(True))] * N_OPS
    cw = c & 7
    within = jnp.where(cw > 0, reg[jnp.clip(cw - 1, 0, R - 1)], True)  # c & 7 > 0 : intersection avec le registre
    # c & 8 : sélection avec ciblage
    branches[OP_FILTER] = lambda: put(filter_mask(bank, s, bank.filt[s.cid[h], e, jnp.clip(b, 0, NF - 1)], p, h)
                                      & within & ~(((c & 8) != 0) & untargetable(bank, s, h)))
    branches[OP_SELF] = lambda: put(jnp.arange(n) == h)
    branches[OP_REQ] = lambda: cond(reg[ra].sum() >= b)
    branches[OP_REQ_FREE] = lambda: cond(free_main(s, rel(p, a), jnp.where(b == Z_SZONE, Z_SZONE, Z_MZONE)) >= c)

    def minatk():
        atk = jnp.where(reg[rb], s.atk_c, BIG)
        return put(reg[rb] & (atk == atk.min()))
    branches[OP_MINATK] = minatk
    branches[OP_CHOOSE] = lambda: (reg[rb], jnp.bool_(True), reg[rb].sum() >= (c & 15))
    lkind = bank.ekind[lc, s.lk_eff[lk]]
    lfield = (s.zone[jnp.maximum(s.lk_src[lk], 0)] & (Z_MZONE | Z_SZONE)) != 0
    branches[OP_COND_CHAIN] = lambda: cond((k > 0) & jnp.where(
        b == 0, (lkind == K_ACTIVATE) & ((bank.ctype[lc] & a) != 0), lfield & (lkind != K_ACTIVATE)))
    branches[OP_CHAINCARD] = lambda: put((jnp.arange(n) == s.lk_src[lk]) & (k > 0)
                                         & ((s.zone & jnp.where(b == 0, Z_SZONE, b)) != 0))
    branches[OP_PAY_LP] = lambda: cond(s.lp[p] >= a)
    evm = ((a >> jnp.arange(NEV)) & 1) != 0
    evb = ((b >> jnp.arange(NEV)) & 1) != 0
    branches[OP_EVENT] = lambda: put((s.ev_g & evb[:, None]).any(axis=0))
    branches[OP_COND_TURN] = lambda: cond(s.tp == rel(p, a))
    branches[OP_COND_EVP] = lambda: cond((evm & s.ev_g.any(axis=1) & (s.ev_p == rel(p, b))).any())
    branches[OP_COND_SELF] = lambda: cond(((a == 0) | ((s.zone[h] & a) != 0)) & ((b == 0) | ((s.pzone[h] & b) != 0)))
    branches[OP_COND_PHASE] = lambda: cond((s.phase & a) != 0)
    branches[OP_COND_ATK] = lambda: cond(jnp.where(a == 0, s.atk, s.atk_tgt) == h)  # la carte attaque / est attaquée
    branches[OP_ATTACKER] = lambda: put((jnp.arange(n) == s.atk) & (s.atk >= 0) & ((s.st & ST_BDES) == 0)
                                        & ~((b != 0) & untargetable(bank, s, h)))  # b : ciblage
    row, wr, c_ok = _switch(op, [lambda _, f=f: f() for f in branches], None)
    return jnp.where(wr, reg.at[ra].set(row), reg), ok & c_ok


def test_ok(bank, s, h, e, p):
    """Mode test (chk==0 des scripts) : coût + cible exécutables par p avec la carte h, effet e ?"""
    n = s.cid.shape[0]
    progs = bank.tprog[s.cid[h], e]

    def body(carry, ins):
        reg, ok = carry
        return _pure(bank, s, reg, ok, h, e, p, s.n_chain, ins), None
    (_, ok), _ = lax.scan(body, (jnp.zeros((R, n), bool), jnp.bool_(True)), progs)
    return ok


def current_ins(bank, s):
    return bank.prog[s.cid[s.ch_src], s.ch_eff, s.pg, s.pc]


def begin_act(bank, s, c, e):
    """Activation de l'effet e de c par le joueur qui décide : coût puis cible, puis mise en chaîne."""
    n = s.cid.shape[0]
    leave = (bank.ekind[s.cid[c], e] == K_ACTIVATE) & ((bank.ctype[s.cid[c]] & TYPE_CONTINUOUS) == 0)
    p = s.dec_player
    opt = bank.eopt[s.cid[c], e]
    s = s._replace(opt_cid=s.opt_cid.at[p].set(jnp.where(opt == OPT_NAME, name_add(s.opt_cid[p], s.cid[c]), s.opt_cid[p])),
                   st=s.st.at[c].set(s.st[c] | jnp.where(opt == OPT_CARD, ST_CNT << e, jnp.where(opt == OPT_CARD_NR, ST_CNTN << e, 0))))
    # AddChain : une carte du terrain sous une annulation temporaire (Forbidden Chalice, Effect Veiler :
    # EFFECT_DISABLE_EFFECT) voit son maillon annulé, même si elle quitte ensuite le terrain (avant le coût)
    tdis = (s.tm_m[:, c] & (s.mv[c] <= s.tm_t) & (s.tm_code == MC_DISABLE)).any() & (s.zone[c] == Z_MZONE)
    return s._replace(st=s.st.at[c].set(jnp.where(leave, s.st[c] | ST_LEAVE, s.st[c])), ch_dis=tdis,
                      ch_src=_i(c), ch_eff=_i(e), ch_p=s.dec_player, ch_k=s.n_chain, pg=_i(PG_COST), pc=_i(0),
                      reg=jnp.zeros((R, n), bool), dec=_i(D_RUN), pend=_i(-1))


def push_link(bank, s):
    s = adjust(s)  # ygopro : break_effect après coût et cible (AddChain étape 8)
    k, src = s.n_chain, s.ch_src
    s = s._replace(lk_src=s.lk_src.at[k].set(s.ch_src), lk_eff=s.lk_eff.at[k].set(s.ch_eff),
                   lk_p=s.lk_p.at[k].set(s.ch_p), lk_reg=s.lk_reg.at[k].set(s.reg),
                   lk_neg=s.lk_neg.at[k].set(s.ch_dis), lk_t=s.lk_t.at[k].set(s.clk), n_chain=k + 1, ch_src=_i(-1))
    return _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s._replace(tg=s.tg.at[src].set(False))),
                    lambda s: open_quick(s, 1 - s.ch_p), s)


def load_link(s, k):
    """Résolution du maillon k ; un maillon nié saute directement à la fin de son programme, comme celui d'un
    monstre dont les effets sont annulés et qui est encore lié au maillon (solve_chain : STATUS_DISABLED)."""
    c = jnp.maximum(s.lk_src[k], 0)
    dis = ((s.fl_c[c] & FL_DISABLED) != 0) & (s.zone[c] == Z_MZONE) & (s.mv[c] <= s.lk_t[k])
    return s._replace(ch_k=k, ch_src=s.lk_src[k], ch_eff=s.lk_eff[k], ch_p=s.lk_p[k], reg=s.lk_reg[k] & (s.mv <= s.lk_t[k]),
                      pg=_i(PG_RESOLVE), pc=jnp.where(s.lk_neg[k] | dis, P - 1, 0).astype(I32), dec=_i(D_RUN),
                      ev_g=jnp.zeros_like(s.ev_g))


def chain_end(bank, s):
    """Chaîne résolue : les magies/pièges activés encore sur le terrain partent ensemble au cimetière, puis
    nouvelle fenêtre pour le joueur du tour."""
    n = s.cid.shape[0]
    s = group_to_grave(s, (s.st & ST_LEAVE) != 0)
    s = s._replace(n_chain=_i(0), ch_src=_i(-1), ch_k=_i(0), pg=_i(0), pc=_i(0), reg=jnp.zeros((R, n), bool))
    return trig_point(bank, open_quick(s, s.tp, chained=True))


def exec_one(bank, s):
    """Exécute une instruction de l'effet en cours (état D_RUN)."""
    h, e, p = s.ch_src, s.ch_eff, s.ch_p
    ins = current_ins(bank, s)
    op, a, b, c = ins[0], ins[1], ins[2], ins[3]
    ra = jnp.clip(a, 0, R - 1)
    adv = lambda s: s._replace(pc=s.pc + 1)
    # résolution d'un effet de piège : les monstres immunisés ne sont ni déplacés ni modifiés (is_affect_by_effect)
    imm = (s.pg == PG_RESOLVE) & is_trap(bank, s, h) & ((s.fl_c & FL_IMM_TRAP) != 0) & (s.zone == Z_MZONE)
    target = lambda s: s.reg[ra] & ~imm

    def next_link(s):
        # maillon résolu (non nié) : les effets continus de sa carte s'appliquent (enable_field_effect)
        on = ~s.lk_neg[s.ch_k]
        s = enable(s._replace(st=s.st.at[h].set(jnp.where(on, s.st[h] | ST_ON, s.st[h]))),
                   on & (jnp.arange(s.cid.shape[0]) == h))
        return _cond(s.ch_k > 0, lambda s: load_link(s, s.ch_k - 1), lambda s: chain_end(bank, s), s)

    def end(s):
        return _switch(s.pg, [lambda s: s._replace(pg=_i(PG_TARGET), pc=_i(0)), lambda s: push_link(bank, s),
                                 next_link], s)

    def pure(s):
        reg, ok = _pure(bank, s, s.reg, jnp.bool_(True), h, e, p, s.ch_k, ins)
        return _cond(ok, adv, end, s._replace(reg=reg))  # condition fausse à la résolution : l'effet s'arrête

    def choose(s):  # au plus (c >> 4) & 15 cartes, moins s'il n'y a pas assez de candidats (rien : pas de décision)
        need = jnp.minimum((c >> 4) & 15, s.reg[jnp.clip(b, 0, R - 1)].sum()).astype(I32)
        return _cond(need > 0,
                        lambda s: s._replace(dec=_i(D_SELECT), dec_player=rel(p, (c >> 8) & 1), need=need, picks=_i(0)),
                        lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def move_(s):  # vers le cimetière (détruire, libérer, défausser, envoyer), la main du propriétaire ou les bannies
        return adv(_switch(jnp.where(b == Z_HAND, 1, jnp.where(b == Z_REMOVED, 2, 0)),
                              [lambda s: group_to_grave(s, target(s)),
                               lambda s: group_to_hand(bank, s, target(s)),
                               lambda s: group_send(s, target(s), Z_REMOVED, P_GRAVE)], s))

    def fusmat(s):  # matériaux du monstre fusion du registre b, choisis un par un par p (D_SELECT), rangés dans a
        f = jnp.argmax(s.reg[jnp.clip(b, 0, R - 1)])
        go = s.reg[jnp.clip(b, 0, R - 1)].any()
        return _cond(go, lambda s: s._replace(dec=_i(D_SELECT), dec_player=p, need=bank.fmn[s.cid[f]], picks=_i(0),
                                                fsel=jnp.bool_(True)),
                     lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def ritmat(s):  # libérations du monstre rituel du registre b (mode c), choisies une par une par p, rangées dans a
        go = s.reg[jnp.clip(b, 0, R - 1)].any()
        return _cond(go, lambda s: s._replace(dec=_i(D_SELECT), dec_player=p, need=_i(99), picks=_i(0), rsel=c),
                     lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def spsummon(s):
        return _cond(s.reg[ra].any(), lambda s: s._replace(dec=_i(D_ZONE), zctx=_i(ZC_SPSUM), dec_player=p), adv, s)

    def lock(s):  # p ne peut plus activer les effets des cartes de ce nom ce tour
        return adv(_cond(s.reg[ra].any(), lambda s: s._replace(
            lock_cid=s.lock_cid.at[p].set(name_add(s.lock_cid[p], s.cid[jnp.argmax(s.reg[ra])]))), lambda s: s, s))

    def draw_n(s):
        q = rel(p, a)
        # sous vmap, le switch évalue toutes les branches avec le b de l'instruction courante (1000 pour DAMAGE...) :
        # le nombre de tours de boucle est celui de la partie qui en demande le plus, d'où le masque sur op
        nb = jnp.where(op == OP_DRAW, jnp.clip(b, 0, s.cid.shape[0]), 0)
        return adv(lax.fori_loop(0, nb, lambda i, s: draw(s, q), s))

    def modify(s):  # modificateur temporaire sur les cartes du registre a : b = MC_* | MV_* << 5 | TD_* << 9 | RD << 13
        j = jnp.minimum(s.tm_n, TM - 1)
        m = target(s) & (s.zone == Z_MZONE)
        kill = ((jnp.asarray(MC_KILL, I32)[b & 31] >> s.tm_code) & 1) != 0
        s = s._replace(tm_m=s.tm_m & ~(kill[:, None] & m[None, :]))
        mode = (b >> 5) & 15
        # Lua x/2 est un flottant, arrondi par std::round dans effect::get_value : moitié supérieure
        v = jnp.where(mode == MV_HALF_BASE_ATK, (s.batk_c + 1) // 2, jnp.where(mode == MV_HALF_ATK, (s.atk_c + 1) // 2, c))
        return adv(s._replace(tm_code=s.tm_code.at[j].set(b & 31), tm_m=s.tm_m.at[j].set(m),
                              tm_rd=s.tm_rd.at[j].set(((b >> 13) & 1) != 0),
                              tm_v=s.tm_v.at[j].set(jnp.where(m, v, 0)), tm_t=s.tm_t.at[j].set(s.clk + 1), clk=s.clk + 1,
                              tm_dur=s.tm_dur.at[j].set((b >> 9) & 15),
                              tm_n=jnp.minimum(s.tm_n + 1, TM), ovf=s.ovf | jnp.where(s.tm_n >= TM, 2, 0).astype(I32)))

    def damage_(s):
        mul = jnp.where(c > 0, s.reg[jnp.clip(c - 1, 0, R - 1)].sum(), 1)
        return adv(damage(s, rel(p, a), b * mul))

    branches = [lambda s: s] * N_OPS
    branches[OP_END] = end
    for o in (OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_COND_CHAIN, OP_CHAINCARD, OP_EVENT, OP_COND_TURN,
              OP_COND_EVP, OP_COND_SELF, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE):
        branches[o] = pure
    branches[OP_CHOOSE] = choose
    branches[OP_LOCK] = lock
    branches[OP_MODIFY] = modify
    branches[OP_FUSMAT] = fusmat
    branches[OP_RITMAT] = ritmat
    branches[OP_MOVE] = move_
    branches[OP_SPSUMMON] = spsummon
    branches[OP_DRAW] = draw_n
    branches[OP_DAMAGE] = damage_
    branches[OP_RECOVER] = lambda s: adv(s._replace(lp=s.lp.at[rel(p, a)].add(b)))
    branches[OP_SHUFFLE_HAND] = adv  # l'ordre de la main n'est pas observé : rien à faire
    branches[OP_NEGATE] = lambda s: adv(s._replace(lk_neg=s.lk_neg.at[jnp.clip(s.ch_k - 1, 0, CH - 1)].set(True)))
    branches[OP_PAY_LP] = lambda s: adv(damage(s, p, a))
    return _switch(op, branches, s)


# ---------------------------------------------------------------- actions légales

def act_cand(bank, s):
    """[E, N] : effets que le joueur qui décide pourrait activer, avant le test du coût et de la cible (act_ok)."""
    p = s.dec_player
    mine = s.ctrl == p
    ct = bank.ctype[s.cid]
    spell, trap = (ct & TYPE_SPELL) != 0, (ct & TYPE_TRAP) != 0
    quick = spell & ((ct & TYPE_QUICKPLAY) != 0)
    in_hand = mine & (s.zone == Z_HAND)
    my_mon = mine & (s.zone == Z_MZONE)
    free_s = free_main(s, p, Z_SZONE)
    idle, bat, chn = s.dec == D_IDLE, s.dec == D_BATTLE, s.dec == D_CHAIN
    set_st = mine & (s.zone == Z_SZONE) & (s.pos == P_SFD)
    ready = set_st & ((s.st & ST_SET_TURN) == 0)       # posée avant ce tour
    from_hand = in_hand & (free_s > 0)
    faceup_mon = my_mon & ((s.pos & P_FACEUP) != 0)
    trig_w = chn & (s.wk == W_TRIG)
    fast = bat | (chn & ~trig_w)                       # vitesse 2 minimum
    top = jnp.clip(s.n_chain - 1, 0, CH - 1)
    top_spd = jnp.where(s.n_chain > 0, speed(bank, s.cid[jnp.maximum(s.lk_src[top], 0)], s.lk_eff[top]), 0)
    evb = ev_bits(s)
    tkind = jnp.where(s.tst < 2, K_TRIG_F, K_TRIG_O)
    cand = []
    for e in range(E):
        kind = bank.ekind[s.cid, e]
        trig = bank.etrig[s.cid, e]
        act = kind == K_ACTIVATE
        rng = bank.erng[s.cid, e]
        qm = (kind == K_QUICK) & ((in_hand & ((rng & Z_HAND) != 0)) | (faceup_mon & ((rng & Z_MZONE) != 0)))
        where = ((idle & spell & ~quick & act & (from_hand | set_st))
                 | ((idle | fast) & qm)
                 | ((idle | fast) & quick & act & ((from_hand & (p == s.tp)) | ready))
                 | ((idle | fast) & trap & act & ready & ((s.pf_c[p] & PF_NO_TRAP) == 0))
                 | (idle & faceup_mon & (kind == K_IGNITION)))
        where &= speed(bank, s.cid, e) >= top_spd
        where &= (trig == 0) | ((trig & evb) != 0)
        # damage step (effect::is_activateable) : drapeaux DAMAGE_STEP / DAMAGE_CAL, sauf contre-pièges et effets
        # sur un événement de combat ; après le calcul, la condition StatChangeDamageStepCondition interdit
        dmg = bank.edmg[s.cid, e]
        dsok = jnp.where(s.dstep == DS_BEFORE, (dmg & DMG_STEP) != 0,
                         jnp.where(s.dstep == DS_CAL, (dmg & DMG_CAL) != 0,
                                   ((dmg & DMG_STEP) != 0) & ((dmg & DMG_STAT) == 0)))
        dsok |= (s.dstep == DS_NONE) | (act & ((ct & TYPE_COUNTER) != 0)) | ((trig & (1 << EV_PRE_DMG)) != 0)
        where &= dsok
        where |= trig_w & mine & s.tg & (s.tg_e == e) & (kind == tkind)  # déclencheurs de l'étape SEGOC en cours
        cand.append(where)
    return jnp.stack(cand)


def auto_pass(bank, s):
    """Fenêtre de réponse où le joueur ne peut rien activer : passer est la seule action (jouée par settle, sans
    passer par legal + apply). Les fenêtres W_TRIG ne sont ouvertes que s'il y a un déclencheur activable."""
    return (s.dec == D_CHAIN) & (s.wk != W_TRIG) & ~s.done & ~act_cand(bank, s).any()


def legal(bank, s, fus=True):
    """Masque des actions légales. fus=False : sans le choix des matériaux de fusion (step_batch ne le calcule que si
    une partie du lot en est là)."""
    n = s.cid.shape[0]
    p = s.dec_player
    mine = s.ctrl == p
    ct = bank.ctype[s.cid]
    monster, spell, trap = (ct & TYPE_MONSTER) != 0, (ct & TYPE_SPELL) != 0, (ct & TYPE_TRAP) != 0
    in_hand = mine & (s.zone == Z_HAND)
    my_mon = mine & (s.zone == Z_MZONE)
    op_mon = (s.ctrl == 1 - p) & (s.zone == Z_MZONE)
    free_m, free_s = free_main(s, p), free_main(s, p, Z_SZONE)
    nmon = my_mon.sum()
    lv = bank.level[s.cid]
    need = jnp.where(lv >= 7, 2, jnp.where(lv >= 5, 1, 0))
    # monstres rituels et de l'Extra Deck : pas d'invocation normale
    can_ns = (in_hand & monster & ~s.ns_used & jnp.where(need == 0, free_m > 0, nmon >= need)
              & ((ct & (TYPE_RITUAL | TYPE_EXTRA)) == 0))
    can_repos = my_mon & ((s.st & (ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN)) == 0)
    unsel = (s.st & ST_SEL) == 0
    # get_attack_target (select_target vrai par défaut) : au moins une cible possible, ou attaque directe
    atk_ok = ~op_mon.any() | (op_mon & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    can_atk = my_mon & (s.pos == P_FU_ATK) & ((s.st & ST_ATTACKED) == 0) & ((s.fl_c & FL_NO_ATTACK) == 0) & atk_ok
    ins = current_ins(bank, s)
    eff_cand = s.reg[jnp.clip(ins[2], 0, R - 1)]
    if fus:
        src = jnp.argmax(eff_cand)
        eff_cand = jnp.where(s.fsel, fus_picks(bank, s, src),
                             jnp.where(s.rsel > 0, rit_picks(bank, s, src, s.rsel), eff_cand))

    # libérations : sans case principale libre, l'une d'elles doit libérer une case principale (pas une zone extra)
    mainz = my_mon & (s.seq < N_MAIN_ZONES)
    trib_ok = ((free_m > 0) | mainz | (mainz & ~unsel).any()
               | ((s.need >= 2) & (mainz & unsel).any()))
    idle, bat, chn = s.dec == D_IDLE, s.dec == D_BATTLE, s.dec == D_CHAIN
    verbs = [
        idle & can_ns,
        idle & can_ns,
        idle & can_repos,
        bat & can_atk,
        ((s.dec == D_TRIBUTE) & my_mon & unsel & trib_ok) | ((s.dec == D_TARGET) & op_mon & ((s.fl_c & FL_NO_BTARGET) == 0))
        | ((s.dec == D_DISCARD) & in_hand & unsel) | ((s.dec == D_SELECT) & eff_cand & unsel),
        idle & in_hand & (spell | trap) & (free_s > 0),
    ]
    trig_w = chn & (s.wk == W_TRIG)
    cand = act_cand(bank, s)
    # coût + cible testés seulement sur les KC premiers candidats (sinon N × programmes × N : quadratique sous vmap)
    ce, ch = jnp.nonzero(cand, size=KC, fill_value=0)
    okc = jax.vmap(lambda h, e: act_ok(bank, s, h, e, p))(ch, ce) & (jnp.arange(KC) < cand.sum())
    ok = jnp.zeros((E, n), I32).at[ce, ch].add(okc.astype(I32)) > 0
    verbs = jnp.concatenate([jnp.stack(verbs), cand & ok])
    sp = jnp.zeros(N_SPECIAL, bool)
    sp = sp.at[A_TO_BP].set(idle & (s.phase == PH_MAIN1) & (s.turn > 1))
    sp = sp.at[A_TO_M2].set(bat)
    sp = sp.at[A_TO_EP].set(idle | bat)
    mz = (s.zctx == ZC_SUMMON) | (s.zctx == ZC_SPSUM)
    used = jnp.where(mz, zone_used(s, p, Z_MZONE), zone_used(s, p, Z_SZONE))[:7]
    spc = jnp.argmax(s.reg[jnp.clip(ins[1], 0, R - 1)])             # carte invoquée spécialement (ZC_SPSUM)
    fromex = (s.zctx == ZC_SPSUM) & (s.zone[spc] == Z_EXTRA)
    zok = jnp.where(fromex, ex_zones(s, p), ~used & (jnp.arange(7) < N_MAIN_ZONES))
    sp = sp.at[A_ZONE0:A_ZONE0 + 7].set((s.dec == D_ZONE) & zok)
    pos_ok = jnp.where(s.zctx == ZC_SPSUM, current_ins(bank, s)[2], P_FU_ATK | P_FU_DEF)  # positions permises
    sp = sp.at[A_POS_ATK].set((s.dec == D_POS) & ((pos_ok & P_FU_ATK) != 0))
    sp = sp.at[A_POS_DEF].set((s.dec == D_POS) & ((pos_ok & P_FU_DEF) != 0))
    sp = sp.at[A_PASS].set(chn & ~(trig_w & (s.tst < 2)))  # les déclencheurs obligatoires ne se refusent pas
    m = jnp.concatenate([verbs.reshape(-1), sp])
    return m & ~s.done & (s.dec != D_NONE)


# ---------------------------------------------------------------- application d'une action

def apply(bank, s, a):
    n = s.cid.shape[0]
    verb, c = a // n, a % n
    k = a - NV * n
    s_id = lambda s: s

    def summon(s, is_set):
        need = tributes_needed(bank, s, c)
        s = s._replace(pend=c, pend_set=jnp.bool_(is_set), need=need, picks=_i(0), zctx=_i(ZC_SUMMON))
        return s._replace(dec=jnp.where(need > 0, D_TRIBUTE, D_ZONE).astype(I32))

    def repos(s):
        new = jnp.where(s.pos[c] == P_FU_ATK, P_FU_DEF, P_FU_ATK)
        fd = (jnp.arange(n) == c) & (s.pos[c] == P_FD_DEF)  # verso → recto : invocation par flip
        s = raise_ev(raise_ev(s, EV_FLIP_SUMMON, fd, s.tp), EV_FLIP, fd, s.tp)
        s = enable(s._replace(pos=s.pos.at[c].set(new),
                              st=s.st.at[c].set(s.st[c] | ST_FORM_CHANGED | jnp.where(fd[c], ST_FLIP_TURN, 0))), fd)
        return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

    def attack(s):
        s = s._replace(st=s.st.at[c].set(s.st[c] | ST_ATTACKED), atk=c, atk_tgt=_i(-1), ds_t=s.clk)
        opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
        has_target = opp.any()
        # aucune cible possible (CANNOT_BE_BATTLE_TARGET) : attack_announce_failed, l'attaque compte, pas de fenêtre
        ok = (opp & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
        return _cond(has_target, lambda s: _cond(ok, lambda s: s._replace(dec=_i(D_TARGET)), to_battle, s),
                            lambda s: open_quick(raise_ev(s, EV_ATTACK, jnp.arange(n) == c, s.tp), s.tp, R_ATK1,
                                                 chained=False), s)

    def select(s):
        def pick(s):
            return s._replace(st=s.st.at[c].set(s.st[c] | ST_SEL), need=s.need - 1, picks=s.picks + 1)

        def tribute(s):
            s = pick(s)
            return _cond(s.need == 0, lambda s: selected_to_grave(s)._replace(dec=_i(D_ZONE)),
                                lambda s: s, s)

        def target(s):
            s = raise_ev(s._replace(atk_tgt=c), EV_ATTACK, jnp.arange(n) == s.atk, s.tp)
            return open_quick(s, s.tp, R_ATK1, chained=False)

        def discard(s):
            s = pick(s)
            return _cond(s.need == 0, lambda s: start_turn(selected_to_grave(s)), lambda s: s, s)

        def eff_select(s):
            def done(s):
                sel = (s.st & ST_SEL) != 0
                dst = jnp.clip(current_ins(bank, s)[1], 0, R - 1)
                return s._replace(reg=s.reg.at[dst].set(sel), st=s.st & ~ST_SEL, picks=_i(0), pc=s.pc + 1,
                                  dec=_i(D_RUN), fsel=jnp.bool_(False), rsel=_i(0))
            s = pick(s)
            rit = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[2], 0, R - 1)])
            fin = jnp.where(s.rsel > 0, rit_done(bank, s, rit, s.rsel), s.need == 0)
            return _cond(fin, done, lambda s: s, s)

        return _switch(jnp.clip(s.dec - D_TRIBUTE, 0, 4), [tribute, s_id, target, discard, eff_select], s)

    def sset(s):
        return s._replace(pend=c, zctx=_i(ZC_SSET), dec=_i(D_ZONE))

    def activate(s, e):
        p = s.dec_player
        # où reprendre une fois la chaîne résolue
        s = _switch(jnp.where(s.dec == D_IDLE, 0, jnp.where(s.dec == D_BATTLE, 1, jnp.where(s.wk == W_PHASE, 2, 3))), [
            lambda s: s._replace(ret=_i(R_IDLE), chained=jnp.bool_(False)),
            lambda s: s._replace(ret=_i(R_BATTLE), chained=jnp.bool_(False)),
            lambda s: s._replace(ret=_i(R_PHASE), ph_restart=p),
            s_id], s)
        s = s._replace(ch_eff=_i(e))
        from_hand = lambda s: s._replace(pend=c, zctx=_i(ZC_ACT), dec=_i(D_ZONE))
        flip = lambda s: begin_act(bank, s._replace(pos=s.pos.at[c].set(P_SFU)), c, e)
        card = bank.ekind[s.cid[c], e] == K_ACTIVATE  # activation de carte (pas un effet de monstre)
        return _switch(jnp.where(card & (s.zone[c] == Z_HAND), 0, jnp.where(card & (s.zone[c] == Z_SZONE), 1, 2)),
                          [from_hand, flip, lambda s: begin_act(bank, s, c, e)], s)

    def zone(s):
        z = k - A_ZONE0

        def to_szone(s, pos):
            return move(leave_hand(s, s.pend), s.pend, Z_SZONE, z, s.dec_player, pos)

        def set_done(s):
            s = to_szone(s, P_SFD)
            s = s._replace(st=s.st.at[s.pend].set(ST_SET_TURN), pend=_i(-1))
            return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

        return _switch(s.zctx, [
            lambda s: place_pending(bank, s, z),
            set_done,
            lambda s: begin_act(bank, to_szone(s, P_SFU), s.pend, s.ch_eff),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
        ], s)

    def position(s):
        c = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[1], 0, R - 1)])
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        # invocation fusion (depuis l'Extra Deck) ou rituelle (SPSUMMON avec proc) : correctement invoquée
        fromex = (s.zone[c] == Z_EXTRA) | (current_ins(bank, s)[3] != 0)
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.ch_p, pos)
        s = raise_ev(s, EV_SPSUMMON, jnp.arange(n) == c, s.ch_p)
        s = s._replace(proc=s.proc.at[c].set(s.proc[c] | fromex))
        return s._replace(st=s.st.at[c].set(ST_SUMMON_TURN), pc=s.pc + 1, dec=_i(D_RUN))

    def to_ep(s):
        return open_quick(s, 1 - s.tp, jnp.where(s.dec == D_BATTLE, R_TO_EPB, R_TO_EP), passed=True, chained=False)

    def spec(s):
        return _switch(jnp.where(k < A_ZONE0, k, jnp.where(k < A_POS_ATK, 3, jnp.where(k < A_PASS, 4, 5))), [
            lambda s: open_quick(s, 1 - s.tp, R_TO_BP, passed=True, chained=False),
            lambda s: open_quick(s, 1 - s.tp, R_TO_M2B, passed=True, chained=False),
            to_ep, zone, position,
            lambda s: _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s._replace(tst=s.tst + 1)),
                               lambda s: pass_priority(bank, s), s),
        ], s)

    s = _cond((s.dec == D_IDLE) | (s.dec == D_BATTLE), clear_ev, lambda s: s, s)
    return _switch(jnp.minimum(verb, NV), [
        lambda s: summon(s, False), lambda s: summon(s, True), repos, attack, select, sset,
        *[lambda s, e=e: activate(s, e) for e in range(E)], spec], s)


MAX_ITER = 256


def forced(bank, s):
    """Action imposée (option unique, ou sélection où toutes les cartes possibles doivent être prises), sinon -1."""
    return forced_m(bank, s, legal(bank, s))


def forced_m(bank, s, m):
    k = m.sum()
    sel = (s.dec == D_TRIBUTE) | (s.dec == D_DISCARD) | ((s.dec == D_SELECT) & ~s.fsel & (s.rsel == 0))
    # libérations rituelles : jouées d'office si l'union des choix possibles est elle-même le seul ensemble valide
    # (l'oracle n'a alors qu'une option) : elle atteint la somme sans carte superflue (Greater) ou exactement (Equal)
    n = s.cid.shape[0]
    cand = m[V_SELECT * n:(V_SELECT + 1) * n] | ((s.st & ST_SEL) != 0)
    lv = jnp.clip(bank.level[s.cid], 0, LVMAX)
    tot = jnp.where(cand, lv, 0).sum()
    mn = jnp.where(cand, lv, LVMAX + 1).min()
    rit = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[2], 0, R - 1)])
    req = bank.level[s.cid[rit]]
    # seulement avec une case principale libre (SelectWithSum, une décision) ; sans case libre, ygopro choisit carte par
    # carte (SelectUnselectGroup) et propose chaque carte de l'ensemble
    room = ex_zones(s, s.dec_player)[:N_MAIN_ZONES].any()
    uniq = (s.rsel > 0) & (s.dec == D_SELECT) & room & jnp.where(s.rsel == RIT_GREATER, (tot >= req) & (tot - mn < req),
                                                          tot == req)
    return jnp.where((k == 1) | (sel & (k > 0) & (k <= s.need)) | (uniq & (k > 0)), jnp.argmax(m), -1).astype(I32)


def run(bank, s):
    """Exécute les effets (D_RUN) jusqu'au prochain choix."""
    def body(carry):
        s, i = carry
        return refresh_if(bank, s, exec_one(bank, s)), i + 1
    return lax.while_loop(lambda c: (c[0].dec == D_RUN) & ~c[0].done & (c[1] < MAX_ITER), body, (s, _i(0)))[0]


def settle(bank, s):
    """Jusqu'au prochain choix réel : run, déclencheurs en attente (trig_eval), passes imposées (auto_pass)."""
    def body(carry):
        s, i = carry
        s = run(bank, s)
        s = lax.cond(s.dec == D_TRIG, lambda s: trig_eval(bank, s), lambda s: s, s)
        return refresh_if(bank, s, lax.cond(auto_pass(bank, s), lambda s: pass_priority(bank, s), lambda s: s, s)), i + 1
    more = lambda c: (((c[0].dec == D_RUN) | (c[0].dec == D_TRIG) | auto_pass(bank, c[0])) & ~c[0].done
                      & (c[1] < MAX_ITER))
    return lax.while_loop(more, body, (s, _i(0)))[0]


def step(bank, s, a):
    """Applique a, puis exécute les effets et joue automatiquement les décisions à option unique.
    Deux boucles : sous vmap, un lax.cond calcule ses deux branches, donc legal (coûteux) n'est évalué qu'aux points
    de décision, pas à chaque instruction exécutée."""
    def body(carry):
        s, a, i = carry
        s = settle(bank, refresh_if(bank, s, apply(bank, s, a)))
        return s, forced(bank, s), i + 1

    s = settle(bank, s)
    s, _, _ = lax.while_loop(lambda c: (c[1] >= 0) & ~c[0].done & (c[2] < MAX_ITER), body, (s, _i(a), _i(0)))
    return s


def _where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


def step_batch(bank, s, a, frac=0.0):
    """Version lot de step, pour l'entraînement : a[b] = action de la partie b (politique, ou l'action forcée rendue
    par l'appel précédent). Rend (états, masques légaux, actions forcées en attente ou -1).

    Les boucles et les calculs rares (trig_eval) sont décidés au niveau du lot. La boucle des actions forcées s'arrête
    dès que la part des parties qui en ont encore une est <= frac : sinon tout le lot attend la plus lente. Les parties
    arrêtées en route reprennent à l'appel suivant avec leur action forcée. frac = 0 : même résultat que vmap(step)."""
    nb = s.done.shape[0]

    def todo(s):
        return ((s.dec == D_RUN) | (s.dec == D_TRIG) | jax.vmap(lambda s: auto_pass(bank, s))(s)) & ~s.done

    def run_b(s):  # boucle d'exécution au niveau du lot : refresh seulement pour les parties modifiées
        def body(c):
            s, i = c
            r = (s.dec == D_RUN) & ~s.done
            return refresh_b(bank, s, _where(r, jax.vmap(lambda s: exec_one(bank, s))(s), s)), i + 1
        return lax.while_loop(lambda c: ((c[0].dec == D_RUN) & ~c[0].done).any() & (c[1] < MAX_ITER), body,
                              (s, _i(0)))[0]

    def settle_body(c):
        s, i = c
        s = run_b(s)
        need = s.dec == D_TRIG
        s = lax.cond(need.any(), lambda s: _where(need, jax.vmap(lambda s: trig_eval(bank, s))(s), s),
                     lambda s: s, s)
        ap = jax.vmap(lambda s: auto_pass(bank, s))(s)
        s = lax.cond(ap.any(), lambda s: refresh_b(bank, s, _where(ap, jax.vmap(lambda s: pass_priority(bank, s))(s), s)),
                     lambda s: s, s)
        return s, i + 1

    def settle_b(s):
        return lax.while_loop(lambda c: todo(c[0]).any() & (c[1] < MAX_ITER), settle_body, (s, _i(0)))[0]

    def body(carry):
        s, m, a, i = carry
        act = (a >= 0) & ~s.done
        s1 = jax.vmap(lambda s, a: apply(bank, s, a))(s, jnp.maximum(a, 0))
        s = _where(act, settle_b(refresh_b(bank, s, _where(act, s1, s))), s)
        lg = lax.cond((s.fsel | (s.rsel > 0)).any(), lambda s: jax.vmap(lambda s: legal(bank, s))(s),
                      lambda s: jax.vmap(lambda s: legal(bank, s, fus=False))(s), s)
        m = jnp.where(act[:, None], lg, m)
        return s, m, jnp.where(act, jax.vmap(lambda s, m: forced_m(bank, s, m))(s, m), -1), i + 1

    def more(c):
        pend = ((c[2] >= 0) & ~c[0].done).sum()
        return (pend > jnp.where(c[3] == 0, 0, frac * nb)) & (c[3] < MAX_ITER)

    s = settle_b(s)
    m = jnp.zeros((nb, n_actions(s.cid.shape[1])), bool)
    s, m, f, _ = lax.while_loop(more, body, (s, m, a.astype(I32), _i(0)))
    return s, m, f

def new_game(bank, deck0, deck1, key=None, extra0=None, extra1=None):
    """deck0/deck1 : [D] index dans la banque, dans l'ordre d'ajout (dernier = dessus du deck).
    extra0/extra1 : [X] Extra Deck (même taille pour les deux joueurs ; None : aucun), dans l'ordre d'ajout.

    key=None : ordre fixe, slots dans l'ordre des decks (pour coller à l'oracle en DUEL_PSEUDO_SHUFFLE).
    Avec une clé : decks mélangés et cartes réparties au hasard dans les slots, pour que ni l'ordre du deck
    ni le numéro de slot ne trahissent une carte cachée.
    """
    d = deck0.shape[0]
    x = 0 if extra0 is None else extra0.shape[0]
    nd = 2 * d
    n = nd + 2 * x
    cid = jnp.concatenate([deck0, deck1] + ([] if x == 0 else [extra0, extra1])).astype(I32)
    owner = jnp.concatenate([jnp.repeat(jnp.arange(2, dtype=I32), d), jnp.repeat(jnp.arange(2, dtype=I32), x)])
    seq = jnp.concatenate([jnp.tile(jnp.arange(d, dtype=I32), 2), jnp.tile(jnp.arange(x, dtype=I32), 2)])
    zone0 = jnp.concatenate([jnp.full(nd, Z_DECK, I32), jnp.full(2 * x, Z_EXTRA, I32)])
    if key is not None:
        k1, k2 = jax.random.split(key)
        own = owner[:nd]
        order = jnp.lexsort((jax.random.uniform(k1, (nd,)), own))  # mélange à l'intérieur de chaque deck
        seq = seq.at[:nd].set(jnp.zeros(nd, I32).at[order].set(jnp.arange(nd, dtype=I32)) - own * d)
        perm = jax.random.permutation(k2, n)
        cid, owner, seq, zone0 = cid[perm], owner[perm], seq[perm], zone0[perm]
    z = jnp.zeros(n, I32)
    f = jnp.bool_(False)
    s = State(cid=cid, zone=zone0,
              seq=seq, ctrl=owner, owner=owner, pos=jnp.full(n, P_FD_DEF, I32),
              st=z, lp=jnp.full(2, 8000, I32), turn=_i(0), tp=_i(1), phase=_i(PH_DRAW), ns_used=f,
              dec=_i(D_NONE), dec_player=_i(0), pend=_i(-1), pend_set=f, need=_i(0), picks=_i(0),
              zctx=_i(ZC_SUMMON), zsel=_i(0), ch_src=_i(-1), ch_eff=_i(0), ch_p=_i(0), ch_k=_i(0), ch_dis=f, pg=_i(0),
              pc=_i(0), reg=jnp.zeros((R, n), bool), done=f, odraw=f, winner=_i(-1),
              lk_src=jnp.full(CH, -1, I32), lk_eff=jnp.zeros(CH, I32), lk_p=jnp.zeros(CH, I32),
              lk_reg=jnp.zeros((CH, R, n), bool), lk_neg=jnp.zeros(CH, bool), lk_t=jnp.zeros(CH, I32), mv=z, clk=_i(0), n_chain=_i(0),
              wk=_i(W_QUICK), cw_passed=f, ret=_i(R_IDLE), chained=f, ph_restart=_i(0), wph=_i(WP_DRAW),
              bend_ep=f, atk=_i(-1), atk_tgt=_i(-1), ev_g=jnp.zeros((NEV, n), bool), ev_p=jnp.zeros(NEV, I32),
              pzone=zone0, tev=jnp.zeros((NEV, n), bool), tg=jnp.zeros(n, bool), tg_e=z,
              tst=_i(0), pt_p=_i(0), pt_passed=f, dstep=_i(DS_NONE), ds_t=_i(0), ds_flip=f, opt_cid=jnp.zeros((2, KO), I32),
              lock_cid=jnp.zeros((2, KO), I32), atk_c=bank.atk[cid], df_c=bank.df[cid], batk_c=bank.atk[cid], tm_code=jnp.zeros(TM, I32),
              tm_m=jnp.zeros((TM, n), bool), tm_v=jnp.zeros((TM, n), I32), tm_t=jnp.zeros(TM, I32), tm_dur=jnp.zeros(TM, I32), tm_rd=jnp.zeros(TM, bool), tm_n=_i(0),
              ovf=_i(0), en_t=z, fl_c=z, ib_c=jnp.full(n, BIG, I32), pf_c=jnp.zeros(2, I32), cav=jnp.zeros((n, E), bool), cav_t=jnp.zeros((n, E), I32),
              proc=jnp.zeros(n, bool), fz_c=jnp.zeros(n, bool), fsel=f, rz_c=z, rsel=_i(0))
    s = refresh(bank, s)
    for p in (0, 1):
        for _ in range(5):
            s = draw(s, p)
    s = start_turn(s)
    # la première décision peut elle aussi être à option unique
    a = forced(bank, s)
    return jax.lax.cond(a >= 0, lambda s: step(bank, s, a), lambda s: s, s)
