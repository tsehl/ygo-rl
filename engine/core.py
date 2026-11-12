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
Z_OVERLAY = 128  # matériau Xyz attaché (ov_par : le Xyz, seq : rang parmi ses matériaux, ctrl : son contrôleur)
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
TYPE_TUNER = 0x1000
TYPE_PENDULUM = 0x1000000
TYPE_SPSUMMON = 0x2000000  # aucune invocation normale / pose (monstre à invocation spéciale)
P_SPELL = TYPE_PENDULUM | TYPE_SPELL  # type d'un pendule en zone pendule (card::get_type), et de son activation
# statuts (bits)
ST_SUMMON_TURN, ST_ATTACKED, ST_FORM_CHANGED, ST_SEL = 1, 2, 4, 8  # ST_SEL : choisie dans la sélection en cours
ST_SET_TURN, ST_LEAVE = 16, 32  # posée ce tour (magie/piège) ; activée, ira au cimetière en fin de chaîne
ST_BDES = 64  # détruite au combat, part au cimetière à la fin du damage step (si elle est encore sur le terrain)
ST_ON = 128  # magie/piège continue résolue : ses effets continus s'appliquent (STATUS_EFFECT_ENABLED)
ST_CNT = 256  # ST_CNT << e : effet e déjà utilisé ce tour (SetCountLimit(1) sans code : par carte, effacé si elle bouge)
ST_CNTN = 1024  # ST_CNTN << e : idem, sans remise à zéro au tour suivant (EFFECT_FLAG_NO_TURN_RESET)
ST_FLIP_TURN = 4096  # invoquée par flip ce tour (STATUS_FLIP_SUMMON_TURN)
ST_SPSUM, ST_LINKSUM = 8192, 16384  # invoquée spécialement / par lien (IsSpecialSummoned, IsLinkSummoned) : tant
# qu'elle reste sur le terrain
SUMMON_NORMAL, SUMMON_TRIBUTE, SUMMON_SPECIAL = 0x10000000, 0x11000000, 0x40000000
SUMMON_FUSION, SUMMON_RITUAL, SUMMON_SYNCHRO = 0x43000000, 0x45000000, 0x46000000
SUMMON_XYZ, SUMMON_PENDULUM, SUMMON_LINK = 0x49000000, 0x4a000000, 0x4c000000
# suminfo : type natif (bits 24..30 et 0..15), zone d'origine (16..23), joueur (31).
# L'information survit aux déplacements ; une nouvelle invocation la remplace.
# Raisons natives composables (ocgapi_constants.h).
REASON_DESTROY, REASON_RELEASE, REASON_MATERIAL, REASON_SUMMON = 1, 2, 8, 16
REASON_BATTLE, REASON_EFFECT, REASON_COST, REASON_RULE = 32, 64, 128, 1024
REASON_LOST_TARGET, REASON_DISCARD = 512, 16384
REASON_FUSION, REASON_SYNCHRO, REASON_RITUAL, REASON_XYZ, REASON_LINK = 0x40000, 0x80000, 0x100000, 0x200000, 0x10000000
MOVE_REASON_EXPLICIT = 1 << 30

ST_ACTDIS = 32768  # activation niée (STATUS_ACTIVATE_DISABLED) : un pendule va alors au cimetière, pas dans l'Extra Deck
ST_PSTEP = 65536  # placée par l'invocation pendule en cours (EVENT_SPSUMMON du groupe à la fin)
ST_TURN_RESET = ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN | ST_CNT | (ST_CNT << 1) | ST_FLIP_TURN
# phases (= PHASE_* d'ygopro-core, telles que vues aux décisions)
PH_DRAW, PH_STANDBY, PH_MAIN1, PH_BATTLE, PH_MAIN2, PH_END = 0x1, 0x2, 0x4, 0x8, 0x100, 0x200
# décisions en attente (D_RUN : effet en cours d'exécution, interne au moteur)
D_NONE, D_IDLE, D_BATTLE, D_TRIBUTE, D_ZONE, D_TARGET, D_DISCARD, D_SELECT, D_POS, D_RUN, D_CHAIN, D_TRIG, D_YESNO = range(13)
# D_YESNO : question oui / non pendant la résolution (Duel.SelectYesNo) : A_YES continue l'effet, A_PASS l'arrête
# D_TRIG (interne) : déclencheurs en attente à tester, traité par settle hors de la boucle d'exécution
# à quoi sert le choix de case en cours
ZC_SUMMON, ZC_SSET, ZC_ACT, ZC_SPSUM, ZC_PROC, ZC_PEND, ZC_REPLAY = range(7)  # ZC_PROC : invocation par procédure (synchro, Xyz,
ZC_INLINE = 7  # choix d'un effet lors de l'activation d'un piège continu (description native 94)
CF_ACT, CF_CONT = 1, 2  # nature du maillon, indépendante du genre de l'effet choisi
CF_ACT_NEG = 4  # activation niée ; distincte de DISABLE_CHAIN
CF_SRC_UNRELATED = 8  # la source a bougé après le début de l’activation
# Link) ; ZC_PEND : invocation pendule (une carte à la fois, dans l'ordre des uid) ; ZC_REPLAY : la question D_YESNO
# est « relancer l'attaque ? » (SelectYesNo 30), pas un Duel.SelectYesNo d'effet
PS_SYN, PS_XYZ, PS_LINK, PS_PEND = 1, 2, 3, 4  # psel
# flèches Link (LINK_MARKER_*)
LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T, LM_TR = 1, 2, 4, 8, 0x20, 0x40, 0x80, 0x100
# fenêtres de réponse ; W_TRIG : mise en chaîne des effets déclenchés (SEGOC)
W_QUICK, W_PHASE, W_TRIG = range(3)
# événements (EVENT_* d'ygopro-core) : disponibles pendant la fenêtre qui suit et la chaîne bâtie dessus ;
# ceux qui surviennent pendant la résolution d'un maillon remplacent les précédents (seul le dernier maillon compte)
# Les mêmes événements alimentent aussi `tev` (cumul depuis le dernier point d'événement) pour les déclencheurs.
EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON, EV_ATTACK, EV_TO_GRAVE, EV_BATTLE_DESTROYED, EV_FLIP, EV_PRE_DMG = range(8)
NEV = 8  # EV_PRE_DMG : EVENT_PRE_DAMAGE_CALCULATE (attaquant et cible, au début de la fenêtre du calcul des dommages)
# suite d'une fenêtre W_QUICK fermée sans chaîne (`chained` : une chaîne s'est résolue depuis l'ouverture)
R_IDLE, R_BATTLE, R_TO_BP, R_TO_EP, R_TO_M2B, R_TO_EPB, R_DP, R_PHASE, R_ATK1, R_ATK2, R_DMG1, R_DMG2, \
    R_DS1, R_DS2, R_DS3, R_HAND = range(16)
# R_HAND : après la défausse de fin de tour (PhaseEvent, étapes 21-22 : point d'événement, puis nouveau contrôle de la
# limite de main ; les déclencheurs optionnels de la End Phase ne sont plus proposés, hand_adjusted)
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
# SetCountLimit(1, {id, k}) avec k ≥ 1 : compteur par nom distinct pour chaque k (eopt = OPT_NAMEK + k - 1)
OPT_NAMEK = 4
# bank.etrig : déclencheur optionnel sans EFFECT_FLAG_DELAY (« lorsque … ») : peut manquer le timing (brk)
TRIG_WHEN = 1 << 30
# déclencheur FIELD : n'importe quelle carte subit l'événement, la carte de l'effet est dans les zones erng (main,
# cimetière, zone monstres / magie-piège recto) ; la condition sur eg passe par OP_EVENT(c=1) dans le coût
TRIG_FIELD = 1 << 29
# fenêtres de phase (PhaseEvent)
WP_DRAW, WP_STANDBY, WP_BSTART, WP_BEND, WP_END = range(5)

HAND_LIMIT = 6
N_MAIN_ZONES = 5

# effets
E = 2           # effets par carte
P = 12          # instructions par programme
PT = 6          # instructions testées par programme de coût / cible (test_ok)
R = 4           # registres ensembles
NF = 4          # filtres par effet
KC = 24         # (carte, effet) candidats à l'activation testés par legal (main + terrain du joueur qui décide)
MAX_SLOTS = 192  # couvre 60+15 cartes par joueur et les emplacements de jetons sur le terrain
ACT_ROUNDS = (MAX_SLOTS * E + KC - 1) // KC  # borne constante : pas de matrice N × N
KA = 2          # candidats testés par auto_pass (au-delà : pas de passe automatique, legal décide)
K_NONE, K_ACTIVATE, K_IGNITION, K_TRIG_F, K_TRIG_O, K_QUICK, K_CONT, K_PROC = range(8)  # déclencheur obligatoire / optionnel ;
# K_QUICK : effet rapide de monstre (vitesse 2), activable depuis les zones erng ; K_CONT : effet continu, appliqué
# tant que sa carte est recto dans les zones erng (et résolue, pour une magie/piège)
PG_COST, PG_TARGET, PG_RESOLVE = range(3)
CH = 12          # maillons de chaîne (au plus 1 ignition + 2 × 5 cases magie/piège)
(OP_END, OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_CHOOSE, OP_MOVE, OP_SPSUMMON, OP_DRAW,
 OP_DAMAGE, OP_RECOVER, OP_SHUFFLE_HAND, OP_COND_CHAIN, OP_CHAINCARD, OP_NEGATE, OP_PAY_LP, OP_EVENT, OP_COND_TURN,
 OP_COND_EVP, OP_COND_SELF, OP_LOCK, OP_MODIFY, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE, OP_FUSMAT, OP_RITMAT, OP_LABEL,
 OP_COND_LABEL, OP_YESNO) = range(31)
N_OPS = 31
# OP_LABEL : le maillon retient si le registre a est non vide (e:SetLabel à l'activation) ; OP_COND_LABEL : condition
# sur ce label à la résolution ; OP_YESNO : le joueur de l'effet choisit de continuer (D_YESNO)
# champs d'un filtre ; les zones sont relatives au joueur de l'effet (moi / adversaire)
F_LOC_ME, F_LOC_OP, F_POS, F_TYPE, F_NOT_SELF, F_ATK_LO, F_ATK_HI, F_DEF_LO, F_DEF_HI, F_ATTR, F_RACE, F_SET, \
    F_CODE1, F_CODE2, F_ST, F_LV_LO, F_LV_HI, F_SPSUM, F_NODIS, F_FUS, F_RIT, F_ZX, F_REASON, F_NOTTYPE = range(24)
FW = 24  # F_ZX : bits ZX_* (co-lié, case principale)
ZX_COLINK, ZX_MAIN, ZX_EXTRA = 1, 2, 4  # GetMutualLinkedGroupCount() > 0 (cache cl_c) ; GetSequence() < 5  # F_RIT : monstre rituel invocable maintenant (cache rz_c) en mode RIT_GREATER ou RIT_EQUAL  # F_FUS : monstre fusion de l'Extra Deck invocable par fusion maintenant (cache fz_c)  # F_ST : l'un des bits de statut ; F_SPSUM : peut être invoquée spécialement (pas de FL_NO_SS) ;
# F_NODIS : effets non annulés (not IsDisabled)
NSET = 4  # archétypes (setcode 16 bits) par carte
# modificateurs de stats (EFFECT_*), effets continus (bank.emod) et temporaires (table tm_* de l'état)
# UPDATE_*, SET_*, SET_*_FINAL, SET_BASE_* pour l'ATK puis la DEF
MC_NONE, MC_ATK, MC_SET_ATK, MC_FINAL_ATK, MC_BASE_ATK, MC_DEF, MC_SET_DEF, MC_FINAL_DEF, MC_BASE_DEF = range(9)
# interdictions et protections (effets continus) : EFFECT_CANNOT_ATTACK, CANNOT_BE_BATTLE_TARGET, CANNOT_BE_EFFECT_TARGET,
# IMMUNE_EFFECT (effets de piège), CANNOT_SPECIAL_SUMMON (cartes visées, toutes zones), INDESTRUCTABLE_BATTLE (valeur :
# ATK minimale du monstre adverse, 0 = toujours), CANNOT_ACTIVATE (cartes piège, joueurs M_VAL : 1 = moi, 2 = adversaire)
MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP, MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP = range(9, 16)
MC_NO_DIRECT = 17  # EFFECT_CANNOT_DIRECT_ATTACK
MC_DISABLE = 16  # EFFECT_DISABLE : effets de la carte annulés (continus inactifs, pas d'activation, maillons annulés)
MC_FUS_SUB = 19  # substitut de matériau nommé (métadonnée passive de banque)
MC_INDES_EFF = 18  # EFFECT_INDESTRUCTABLE_EFFECT : pas détruite par les effets (OP_MOVE avec destruction)
FL_NO_ATTACK, FL_NO_BTARGET, FL_NO_ETARGET, FL_IMM_TRAP, FL_NO_SS, FL_DISABLED = 1, 2, 4, 8, 16, 32  # bits de fl_c
FL_NO_DIRECT = 128  # pas d'attaque directe
FL_INDES_EFF = 256  # indestructible par les effets
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
TD_TURN, TD_DCAL, TD_PERM = range(3)  # TD_PERM : tant que la carte reste sur le terrain (RESETS_STANDARD)
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
EXP = 16        # monstres synchro / Xyz / Link de l'Extra Deck du joueur qui décide évalués par refresh_ex (15 au plus)
BIG = 1 << 20

# actions : verbe × slot, puis actions spéciales
V_SUMMON, V_MSET, V_REPOS, V_ATTACK, V_SELECT, V_SSET, V_ACT0 = range(7)
NV = V_ACT0 + E
A_TO_BP, A_TO_M2, A_TO_EP, A_ZONE0, A_POS_ATK, A_POS_DEF, A_PASS, A_YES = 0, 1, 2, 3, 10, 11, 12, 13  # A_ZONE0 + z
N_SPECIAL = 14  # A_YES : « oui » à D_YESNO (« non » : A_PASS)


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
    fsub: jnp.ndarray   # [C] masque des emplacements nommés acceptant un substitut
    submat: jnp.ndarray # [C,0 ou 1] zones des substituts ; largeur 0 élimine le chemin au traçage
    erl: jnp.ndarray    # [C] 0 : libre ; 1 : EnableReviveLimit ; 2 : interdit par effet même après invocation correcte
    smat: jnp.ndarray   # [C, 2, FW] synchro : filtres du Syntoniseur (un seul) et des non-Syntoniseurs (1 ou plus)
    syn: jnp.ndarray    # [C] monstre synchro (procédure d'invocation synchro)
    xmat: jnp.ndarray   # [C, FW] Xyz : filtre des matériaux (en plus du niveau = rang)
    xct: jnp.ndarray    # [C] Xyz : nombre de matériaux (0 : pas Xyz)
    lmark: jnp.ndarray  # [C] Link : flèches (LM_*) ; le lien est dans level, la DEF est nulle
    lmat: jnp.ndarray   # [C, FW] Link : filtre des matériaux
    lmin: jnp.ndarray   # [C] Link : nombre de matériaux, au moins
    lmax: jnp.ndarray   # [C] Link : au plus (0 : pas de procédure Link)
    lneed: jnp.ndarray  # [C] Link : les matériaux doivent comprendre un monstre Link
    ldiff: jnp.ndarray  # [C] Link : matériaux de noms différents (GetClassCount(Card.GetCode) == #g)
    lsc: jnp.ndarray    # [C] pendule : échelle gauche
    rsc: jnp.ndarray    # [C] pendule : échelle droite
    einline: jnp.ndarray # [C,0 ou E] effet proposé lors de l'activation (index + 1, 0 : aucun)
    erefund: jnp.ndarray # [C,0 ou E] EFFECT_COUNT_CODE_OATH : rendre le compte si l'activation est niée


def make_bank(code, atk, df, level, ctype=None, ekind=None, prog=None, filt=None, etrig=None, eopt=None, attr=None,
              race=None, erng=None, setcode=None, emod=None, edmg=None, fmat=None, fmn=None, erl=None,
              smat=None, syn=None, xmat=None, xct=None, lmat=None, lmin=None, lmax=None, lneed=None, ldiff=None,
              lsc=None, rsc=None, fsub=None, submat=None, einline=None, erefund=None):
    """df : DEF, ou flèches pour un monstre Link (champ def de la base ygopro) : rangées dans lmark, DEF nulle."""
    c = code.shape[0]
    i = lambda x, shape: jnp.zeros(shape, I32) if x is None else jnp.asarray(x, I32)
    is_link = (jnp.full(c, TYPE_MONSTER | TYPE_NORMAL, I32) if ctype is None else jnp.asarray(ctype, I32)) & TYPE_LINK != 0
    df = jnp.asarray(df, I32)
    return Bank(code=jnp.asarray(code, I32), atk=jnp.asarray(atk, I32), df=jnp.where(is_link, 0, df), tprog=test_prog(prog, c),
                lmark=jnp.where(is_link, df, 0), lmat=i(lmat, (c, FW)), lmin=i(lmin, (c,)), lmax=i(lmax, (c,)),
                lneed=i(lneed, (c,)), ldiff=i(ldiff, (c,)), lsc=i(lsc, (c,)), rsc=i(rsc, (c,)),
                level=jnp.asarray(level, I32),
                ctype=jnp.full(c, TYPE_MONSTER | TYPE_NORMAL, I32) if ctype is None else jnp.asarray(ctype, I32),
                attr=i(attr, (c,)), race=i(race, (c,)),
                ekind=i(ekind, (c, E)), prog=i(prog, (c, E, 3, P, 4)), filt=i(filt, (c, E, NF, FW)),
                etrig=i(etrig, (c, E)), eopt=i(eopt, (c, E)), erng=i(erng, (c, E)), setcode=i(setcode, (c, NSET)),
                emod=i(emod, (c, E, MW)), edmg=i(edmg, (c, E)), fmat=i(fmat, (c, FK, FW)), fmn=i(fmn, (c,)),
                fsub=i(fsub, (c,)), submat=i(submat, (c, 0)),
                erl=i(erl, (c,)), smat=i(smat, (c, 2, FW)), syn=i(syn, (c,)),
                xmat=i(xmat, (c, FW)), xct=i(xct, (c,)), einline=i(einline, (c, 0)), erefund=i(erefund, (c, 0)))


def test_prog(prog, c):
    """Programmes de test compacts (numpy, à la construction de la banque) : coût puis cible, sans les OP_END de
    remplissage (sans effet en mode test). test_ok ne parcourt que la longueur du plus long, pas 2 × PT."""
    import numpy as np
    pr = np.zeros((c, E, 3, P, 4), np.int32) if prog is None else np.asarray(prog, np.int32)
    rows=[]
    for card in range(c):
        effects=[]
        for effect in range(E):
            result=[]
            for opcode,a,b,cc in pr[card,effect,:PG_RESOLVE,:PT].reshape(-1,4):
                opcode,a,b,cc=map(int,(opcode,a,b,cc))
                if opcode==OP_END:
                    continue
                if opcode==OP_FILTER:
                    cc &= 15 | (1 << 20)  # conserve la copie ; les autres bits hauts servent au test dérivé
                can_join = result and result[-1][0]==OP_FILTER and result[-1][1]==a
                minimum = (opcode==OP_REQ and cc==0) or (opcode==OP_CHOOSE and a==b)
                count = cc&15 if opcode==OP_CHOOSE else b
                if can_join and (opcode==OP_REQ or opcode==OP_CHOOSE and a==b) and 0<=count<=(255 if minimum else 254):
                    previous=list(result[-1]);packed=previous[3]
                    if minimum:
                        packed=(packed & ~(255<<4)) | (max((packed>>4)&255,count)<<4)
                    else:
                        old=(packed>>12)&255
                        packed=(packed & ~(255<<12)) | ((min(old,count+1) if old else count+1)<<12)
                    previous[3]=packed;result[-1]=tuple(previous)
                else:
                    result.append((opcode,a,b,cc))
            effects.append(result)
        rows.append(effects)
    n=max(1,max(len(effect) for card in rows for effect in card))
    out=np.zeros((c,E,n,4),np.int32)
    for card,effects in enumerate(rows):
        for effect,instructions in enumerate(effects):
            if instructions:out[card,effect,:len(instructions)]=instructions
    return jnp.asarray(out)


class State(NamedTuple):
    reason: jnp.ndarray # [N] dernier masque de raisons natif connu
    suminfo: jnp.ndarray # [N] dernière invocation : type, zone d’origine et joueur
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
    smin: jnp.ndarray    # CHOOSE à taille variable : nombre minimal de cartes (A_PASS termine la sélection au-delà)
    zctx: jnp.ndarray    # ZC_* : usage du choix de case en cours
    zsel: jnp.ndarray    # case choisie pour l'invocation spéciale en cours
    ch_src: jnp.ndarray  # effet en cours (activation ou résolution) : carte source (-1 si aucun)
    ch_src_mv: jnp.ndarray  # horloge de la source au début de l’activation
    ch_tgt_t: jnp.ndarray  # horloge de création des relations de cible
    ch_eff: jnp.ndarray  # index d'effet
    ch_p: jnp.ndarray    # joueur qui active
    ch_k: jnp.ndarray    # index de son maillon (= n_chain pendant l'activation)
    ch_dis: jnp.ndarray  # activé par une carte sous EFFECT_DISABLE_EFFECT : le maillon sera annulé (DISABLE_CHAIN)
    ch_lab: jnp.ndarray  # label de l'effet en cours (OP_LABEL)
    ch_flags: jnp.ndarray # CF_* du maillon en cours
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
    lk_tgt_t: jnp.ndarray  # [CH] horloge de création des relations de cible
    lk_lab: jnp.ndarray  # [CH] label (OP_LABEL)
    lk_flags: jnp.ndarray # [CH] CF_* conservés après le choix d'un effet inline
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
    tew: jnp.ndarray     # [NEV, N] idem, depuis le dernier break_effect (déclencheurs optionnels « lorsque »)
    teg: jnp.ndarray     # [NEV, N] événements du dernier point d'événement (eg des déclencheurs, OP_EVENT c=1)
    tg: jnp.ndarray      # [N] effet déclenché en attente
    tg_e: jnp.ndarray    # [N] son index d'effet
    tst: jnp.ndarray     # étape SEGOC : 0/1 obligatoires (tour, adversaire), 2/3 optionnels
    pt_p: jnp.ndarray    # fenêtre à ouvrir si aucun déclencheur n'est activé : joueur, autre joueur a passé
    pt_passed: jnp.ndarray
    dstep: jnp.ndarray   # DS_* : où en est le damage step
    ds_t: jnp.ndarray    # horloge au début du damage step (attaquant ou cible déplacés depuis : attaque annulée)
    ds_flip: jnp.ndarray  # la cible a été retournée au début du damage step (événement FLIP au calcul des dommages)
    rb_m: jnp.ndarray    # [N] monstres adverses à la (re)déclaration de l'attaque (opp_mzone d'ygopro-core)
    rb_t: jnp.ndarray    # horloge à la (re)déclaration : un monstre déplacé depuis n'est plus le même (fieldid_r)
    rpl: jnp.ndarray     # attaque relancée (is_replaying_attack : pas de nouvel EVENT_ATTACK_ANNOUNCE)
    opt_cid: jnp.ndarray   # [2, N*E] clés + 1 des effets « une fois par tour » déjà utilisés
    lock_cid: jnp.ndarray  # [2, N] cid + 1 des cartes dont le joueur ne peut plus activer les effets ce tour
    # stats courantes (cache recalculé par refresh après chaque micro-pas) et modificateurs temporaires
    atk_c: jnp.ndarray   # [N] ATK courante (hors zone monstres : ATK de base)
    df_c: jnp.ndarray    # [N]
    batk_c: jnp.ndarray  # [N] ATK de base courante (GetBaseAttack)
    tm_code: jnp.ndarray  # [TM] MC_* (dans l'ordre de création ; table vidée en fin de tour)
    tm_src: jnp.ndarray  # [TM] carte source du modificateur (propriétaire de l'effet)
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
    sz_c: jnp.ndarray    # [N] monstre synchro de l'Extra Deck invocable maintenant (matériaux sur le terrain)
    xz_c: jnp.ndarray    # [N] monstre Xyz de l'Extra Deck invocable maintenant
    lz_c: jnp.ndarray    # [N] monstre Link de l'Extra Deck invocable maintenant
    cl_c: jnp.ndarray    # [N] monstre Link co-lié (GetMutualLinkedGroupCount() > 0), calculé par refresh
    psel: jnp.ndarray    # sélection de matériaux d'une procédure en cours (D_TRIBUTE, carte pend) : PS_*, 0 sinon
    ov_par: jnp.ndarray  # [N] Z_OVERLAY : index du Xyz qui porte ce matériau (-1 sinon)
    ps_used: jnp.ndarray  # invocation pendule faite ce tour (flag 10000000 de proc_pendulum.lua)
    uid: jnp.ndarray     # [N] ordre de création de la carte (cardid d'ygopro : deck 0, Extra 0, deck 1, Extra 1)
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


def summon_mask(typ, negate=False):
    """Sélecteur négatif pour F_ST / COND_SELF : bits de type natif compactés."""
    assert typ & 0x00ff0000 == 0
    return -(((typ >> 24) & 255) | ((typ & 65535) << 8) | (int(negate) << 24))


def summon_matches(info, encoded):
    packed = -encoded
    mask = ((packed & 255) << 24) | ((packed >> 8) & 65535)
    return (((info & 0x7f00ffff) & mask) == mask) ^ ((packed & (1 << 24)) != 0)


def effect_summon_type(mode):
    """SPSUMMON : procédure positive ; marqueur générique encodé -(tag+1)."""
    ordinary = jnp.where(mode == 1, SUMMON_RITUAL, jnp.where(mode == 2, SUMMON_FUSION, SUMMON_SPECIAL))
    return jnp.where(mode < 0, SUMMON_SPECIAL | (-mode - 1), ordinary)


def summon_info(typ, location, player):
    return _i(typ) | (location << 16) | (player << 31)


def move(s, c, zone, seq, ctrl, pos):
    return s._replace(pzone=s.pzone.at[c].set(s.zone[c]), zone=s.zone.at[c].set(zone), seq=s.seq.at[c].set(seq), ctrl=s.ctrl.at[c].set(ctrl),
                      pos=s.pos.at[c].set(pos), st=s.st.at[c].set(0), mv=s.mv.at[c].set(s.clk + 1),
                      en_t=s.en_t.at[c].set(s.clk + 1), clk=s.clk + 1)


def enable(s, mask):
    """Les effets continus des cartes mask s'activent maintenant (nouvel id d'effet dans ygopro)."""
    return s._replace(en_t=jnp.where(mask, s.clk + 1, s.en_t), clk=s.clk + 1)


LOC_RANK = (Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED)


def group_send(s, mask, zone, pos, zone2=None, pos2=None, alt=None, reason=0):
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
            base2 = count(s, zone2, o)
            if zone2 == Z_EXTRA and pos2 == P_FD_DEF:  # verso : avant les pendules recto (end - extra_p_count)
                base2 = base2 - ((s.zone == Z_EXTRA) & (s.ctrl == o) & (s.pos == P_FU_DEF)).sum()
            base = jnp.where(alt, base2, count(s, zone, o))
        else:
            base = count(s, zone, o)
        seq = jnp.where(mo, base + rank, seq)
    s = compact(s, mask)
    s = s._replace(reason=jnp.where(mask, reason, s.reason))
    if alt is not None and zone2 == Z_EXTRA and pos2 == P_FD_DEF:  # les pendules recto se décalent
        fu = (s.zone == Z_EXTRA) & (s.pos == P_FU_DEF) & ~mask
        nin = jnp.stack([(mask & alt & (s.owner == o)).sum() for o in (0, 1)])
        s = s._replace(seq=jnp.where(fu, s.seq + nin[s.ctrl], s.seq))
    out = (dest == Z_HAND) | (dest == Z_DECK) | (dest == Z_EXTRA)  # STATUS_PROC_COMPLETE perdu
    s = s._replace(proc=jnp.where(mask & out, False, s.proc))
    return s._replace(pzone=jnp.where(mask, s.zone, s.pzone), zone=jnp.where(mask, dest, s.zone),
                      seq=jnp.where(mask, seq, s.seq), ctrl=jnp.where(mask, s.owner, s.ctrl),
                      pos=jnp.where(mask, pos, s.pos), st=jnp.where(mask, 0, s.st), mv=jnp.where(mask, s.clk + 1, s.mv),
                      en_t=jnp.where(mask, s.clk + 1, s.en_t),
                      clk=s.clk + 1)


def detach_fix(s):
    """Après un détachement (un matériau a quitté Z_OVERLAY) : ov_par effacé, rangs des matériaux restants resserrés
    (card::xyz_remove). O(8N)."""
    n = s.cid.shape[0]
    gone = (s.zone != Z_OVERLAY) & (s.ov_par >= 0)

    def fix(s):
        ov = s.zone == Z_OVERLAY
        par = jnp.clip(s.ov_par, 0, n - 1)
        key = par * 8 + jnp.clip(s.seq, 0, 7)
        cnt = jnp.zeros(8 * n, I32).at[key].add(ov.astype(I32))
        cs = jnp.cumsum(cnt) - cnt
        return s._replace(ov_par=jnp.where(gone, -1, s.ov_par), seq=jnp.where(ov, cs[key] - cs[par * 8], s.seq))
    return _cond(gone.any(), fix, lambda s: s, s)


def drop_overlays(s):
    """Les matériaux des Xyz qui ont quitté le terrain (ailleurs que sur le terrain, déplacés après l'attachement : pas
    un Xyz de l'Extra Deck en cours d'invocation) vont au cimetière de leur propriétaire, d'un coup (ygopro : send_to
    des overlays après le déplacement du groupe), triés par contrôleur, seq du Xyz puis rang. Appelé par refresh : une
    seule copie dans le graphe, au lieu d'une par appel de group_send."""
    par = jnp.clip(s.ov_par, 0, s.cid.shape[0] - 1)
    ovm = (s.zone == Z_OVERLAY) & (s.ov_par >= 0) & (s.zone[par] != Z_MZONE) & (s.mv[par] > s.mv)

    def drop(s):  # seulement s'il y en a : refresh est appelé plus souvent dans step que dans step_batch (clk, ev_p)
        s2 = s._replace(seq=jnp.where(ovm, s.seq[par] * 8 + s.seq, s.seq), ctrl=jnp.where(ovm, s.ctrl[par], s.ctrl))
        s2 = group_send(s2, ovm, Z_GRAVE, P_GRAVE, reason=REASON_RULE | REASON_LOST_TARGET)
        return raise_ev(s2._replace(ov_par=jnp.where(ovm, -1, s2.ov_par)), EV_TO_GRAVE, ovm, s.tp)
    return _cond(ovm.any(), drop, lambda s: s, s)


def attach(s, sel, par):
    """Les cartes sel deviennent les matériaux du Xyz par, après ceux qu'il a déjà, dans l'ordre de création (card_set
    trié par cardid : Duel.Overlay avec envoi des matériaux au cimetière ne trie pas). O(N)."""
    n = s.cid.shape[0]
    cnt = jnp.zeros(n, I32).at[s.uid].add(sel.astype(I32))
    rank = (jnp.cumsum(cnt) - cnt)[s.uid]
    base = ((s.zone == Z_OVERLAY) & (s.ov_par == par)).sum()
    return s._replace(pzone=jnp.where(sel, s.zone, s.pzone), zone=jnp.where(sel, Z_OVERLAY, s.zone),
                      seq=jnp.where(sel, base + rank, s.seq), ov_par=jnp.where(sel, par, s.ov_par),
                      st=jnp.where(sel, 0, s.st), mv=jnp.where(sel, s.clk + 1, s.mv),
                      en_t=jnp.where(sel, s.clk + 1, s.en_t), clk=s.clk + 1)


def group_to_hand(bank, s, mask, reason=0):
    """Renvoi en main ; un monstre de l'Extra Deck retourne dans l'Extra Deck (verso)."""
    ex = mask & ((bank.ctype[s.cid] & TYPE_EXTRA) != 0)
    return group_send(s, mask, Z_HAND, P_HAND, Z_EXTRA, P_FD_DEF, alt=ex, reason=reason)


def group_to_grave(bank, s, mask, reason=0):
    """Envoi au cimetière. Un pendule qui quitte le terrain (zone monstres, ou zone magie/piège sans activation niée)
    va plutôt dans l'Extra Deck, recto, après les autres cartes (field::send_to, pas d'EVENT_TO_GRAVE)."""
    pen = (bank.ctype[s.cid] & TYPE_PENDULUM) != 0
    red = mask & pen & ((s.zone == Z_MZONE) | ((s.zone == Z_SZONE) & ((s.st & ST_ACTDIS) == 0)))
    return raise_ev(group_send(s, mask, Z_GRAVE, P_GRAVE, Z_EXTRA, P_FU_DEF, alt=red, reason=reason), EV_TO_GRAVE, mask & ~red, s.tp)


def selected_to_grave(bank, s, reason=0):
    return group_to_grave(bank, s, (s.st & ST_SEL) != 0, reason)._replace(picks=_i(0))


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
    """Point d'ajustement (Processors::Adjust d'ygopro, contrôle de victoire) : pioche dans un deck vide, LP à 0
    après un coût payé pendant une activation (la défaite n'est constatée qu'ici, après le choix des cibles)."""
    s = lose(s, s.tp, s.odraw)._replace(odraw=jnp.bool_(False))
    s = lose(s, 0, (s.lp[0] <= 0) & (s.lp[1] > 0))
    return lose(s, 1, (s.lp[1] <= 0) & (s.lp[0] > 0))


def raise_ev(s, ev, mask, p):
    return s._replace(ev_g=s.ev_g.at[ev].set(s.ev_g[ev] | mask), ev_p=s.ev_p.at[ev].set(p),
                      tev=s.tev.at[ev].set(s.tev[ev] | mask), tew=s.tew.at[ev].set(s.tew[ev] | mask))


def brk(s):
    """break_effect d'ygopro-core : les déclencheurs optionnels sans EFFECT_FLAG_DELAY (« lorsque », TRIG_WHEN) des
    événements précédents manquent le timing."""
    return s._replace(tew=jnp.zeros_like(s.tew))


def name_in(table, cid):
    """cid [...] présent dans le registre par nom `table` ?"""
    return (table == (cid + 1)[..., None]).any(-1)


def opt_key(cid, opt, oath=0):
    """Clé par nom : identifiant, index natif 8 bits et indicateur OATH."""
    k = jnp.where(opt >= OPT_NAMEK, opt - OPT_NAMEK + 1, 0)
    return jnp.where((opt == OPT_NAME) | (opt >= OPT_NAMEK), cid + (k << 16) + (_i(oath) << 24), -1)


def name_add(table, cid):
    """Ajoute un nom sans dupliquer les entrées ; les capacités couvrent tous les noms du duel."""
    i = jnp.where((table == 0).any(), jnp.argmax(table == 0), table.shape[0] - 1)
    return jnp.where(name_in(table, cid), table, table.at[i].set(cid + 1))


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


def act_ok(bank, s, h, e, p, enabled=None):
    """L'effet e de h est activable par p : coût + cible exécutables, pas d'interdiction ni de limite par nom."""
    cid = s.cid[h]
    # une carte aux effets annulés peut encore les activer (is_activateable ne teste pas STATUS_DISABLED) : ils sont
    # annulés à la résolution (load_link)
    def check():
        oath = bank.erefund[cid, e] if bank.erefund.shape[1] else 0
        key = opt_key(cid, bank.eopt[cid, e], oath)
        return (test_ok(bank, s, h, e, p, enabled=enabled) & ((bank.ekind[cid, e] == K_PROC) | ~name_in(s.lock_cid[p], cid))
                & ~((key >= 0) & name_in(s.opt_cid[p], key))
                & ~(((bank.eopt[cid, e] == OPT_CARD) | (bank.eopt[cid, e] == OPT_CARD_NR))
                    & ((s.st[h] & ((ST_CNT | ST_CNTN) << e)) != 0)))
    if enabled is None:
        return check()
    return lax.fori_loop(0, jnp.where(enabled, 1, 0), lambda _, previous: check(), jnp.bool_(True))


def trig_next(bank, s):
    """Diffère trig_eval à settle : sous vmap, l'appeler ici le calculerait à chaque instruction exécutée."""
    return s._replace(dec=_i(D_TRIG))


def trig_eval(bank, s):
    """Prochaine décision de mise en chaîne des déclencheurs (à partir de l'étape s.tst) ; s'il n'y en a plus,
    fenêtre de réponse (après la chaîne bâtie : l'adversaire du dernier maillon a la priorité)."""
    # étape SEGOC de chaque déclencheur en attente ; activabilité testée une seule fois pour les 4 étapes
    stage = (s.ctrl != s.tp) + 2 * (bank.ekind[s.cid, s.tg_e] == K_TRIG_O)
    def more(carry):
        remaining, _, rounds = carry
        return remaining.any() & (rounds < ACT_ROUNDS)
    def batch(carry):
        remaining, accepted, rounds = carry
        hs = jnp.nonzero(remaining, size=KC, fill_value=0)[0]
        valid = jnp.arange(KC) < remaining.sum()
        passed = jax.vmap(lambda h, active: act_ok(bank, s, h, s.tg_e[h], s.ctrl[h], enabled=active))(hs, valid) & valid
        accepted |= jnp.zeros_like(s.seq).at[hs].add(passed.astype(I32)) > 0
        return remaining.at[hs].set(False), accepted, rounds + 1
    pending = s.tg & (s.dec == D_TRIG) & ~s.done
    _, ok, _ = lax.while_loop(more, batch, (pending, jnp.zeros_like(s.tg), _i(0)))
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


def in_range(s):
    """[N] zone de chaque carte au sens des portées d'effet (erng) : main, cimetière, zone monstres ou magie/piège
    recto ; 0 ailleurs."""
    up = (s.pos & P_FACEUP) != 0
    return jnp.where((s.zone == Z_HAND) | (s.zone == Z_GRAVE) | (((s.zone == Z_MZONE) | (s.zone == Z_SZONE)) & up),
                     s.zone, 0)


def trig_point(bank, s):
    """Point d'événement, juste après l'ouverture d'une fenêtre W_QUICK (chaîne vide) : les effets déclenchés par
    les événements cumulés dans `tev` passent d'abord (SEGOC), la fenêtre ne s'ouvre qu'ensuite."""
    bits = (s.tev * (1 << jnp.arange(NEV))[:, None]).sum(0)
    wbits = (s.tew * (1 << jnp.arange(NEV))[:, None]).sum(0)
    abits, awbits = (s.tev.any(1) * (1 << jnp.arange(NEV))).sum(), (s.tew.any(1) * (1 << jnp.arange(NEV))).sum()
    zr = in_range(s)
    tg, tg_e = s.tg, s.tg_e
    for e in range(E):
        k = bank.ekind[s.cid, e]
        et = bank.etrig[s.cid, e]
        when = (k == K_TRIG_O) & ((et & TRIG_WHEN) != 0)
        fld = (et & TRIG_FIELD) != 0
        evs = jnp.where(fld, jnp.where(when, awbits, abits), jnp.where(when, wbits, bits))
        # effect::is_activateable : au damage step, un déclencheur FIELD optionnel sans EFFECT_FLAG_DAMAGE_STEP ne
        # répond qu'aux événements de combat (codes 1132-1149)
        ds_lock = (k == K_TRIG_O) & fld & (s.dstep != DS_NONE) & ((bank.edmg[s.cid, e] & DMG_STEP) == 0)
        evs = jnp.where(ds_lock, evs & ((1 << EV_BATTLE_DESTROYED) | (1 << EV_PRE_DMG)), evs)
        hit = ((k == K_TRIG_F) | (k == K_TRIG_O)) & ((et & evs) != 0) & ~tg & (~fld | ((zr & bank.erng[s.cid, e]) != 0))
        tg_e, tg = jnp.where(hit, e, tg_e).astype(I32), tg | hit
    s = s._replace(tg=tg, tg_e=tg_e, teg=s.tev, tev=jnp.zeros_like(s.tev), tew=jnp.zeros_like(s.tew), pt_p=s.dec_player, pt_passed=s.cw_passed, tst=_i(0))
    return _cond(tg.any(), lambda s: trig_next(bank, s), lambda s: s, s)


def keep_perm(s):
    """Fin de tour pour la table des modificateurs : seuls restent les TD_PERM encore sur une carte du terrain qui n'a
    pas bougé depuis, regroupés en tête (l'ordre d'application vient de tm_t, pas du rang)."""
    alive = (s.tm_m & (s.mv[None, :] <= s.tm_t[:, None]) & (s.zone == Z_MZONE)[None, :]).any(1)
    keep = (s.tm_dur == TD_PERM) & (jnp.arange(TM) < s.tm_n) & alive
    o = jnp.argsort(jnp.where(keep, 0, 1) * TM + jnp.arange(TM))
    k = keep[o]
    return s._replace(tm_code=jnp.where(k, s.tm_code[o], 0), tm_m=s.tm_m[o] & k[:, None], tm_v=s.tm_v[o],
                      tm_t=s.tm_t[o], tm_dur=s.tm_dur[o], tm_rd=s.tm_rd[o], tm_src=s.tm_src[o],
                      tm_n=keep.sum().astype(I32))


def start_turn(s):
    tp = jnp.where(s.turn == 0, 0, 1 - s.tp).astype(I32)
    s = s._replace(turn=s.turn + 1, tp=tp, ns_used=jnp.bool_(False), ps_used=jnp.bool_(False), st=s.st & ~ST_TURN_RESET, phase=_i(PH_DRAW),
                   tev=jnp.zeros_like(s.tev), tew=jnp.zeros_like(s.tew), opt_cid=jnp.zeros_like(s.opt_cid), lock_cid=jnp.zeros_like(s.lock_cid))
    s = keep_perm(s)
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


def declare(s, tgt, replay):
    """Déclaration ou relance de l'attaque (étapes 6-8) : cible, instantané des monstres adverses (opp_mzone) ;
    EVENT_ATTACK_ANNOUNCE seulement à la première déclaration, puis la fenêtre de l'attaque."""
    n = s.cid.shape[0]
    s = s._replace(atk_tgt=tgt, rb_m=(s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE), rb_t=s.clk, rpl=replay)
    r = raise_ev(s, EV_ATTACK, (jnp.arange(n) == s.atk) & ~replay, s.tp)
    return open_quick(r._replace(ev_p=jnp.where(replay, s.ev_p, r.ev_p)), s.tp, R_ATK1, chained=False)


def atk_rollback(s):
    """attack_rollback (adjust, étape 14, hors damage step) : les monstres adverses ont changé depuis la déclaration
    (arrivée, départ ou déplacement), ou l'attaque directe n'est plus possible (confirm_attack_target, sans le test de
    CANNOT_BE_BATTLE_TARGET). Calculé à la fermeture des fenêtres de l'attaque : ygopro-core le fige au premier adjust
    où il devient vrai, ce qui ne diffère que si la condition redevient fausse (attaque directe interdite puis
    permise de nouveau)."""
    opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
    same = opp & (s.mv <= s.rb_t)
    nodir = (s.atk_tgt < 0) & ((s.fl_c[s.atk] & FL_NO_DIRECT) != 0)
    return ((s.rb_m | opp) & ~same).any() | nodir


def replay_ask(s):
    """Étape 11, relance : sans cible possible ni attaque directe, l'attaque s'arrête (elle compte) ; sinon le joueur
    du tour choisit de la relancer (SelectYesNo 30 ; EFFECT_MUST_ATTACK n'existe pas dans le moteur)."""
    opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
    tg = (opp & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    dr = ~opp.any() & ((s.fl_c[s.atk] & FL_NO_DIRECT) == 0)
    return _cond(tg | dr, lambda s: s._replace(dec=_i(D_YESNO), dec_player=s.tp, zctx=_i(ZC_REPLAY)), battle_over, s)


def replay_go(s):
    """Relance acceptée (étape 4) : attaque directe s'il n'y a plus de monstre adverse (sans EFFECT_DIRECT_ATTACK, la
    question 31 « attaque directe ? » ne se pose jamais), sinon choix de la cible."""
    dr = ~((s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)).any()
    return _cond(dr, lambda s: declare(s, _i(-1), jnp.bool_(True)),
                 lambda s: s._replace(dec=_i(D_TARGET), dec_player=s.tp, rpl=jnp.bool_(True)), s)


def damage_step(bank, s):
    """Début du damage step (étapes 19-20) : première fenêtre."""
    start = lambda s: trig_point(bank, open_quick(s._replace(dstep=_i(DS_BEFORE), ds_t=s.clk, ds_flip=jnp.bool_(False),
                                                             rpl=jnp.bool_(False)),
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
    s = raise_ev(group_to_grave(bank, s, m, REASON_BATTLE | REASON_DESTROY), EV_BATTLE_DESTROYED, m, s.tp)
    return trig_point(bank, open_quick(s, s.tp, R_DMG2, chained=False))


def battle_over(s):
    return to_battle(s._replace(atk=_i(-1), atk_tgt=_i(-1), dstep=_i(DS_NONE), rpl=jnp.bool_(False)))


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
                        lambda s: _cond(atk_rollback(s), replay_ask,                # relance (étape 11)
                                        lambda s: open_quick(s, s.tp, R_ATK2, chained=False), s), s),
        # R_ATK2 (après une chaîne : la fenêtre de SolveChain, puis le damage step ; étape 11 : relance)
        lambda s: _cond(~atk_canceled(s) & atk_rollback(s), replay_ask, lambda s: damage_step(bank, s), s),
        lambda s: dmg_send(bank, s),                                                # R_DMG1
        battle_over,                                                                # R_DMG2
        lambda s: ds_next(bank, s, lambda s: ds_flip(bank, s)),                     # R_DS1
        lambda s: ds_next(bank, s, lambda s: ds_cal(bank, s)),                      # R_DS2
        lambda s: ds_next(bank, s, lambda s: ds_damage(bank, s)),                   # R_DS3
        end_phase,                                                                  # R_HAND
    ], clear_ev(s))


def pass_priority(bank, s):
    p = s.dec_player
    other = lambda s: s._replace(cw_passed=jnp.bool_(True), dec_player=1 - p)

    def close(s):
        return _cond(s.wk == W_PHASE, phase_end,
                        lambda s: _cond(s.n_chain > 0, lambda s: load_link(bank, s, s.n_chain - 1),
                                           lambda s: quick_end(bank, s), s), s)
    return _cond(s.cw_passed, close, other, s)


def place_pending(bank, s, z):
    c = s.pend
    s = leave_hand(s, c)
    pos = jnp.where(s.pend_set, P_FD_DEF, P_FU_ATK)
    s = move(s, c, Z_MZONE, z, s.tp, pos)
    s = s._replace(st=s.st.at[c].set(ST_SUMMON_TURN), ns_used=jnp.bool_(True), pend=_i(-1))
    s = brk(s)  # fin de l'invocation (après les libérations) : break_effect
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


def special_allowed(bank, s, proper=False):
    """Invocabilité courante par effet ; une vraie procédure contourne la seule limite de réanimation."""
    revive = (bank.erl[s.cid] == 0) | ((bank.erl[s.cid] == 1) &
             (s.proc | ((s.zone == Z_EXTRA) & ((s.pos & P_FACEUP) == 0))))
    return ((s.zone != Z_MZONE) & ((s.fl_c & FL_NO_SS) == 0)
            & ((bank.ctype[s.cid] & TYPE_MONSTER) != 0) & (revive | proper))


def reason_matches(actual, selector):
    """Positif : au moins un bit ; négatif : tous les bits demandés."""
    mask = jnp.abs(selector)
    return jnp.where(selector < 0, (actual & mask) == mask, (actual & mask) != 0)


def filter_mask(bank, s, f, p, h, bdes=True, targ=False):
    """Évalue un filtre (enregistrement fixe) pour le joueur p, carte source h. O(N).
    bdes : exclut les monstres détruits au combat (sélections) ; False pour les cibles des effets continus.
    targ : sélection avec ciblage (IsCanBeEffectTarget) : exclut les cartes qui ne peuvent pas être ciblées."""
    n = s.cid.shape[0]
    loc = jnp.where(s.ctrl == p, f[F_LOC_ME], f[F_LOC_OP])
    m = (s.zone & loc) != 0
    m &= (s.zone != Z_OVERLAY) | (s.ov_par == h)  # matériaux Xyz : seulement ceux de la carte source
    if bdes:
        m &= (s.st & ST_BDES) == 0  # détruite au combat : plus ciblable (ygopro is_capable_be_effect_target)
    m &= (f[F_POS] == 0) | ((s.pos & f[F_POS]) != 0)
    ct = bank.ctype[s.cid]
    ct = jnp.where(((ct & TYPE_PENDULUM) != 0) & (s.zone == Z_SZONE), P_SPELL, ct)  # zone pendule : magie pendule
    m &= (f[F_TYPE] == 0) | ((ct & f[F_TYPE]) != 0)
    m &= (ct & f[F_NOTTYPE]) == 0
    m &= ~((f[F_NOT_SELF] != 0) & (jnp.arange(n) == h))
    atk, df = s.atk_c, s.df_c
    m &= (f[F_ATTR] == 0) | ((bank.attr[s.cid] & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((bank.race[s.cid] & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match(bank, s.cid, f[F_SET])
    code = bank.code[s.cid]
    # F_CODE1 < 0 : tout sauf le nom -F_CODE1 (not c:IsCode(id))
    m &= jnp.where(f[F_CODE1] < 0, code != -f[F_CODE1], (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2]))
    m &= (f[F_ST] == 0) | jnp.where(f[F_ST] < 0, summon_matches(s.suminfo, f[F_ST]), (s.st & f[F_ST]) != 0)
    lv = bank.level[s.cid]
    nolv = (bank.ctype[s.cid] & (TYPE_XYZ | TYPE_LINK)) != 0  # sans niveau : seulement un filtre sans borne
    m &= jnp.where(nolv, (f[F_LV_LO] <= -BIG) & (f[F_LV_HI] >= BIG), (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI]))
    # IsCanBeSpecialSummoned : conditions courantes, et Link uniquement en attaque.
    m &= (f[F_SPSUM] == 0) | (special_allowed(bank, s)
                              & ~((f[F_SPSUM] == 2) & ((bank.ctype[s.cid] & TYPE_LINK) != 0)))
    # fusion / rituel : SummonEffFilter et Ritual.Filter demandent aussi IsCanBeSpecialSummoned (fl_c est à jour ici,
    # pas forcément quand fz_c / rz_c sont calculés)
    nss = (s.fl_c & FL_NO_SS) == 0
    m &= (f[F_FUS] == 0) | (s.fz_c & nss)
    m &= (f[F_RIT] == 0) | ((((s.rz_c >> jnp.maximum(f[F_RIT] - 1, 0)) & 1) != 0) & nss)
    m &= (f[F_NODIS] == 0) | ((s.fl_c & FL_DISABLED) == 0)
    m &= (f[F_REASON] == 0) | reason_matches(s.reason, f[F_REASON])
    m &= ((f[F_ZX] & ZX_COLINK) == 0) | s.cl_c
    m &= ((f[F_ZX] & ZX_MAIN) == 0) | (s.seq < N_MAIN_ZONES)
    need_extra = (f[F_ZX] & ZX_EXTRA) != 0
    extra_space = lax.fori_loop(0, need_extra.astype(I32), lambda _, previous: ex_zones(s, p).any(), jnp.bool_(True))
    faceup_pend = (s.zone == Z_EXTRA) & ((ct & TYPE_PENDULUM) != 0) & ((s.pos & P_FACEUP) != 0)
    need_pend = need_extra & (m & faceup_pend).any()
    pend_space = lax.fori_loop(0, need_pend.astype(I32),
        lambda _, previous: pend_extra_zones(bank, s, p).any(), jnp.bool_(True))
    m &= jnp.where(faceup_pend, pend_space, extra_space)
    if targ:
        m &= ~untargetable(bank, s, h)
    # un monstre Link n'a pas de DEF (IsDefense* faux) : seulement un filtre sans borne
    nodf = (bank.ctype[s.cid] & TYPE_LINK) != 0
    dok = jnp.where(nodf, (f[F_DEF_LO] <= -BIG) & (f[F_DEF_HI] >= BIG), (df >= f[F_DEF_LO]) & (df <= f[F_DEF_HI]))
    return m & (atk >= f[F_ATK_LO]) & (atk <= f[F_ATK_HI]) & dok


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
    if bank.submat.shape[1]:
        act &= md[..., M_CODE] != MC_FUS_SUB  # propriété passive, aucun slot de stats()/KM
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
    # un modificateur posé par une carte sur elle-même est suspendu tant qu'elle est annulée (effect::is_available :
    # owner == handler, pas EFFECT_FLAG_CANNOT_DISABLE)
    ttgt &= ~((jnp.arange(s.cid.shape[0])[None, :] == s.tm_src[:, None]) & dis[None, :]
              & (s.tm_code != MC_DISABLE)[:, None])
    # immunité : les effets continus de piège ne s'appliquent pas aux monstres immunisés
    imm = (ctgt & (ccode == MC_IMM_TRAP)[:, None]).any(0)
    ctgt &= ~(ctrap[:, None] & imm[None, :] & (ccode != MC_IMM_TRAP)[:, None] & (ccode != MC_DISABLE)[:, None])
    craw &= ~(ctrap[:, None] & imm[None, :] & onf[None, :])
    fl = (jnp.where(dis, FL_DISABLED, 0) | jnp.where(onf & faceup, FL_ONF, 0)).astype(I32)
    for c, b in ((MC_NO_ATTACK, FL_NO_ATTACK), (MC_NO_BTARGET, FL_NO_BTARGET), (MC_NO_ETARGET, FL_NO_ETARGET),
                 (MC_IMM_TRAP, FL_IMM_TRAP), (MC_NO_DIRECT, FL_NO_DIRECT), (MC_INDES_EFF, FL_INDES_EFF)):
        # continus et temporaires (EFFECT_CANNOT_ATTACK posé par une résolution, Silent Doom…)
        fl |= jnp.where((ctgt & (ccode == c)[:, None]).any(0) | (ttgt & (s.tm_code == c)[:, None]).any(0), b, 0)
    fl |= jnp.where((craw & (ccode == MC_NO_SS)[:, None]).any(0), FL_NO_SS, 0)  # carte visée, où qu'elle soit
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
    ib = jnp.where(tgt & (code == MC_INDES_BATTLE)[:, None], val, BIG).min(0)
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
    df = jnp.where((bank.ctype[s.cid] & TYPE_LINK) != 0, 0, df)  # card::get_defense : toujours 0 pour un Link
    return (atk.astype(I32), df.astype(I32), batk.astype(I32), fl, ib.astype(I32), pf.astype(I32), cav, cav_t, clk,
            tm_m, act.sum() > KM)


def stats_key(s):
    """Entrées de stats() qui changent pendant la partie (hors ses propres sorties) : si elles n'ont pas bougé, refresh
    est inutile (il est idempotent sur un état inchangé)."""
    return (s.zone, s.pos, s.st, s.ctrl, s.phase, s.tp, s.turn, s.clk, s.tm_n, s.tm_m, s.reason, s.suminfo)


def changed(prev, s):
    """[...] : les entrées de stats() ont changé entre prev et s (par partie si les états sont en lot)."""
    nb = s.done.shape
    d = jnp.zeros(nb, bool)
    for x, y in zip(stats_key(prev), stats_key(s)):
        d |= (x != y).reshape(nb + (-1,)).any(-1)
    return d


def cont_reads_stats(bank):
    """La banque a un effet continu dont un filtre (cible ou compte) borne l'ATK ou la DEF."""
    f = bank.filt[:, :, :2]
    bounded = ((f[..., F_ATK_LO] > -BIG) | (f[..., F_ATK_HI] < BIG) | (f[..., F_DEF_LO] > -BIG) | (f[..., F_DEF_HI] < BIG))
    return ((bank.ekind == K_CONT)[..., None] & bounded).any()


TB = 16  # seuils d'ATK / DEF distincts gardés par borne (au-delà : test conservateur, toute ATK / DEF changée)


def cont_bounds(bank):
    """Seuils des filtres d'effets continus de la banque : [4, TB] (ATK_LO, ATK_HI, DEF_LO, DEF_HI), bourrés avec
    ±BIG (jamais franchis), et un drapeau de débordement."""
    f = bank.filt[:, :, :2]
    cont = (bank.ekind == K_CONT)[..., None]
    out, over = [], jnp.bool_(False)
    for i, lo in ((F_ATK_LO, True), (F_ATK_HI, False), (F_DEF_LO, True), (F_DEF_HI, False)):
        v = f[..., i]
        ok = cont & ((v > -BIG) if lo else (v < BIG))
        u = jnp.where(ok, v, BIG if lo else -BIG).reshape(-1)
        u = jnp.sort(u) if lo else -jnp.sort(-u)
        first = jnp.concatenate([jnp.ones(1, bool), u[1:] != u[:-1]])      # valeurs distinctes en tête
        idx = jnp.nonzero(first, size=TB + 1, fill_value=u.shape[0] - 1)[0]
        w = u[idx]
        over |= (w[TB] != (BIG if lo else -BIG))
        out.append(w[:TB])
    return jnp.stack(out), over


def stats_moved(bank, a, b):
    """[...] : une ATK / DEF en cache a franchi un seuil d'un filtre d'effet continu entre a et b (par partie si les
    états sont en lot) ; seul cas où la deuxième passe peut changer un résultat."""
    nb = a.done.shape
    th, over = cont_bounds(bank)
    flat = lambda x: x.reshape(nb + (-1, 1))
    oa, na, od, nd = flat(a.atk_c), flat(b.atk_c), flat(a.df_c), flat(b.df_c)
    x = (((oa >= th[0]) != (na >= th[0])).any((-1, -2)) | ((oa <= th[1]) != (na <= th[1])).any((-1, -2))
         | ((od >= th[2]) != (nd >= th[2])).any((-1, -2)) | ((od <= th[3]) != (nd <= th[3])).any((-1, -2)))
    moved = ((a.atk_c != b.atk_c) | (a.df_c != b.df_c)).reshape(nb + (-1,)).any(-1)
    return jnp.where(over, moved, x)


# Deuxième passe de refresh. Les filtres des effets continus lisent atk_c / df_c, le résultat du refresh précédent ;
# ygopro les évalue avec l'ATK courante (get_attack) : un bonus qui vient de s'appliquer (changement de phase, carte
# arrivée) compte déjà pour « ne peut pas attaquer s'il a 1500 ATK ou plus ». Si la banque a de tels filtres et que
# le refresh a changé une ATK / DEF, on le refait une fois (sans fusion / rituel, comme refresh_b). Une seule passe
# de plus : une dépendance circulaire (bonus d'ATK filtré sur l'ATK) n'est pas itérée jusqu'au point fixe.
# Une seconde passe complète de stats() à chaque refresh coûtait 16 % du débit CPU (banque procédurale 1, B=64) :
# dans step_batch, elle ne porte que sur le sous-lot des parties dont une ATK / DEF a bougé.
def refresh2(bank, s):
    """refresh, puis la deuxième passe si nécessaire (une partie ; sous vmap, le cond calcule les deux branches)."""
    s1 = refresh(bank, s)
    need = cont_reads_stats(bank) & stats_moved(bank, s, s1)
    return lax.cond(need, lambda x: refresh(bank, x, fus=False), lambda x: x, s1)


def refresh_if(bank, prev, s):
    """refresh seulement si l'état a changé depuis prev (une partie ; sous vmap, le cond calcule les deux branches)."""
    return lax.cond(changed(prev, s), lambda s: refresh2(bank, s), lambda s: s, s)


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
    s1 = lax.cond(d.sum() > k, ref, lambda s: lax.cond(d.any(), sub, lambda s: s, s), s)
    return lax.cond(cont_reads_stats(bank), lambda s1: _second_b(bank, s, s1, k), lambda s1: s1, s1)


def _second_b(bank, s, s1, k):
    """Deuxième passe de refresh2 en lot : seulement les parties dont une ATK / DEF a bougé (sous-lot de k)."""
    nb = s.done.shape[0]
    d = stats_moved(bank, s, s1)
    ref = lambda x: jax.vmap(lambda y: refresh(bank, y, fus=False))(x)

    def sub(x):
        idx = jnp.nonzero(d, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda y: y[jnp.minimum(idx, nb - 1)], x))
        return jax.tree.map(lambda y, z: y.at[idx].set(z, mode="drop"), x, part)
    return lax.cond(d.sum() > k, lambda x: _where(d, ref(x), x), lambda x: lax.cond(d.any(), sub, lambda x: x, x), s1)


SUBS = [[(q >> j) & 1 for j in range(FK)] for q in range(1 << FK)]  # sous-ensembles des emplacements de matériaux


MAT_FIELDS = (F_TYPE, F_NOTTYPE, F_ATTR, F_RACE, F_SET, F_CODE1, F_CODE2, F_LV_LO, F_LV_HI)  # champs permis dans une recette


def mat_mask(bank, s, f):
    """Filtre de matériau allégé (seulement les champs MAT_FIELDS, sans la zone) : moins cher que filter_mask."""
    ct, code, lv = bank.ctype[s.cid], bank.code[s.cid], bank.level[s.cid]
    m = (f[F_TYPE] == 0) | ((ct & f[F_TYPE]) != 0)
    m &= (ct & f[F_NOTTYPE]) == 0
    m &= (f[F_ATTR] == 0) | ((bank.attr[s.cid] & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((bank.race[s.cid] & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match(bank, s.cid, f[F_SET])
    # F_CODE1 < 0 : tout sauf le nom -F_CODE1 (not c:IsCode(id))
    m &= jnp.where(f[F_CODE1] < 0, code != -f[F_CODE1], (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2]))
    nolv = (ct & (TYPE_XYZ | TYPE_LINK)) != 0
    return m & jnp.where(nolv, (f[F_LV_LO] <= -BIG) & (f[F_LV_HI] >= BIG), (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI]))


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
    if bank.submat.shape[1]:
        return fusion_sub_ok(bank, s)
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
    if bank.submat.shape[1]:
        return fusion_sub_picks(bank, s, f)
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



def fusion_sub_cards(bank, s, f):
    """SINGLE non SINGLE_RANGE : disponible aussi face cachée, sauf annulation effective."""
    return ((bank.submat[s.cid, 0] & s.zone) != 0) & ((bank.ctype[s.cid] & TYPE_MONSTER) != 0) & (s.ctrl == s.owner[f]) & \
           ((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & ((s.fl_c & FL_DISABLED) == 0) & ((s.st & ST_BDES) == 0)


def fusion_sub_ok(bank, s):
    from fusion import completions
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (bank.fmn[s.cid] > 0)
    idx = jnp.nonzero(want, size=EXK, fill_value=0)[0]
    live = jnp.arange(EXK) < want.sum()
    def one(f):
        cand = fus_cands(bank, s, f)
        _, any_sub, _, field_sub = completions(cand, fusion_sub_cards(bank, s, f), s.zone == Z_MZONE,
                                              bank.fsub[s.cid[f]], jnp.array([-1], I32))
        use = (1 << bank.fmn[s.cid[f]]) - 1
        return jnp.where(ex_zones(s, s.owner[f]).any(), any_sub[use, 0], field_sub[use, 0])
    ok = jax.vmap(one)(idx) & live
    return jnp.zeros(n, bool).at[jnp.where(live, idx, n)].set(ok, mode="drop"), want.sum() > EXK


def fusion_sub_picks(bank, s, f):
    from fusion import completions
    n, k = s.cid.shape[0], bank.fmn[s.cid[f]]
    cand = fus_cands(bank, s, f)
    subs = fusion_sub_cards(bank, s, f)
    eligible = (bank.fsub[s.cid[f]] & (1 << jnp.arange(FK))) != 0
    sel = (s.st & ST_SEL) != 0
    t = sel.sum()
    chosen = jnp.nonzero(sel, size=FK, fill_value=0)[0]
    field = s.zone == Z_MZONE
    plain, enhanced, plain_field, enhanced_field = completions(cand & ~sel[None, :], subs & ~sel, field,
                                                               bank.fsub[s.cid[f]], jnp.arange(n))
    room = ex_zones(s, s.owner[f]).any() | (sel & field).any()
    def assignment(pm):
        pos = jnp.arange(FK)
        ordinary = cand[pm, chosen]
        alternative = eligible[pm] & subs[chosen]
        ok_s = ((pos >= t) | ordinary | alternative).all()
        used = ((pos < t) & ~ordinary).sum()
        slot_c = pm[jnp.minimum(t, FK - 1)]
        rest = jnp.where((pos > t) & (pm < k), 1 << pm, 0).sum()
        budget = used + (~cand[slot_c]).astype(I32)
        can = cand[slot_c] | (eligible[slot_c] & subs)
        complete = jnp.where(budget == 0, enhanced[rest], plain[rest])
        complete_field = jnp.where(budget == 0, enhanced_field[rest], plain_field[rest])
        return ok_s & (t < k) & (slot_c < k) & can & (budget <= 1) & jnp.where(room | field, complete, complete_field)
    return jax.vmap(assignment)(jnp.asarray(PERMS)).any(0) & ~sel


def ex_zones(s, p):
    """[7] : cases où p peut invoquer un monstre de l'Extra Deck (MR5 sans Link : les 5 cases principales libres ; les
    zones extra 5 et 6 si p n'en occupe aucune et que l'adversaire n'occupe pas la case symétrique)."""
    me, op = zone_used(s, p, Z_MZONE), zone_used(s, 1 - p, Z_MZONE)
    emz = ~me[5] & ~me[6]
    return jnp.concatenate([~me[:5], jnp.stack([emz & ~op[6], emz & ~op[5]])])


def pend_extra_zones(bank, s, p):
    """Pendule recto de l'Extra Deck : cases principales désignées et une zone extra."""
    ez = ex_zones(s, p)
    return jnp.concatenate([ez[:N_MAIN_ZONES] & link_arrows(bank, s, p).any(0), ez[N_MAIN_ZONES:]])


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
            & ((s.st & ST_BDES) == 0) & (bank.level[s.cid] > 0) & ((bank.ctype[s.cid] & (TYPE_XYZ | TYPE_LINK)) == 0))


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


def syn_mats(bank, s, c):
    """[N], [N] : Syntoniseurs et non-Syntoniseurs utilisables pour le monstre synchro c (monstres face recto du terrain
    de son propriétaire, de niveau >= 1, pas Xyz ni Link ; filtres de la banque)."""
    ct, lv = bank.ctype[s.cid], bank.level[s.cid]
    base = ((s.zone == Z_MZONE) & (s.ctrl == s.owner[c]) & ((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0)
            & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0) & (lv >= 1) & ((s.st & ST_BDES) == 0))
    tun = (ct & TYPE_TUNER) != 0
    f = bank.smat[s.cid[c]]
    return base & tun & mat_mask(bank, s, f[0]), base & ~tun & mat_mask(bank, s, f[1])


def reach(h, lo=0):
    """Masque de bits des sommes atteignables (< SMAX) avec au moins lo (0 ou 1) cartes de l'histogramme de niveaux h
    [LVMAX + 1]. Effectifs pris par paquets 1, 2, 4... O(LVMAX × log SMAX)."""
    full = (1 << SMAX) - 1
    A, B = jnp.int32(1), jnp.int32(0)  # A : aucune carte ; B : au moins une
    for L in range(LVMAX, 0, -1):
        cmax = SMAX // L
        c = jnp.minimum(h[L], cmax)
        for i in range(cmax.bit_length()):
            k = jnp.clip(c - (2 ** i - 1), 0, 2 ** i)
            B = jnp.where(k > 0, B | (((A | B) << (k * L)) & full), B)
    return jnp.where(lo > 0, B, A | B)


def bit(x, i):
    return ((x >> jnp.clip(i, 0, 31)) & 1).astype(bool) & (i >= 0) & (i < SMAX)


def synchro_ok(bank, s):
    """[N] sz_c : monstres synchro de l'Extra Deck invocables maintenant : un Syntoniseur de niveau t et des
    non-Syntoniseurs (au moins un) de niveaux de somme exactement L - t. Les matériaux étant sur le terrain, une case
    se libère toujours. O(EXK × (N + LVMAX log SMAX))."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (bank.syn[s.cid] != 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    lvs = jnp.arange(LVMAX + 1)
    lv = jnp.clip(bank.level[s.cid], 0, LVMAX)

    def one(c, ok):
        t, nt = syn_mats(bank, s, c)
        oh = lv[None, :] == lvs[:, None]
        tl = (oh & t[None, :]).any(1)                              # [LVMAX + 1] niveaux de Syntoniseurs présents
        b = reach((oh & nt[None, :]).sum(1, dtype=I32), 1)
        L = bank.level[s.cid[c]]
        return ok & (tl & jax.vmap(lambda l: bit(b, L - l))(lvs)).any()
    oks = jax.vmap(one)(idx, live)
    return jnp.zeros(n, bool).at[jnp.where(live, idx, n)].set(oks, mode="drop")


def syn_picks(bank, s, c):
    """[N] : matériaux qu'on peut encore choisir pour le monstre synchro c : d'abord le Syntoniseur (il reste des
    non-Syntoniseurs qui complètent la somme), puis les non-Syntoniseurs (le reste de la somme reste atteignable)."""
    t, nt = syn_mats(bank, s, c)
    sel = (s.st & ST_SEL) != 0
    lv = jnp.clip(bank.level[s.cid], 0, LVMAX)
    lvs = jnp.arange(LVMAX + 1)
    oh = lv[None, :] == lvs[:, None]
    h = (oh & (nt & ~sel)[None, :]).sum(1, dtype=I32)
    L = bank.level[s.cid[c]]
    u = jnp.where(sel, lv, 0).sum()
    has_t = (sel & t).any()
    b1 = reach(h, 1)
    tun = jax.vmap(lambda l: bit(b1, L - l))(lvs)                  # Syntoniseur de niveau l
    ntv = jax.vmap(lambda l: bit(reach(h - (lvs == l), 0), L - u - l))(lvs)  # non-Syntoniseur de niveau l
    return jnp.where(has_t, nt & ~sel & ntv[lv], t & ~sel & tun[lv])


def syn_done(bank, s):
    sel = (s.st & ST_SEL) != 0
    return jnp.where(sel, jnp.clip(bank.level[s.cid], 0, LVMAX), 0).sum() >= bank.level[s.cid[s.pend]]


def has_synchro(bank, s):
    return ((s.zone == Z_EXTRA) & (bank.syn[s.cid] != 0)).any()


def refresh_syn(bank, s):
    return s._replace(sz_c=synchro_ok(bank, s))


def xyz_mats(bank, s, c):
    """[N] : matériaux possibles du Xyz c : monstres face recto du terrain de son propriétaire, de niveau égal à son
    rang (pas Xyz ni Link : sans niveau), filtre de la banque."""
    ct, lv = bank.ctype[s.cid], bank.level[s.cid]
    return ((s.zone == Z_MZONE) & (s.ctrl == s.owner[c]) & ((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0)
            & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0) & (lv == bank.level[s.cid[c]]) & ((s.st & ST_BDES) == 0)
            & mat_mask(bank, s, bank.xmat[s.cid[c]]))


def xyz_ok(bank, s):
    """[N] xz_c : Xyz de l'Extra Deck invocables (au moins xct matériaux possibles). O(EXK × N)."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (bank.xct[s.cid] > 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    oks = jax.vmap(lambda c, ok: ok & (xyz_mats(bank, s, c).sum() >= bank.xct[s.cid[c]]))(idx, live)
    return jnp.zeros(n, bool).at[jnp.where(live, idx, n)].set(oks, mode="drop")


def has_xyz(bank, s):
    return ((s.zone == Z_EXTRA) & (bank.xct[s.cid] > 0)).any()


def refresh_xyz(bank, s):
    return s._replace(xz_c=xyz_ok(bank, s))


SUB7 = [[(q >> z) & 1 == 1 for z in range(7)] for q in range(128)]  # sous-ensembles des 7 cases monstre
PAIRS7 = ([i for i in range(7) for j in range(i + 1, 7)], [j for i in range(7) for j in range(i + 1, 7)])  # 21 paires de cases


def link_arrows(bank, s, p):
    """[N, 5] : cases principales de p désignées par les flèches de chaque monstre Link du terrain (card::get_linked_zone,
    field::get_linked_zone) : les siens (gauche / droite depuis une case principale, bas depuis une zone extra) et ceux
    de l'adversaire en zone extra (flèches du haut). Les flèches vers les zones extra ne comptent pas : elles sont
    toujours dans les cases permises."""
    mk = bank.lmark[s.cid]
    on = (s.zone == Z_MZONE) & ((bank.ctype[s.cid] & TYPE_LINK) != 0) & ((s.pos & P_FACEUP) != 0)
    q = s.seq[:, None]
    j = jnp.arange(5)[None, :]
    has = lambda b: ((mk & b) != 0)[:, None]
    e = 2 * (q - 5)  # zone extra 5 : cases 0-2 ; 6 : cases 2-4
    own = (((q < 5) & (((j == q - 1) & has(LM_L)) | ((j == q + 1) & has(LM_R))))
           | ((q >= 5) & (((j == e) & has(LM_BL)) | ((j == e + 1) & has(LM_B)) | ((j == e + 2) & has(LM_BR)))))
    o = 4 - e  # zone extra 5 de l'adversaire : haut-gauche → ma case 4, haut → 3, haut-droite → 2 ; zone 6 : 2, 1, 0
    opp = (q >= 5) & (((j == o) & has(LM_TL)) | ((j == o - 1) & has(LM_T)) | ((j == o - 2) & has(LM_TR)))
    return on[:, None] & jnp.where((s.ctrl == p)[:, None], own, opp)


def link_zones(bank, s, p, c):
    """[7] : cases où p peut invoquer le monstre Link c (get_rule_zone_fromex) : cases principales libres désignées par
    une flèche, les zones extra (règle de ex_zones), et l'autre zone extra par Extra Link."""
    ez = ex_zones(s, p)
    e, x = extra_link(bank, s, p, jnp.zeros((1, 7), bool))
    ok = (arrows_at(bank.lmark[s.cid[c]], e[0]) & x[0]) != 0
    return jnp.concatenate([ez[:N_MAIN_ZONES] & link_arrows(bank, s, p).any(0),
                            ez[N_MAIN_ZONES:] | (ok & (jnp.arange(5, 7) == e[0]))])


def pend_info(bank, s, p):
    """Invocation pendule de p (proc_pendulum.lua) : (carte de l'échelle gauche, invocation possible, candidats de la
    main, candidats de l'Extra Deck, ft1, ft2, ft, cases désignées libres, zone extra utilisable). Candidats : monstres
    de la main et pendules recto de l'Extra Deck de niveau strictement entre les échelles, invocables spécialement ;
    ceux de la main seulement avec une case principale libre (ft1 > 0), ceux de l'Extra Deck seulement avec une case
    permise (ft2 > 0 : cases principales libres désignées par une flèche, plus une zone extra). ft : cases utilisables
    (GetUsableMZoneCount). ft1 et ft2 sont bornés par le nombre de candidats (Pendulum.Operation)."""
    ct = bank.ctype[s.cid]
    pen = (ct & TYPE_PENDULUM) != 0
    sz = (s.zone == Z_SZONE) & (s.ctrl == p) & pen
    lm, rm = sz & (s.seq == 0), sz & (s.seq == 4)
    l, r = jnp.argmax(lm), jnp.argmax(rm)
    a, b = bank.lsc[s.cid[l]], bank.rsc[s.cid[r]]
    lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
    lv = bank.level[s.cid]
    ok = ((lv > lo) & (lv < hi) & ((ct & TYPE_MONSTER) != 0) & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0)
          & ((s.fl_c & FL_NO_SS) == 0) & ((bank.erl[s.cid] == 0) | ((bank.erl[s.cid] == 1) & s.proc)) & (s.ctrl == p))
    ez = ex_zones(s, p)
    free = ez[:N_MAIN_ZONES]
    lz = free & link_arrows(bank, s, p).any(0)
    emz = ez[N_MAIN_ZONES] | ez[N_MAIN_ZONES + 1]
    f1, f2 = free.sum(), lz.sum() + emz
    ch = ok & (s.zone == Z_HAND) & (f1 > 0)
    cx = ok & (s.zone == Z_EXTRA) & (s.pos == P_FU_DEF) & pen & (f2 > 0)
    can = lm.any() & rm.any() & ~s.ps_used & (ch | cx).any()
    return l, can, ch, cx, jnp.minimum(f1, ch.sum()), jnp.minimum(f2, cx.sum()), f1 + emz, lz, emz


def pend_zones(bank, s, p, c, lz):
    """[7] : cases de la carte c de l'invocation pendule (operations.cpp, étape 23). ct1 / ct2 : cases principales libres
    / cases permises depuis l'Extra Deck, moins les cartes restantes de la main / de l'Extra Deck (c comprise). Une carte
    de la main évite les cases désignées si ct2 = 0 ; une carte de l'Extra Deck va en zone extra si ct1 = 0."""
    ez = ex_zones(s, p)
    free = ez[:N_MAIN_ZONES]
    rest = (s.st & ST_SEL) != 0
    ct1 = free.sum() - (rest & (s.zone == Z_HAND)).sum()
    ct2 = lz.sum() + (ez[N_MAIN_ZONES] | ez[N_MAIN_ZONES + 1]) - (rest & (s.zone == Z_EXTRA)).sum()
    no2 = jnp.zeros(2, bool)
    return jnp.where(s.zone[c] == Z_EXTRA, jnp.concatenate([lz & (ct1 != 0), ez[N_MAIN_ZONES:]]),
                     jnp.concatenate([free & ~((ct2 == 0) & lz), no2]))


def arrows_at(mk, q):
    """Cases désignées par un monstre Link de flèches mk en case q (0-4 principales, 5-6 zones extra), vues de son
    contrôleur : masque de 12 bits (0-4 ses cases principales, 5-6 ses zones extra, 7-11 cases principales adverses).
    card::get_linked_zone sans les zones magie/piège."""
    h = lambda b: (mk & b) != 0
    bit = lambda i: jnp.left_shift(jnp.int32(1), jnp.clip(i, 0, 11))
    main, ex = q < 5, q >= 5
    e2 = 2 * (q - 5)
    return (jnp.where(main & (q > 0) & h(LM_L), bit(q - 1), 0) | jnp.where(main & (q < 4) & h(LM_R), bit(q + 1), 0)
            | jnp.where(((q == 0) & h(LM_TR)) | ((q == 1) & h(LM_T)) | ((q == 2) & h(LM_TL)), 1 << 5, 0)
            | jnp.where(((q == 2) & h(LM_TR)) | ((q == 3) & h(LM_T)) | ((q == 4) & h(LM_TL)), 1 << 6, 0)
            | jnp.where(ex & h(LM_BL), bit(e2), 0) | jnp.where(ex & h(LM_B), bit(e2 + 1), 0)
            | jnp.where(ex & h(LM_BR), bit(e2 + 2), 0) | jnp.where(ex & h(LM_TL), bit(11 - e2), 0)
            | jnp.where(ex & h(LM_T), bit(10 - e2), 0) | jnp.where(ex & h(LM_TR), bit(9 - e2), 0)).astype(I32)


def _swap12(m):
    """Masque de 12 cases vu de l'adversaire : cases principales échangées, zones extra 5 et 6 échangées."""
    return ((m & 0x1F) << 7) | ((m >> 7) & 0x1F) | (((m >> 5) & 1) << 6) | (((m >> 6) & 1) << 5)


def link_cells(bank, s, p):
    """Monstres Link recto du terrain sur les 12 cases vues de p (voir arrows_at) : case de chaque carte [N] (12 : aucune),
    cases désignées par chaque case [12], cases tenues par un Link [12], voisins co-liés de chaque case [12] (masques)."""
    on = (s.zone == Z_MZONE) & ((bank.ctype[s.cid] & TYPE_LINK) != 0) & ((s.pos & P_FACEUP) != 0)
    mine = s.ctrl == p
    q = s.seq
    m = arrows_at(bank.lmark[s.cid], q)
    m = jnp.where(mine, m, _swap12(m))
    cell = jnp.where(mine, q, jnp.where(q < 5, q + 7, 11 - q))
    cell = jnp.where(on, cell, 12)
    pt = jnp.zeros(12, I32).at[cell].add(m, mode="drop")                 # [12] cases désignées par chaque case
    lk = jnp.zeros(12, bool).at[cell].set(True, mode="drop")             # cases tenues par un Link recto
    j = jnp.arange(12)
    mut = (((pt[:, None] >> j[None, :]) & 1) != 0) & (((pt[None, :] >> j[:, None]) & 1) != 0)
    M = jnp.where(mut, 1 << j[None, :], 0).sum(1).astype(I32)          # [12] voisins co-liés
    return cell, pt, lk, M


def colinked(bank, s):
    """[N] : monstres Link co-liés à au moins un autre (card::get_mutual_linked_cards non vide). O(N + 144)."""
    cell, _, _, M = link_cells(bank, s, 0)
    return (cell < 12) & (M[jnp.clip(cell, 0, 11)] != 0)


def extra_link(bank, s, p, sub):
    """Extra Link (field::get_spsummonable_count_fromex_rule4, check_extra_link, card::is_extra_link_state), pour
    chaque sous-ensemble sub [S, 7] de cases de p dont les monstres partent (matériaux). Si p occupe une seule zone
    extra o et que l'autre zone e est libre (ni lui ni l'adversaire), un monstre Link posé en e y est permis quand sa
    composante de co-liens contient o. Sans cette carte, la composante de o est fixe ; la carte la rejoint par un
    voisin co-lié. Rend (e [S] : 5 ou 6, X [S] : masque des cases de la composante de o qui désignent e ; 0 si la
    règle ne s'applique pas) ; la carte c est permise en e si arrows_at(lmark[c], e) & X ≠ 0. Cases de 12 bits vues
    de p (voir arrows_at). O(N + S × 12 × 11)."""
    _, pt, lk, M = link_cells(bank, s, p)
    j = jnp.arange(12)
    me, op = zone_used(s, p, Z_MZONE)[:7], zone_used(s, 1 - p, Z_MZONE)[:7]
    left = me[None, :] & ~sub                                             # [S, 7]
    l5, l6 = left[:, 5], left[:, 6]
    pre = (l5 & ~l6 & ~op[5]) | (l6 & ~l5 & ~op[6])                      # zone extra symétrique de l'adversaire libre
    o = jnp.where(l5, 5, 6)
    e = 11 - o
    lkb = jnp.where(lk, 1 << j, 0).sum().astype(I32)
    subb = jnp.where(sub, 1 << jnp.arange(7), 0).sum(1).astype(I32)
    alive = lkb & ~subb                                                   # [S]
    # composante de o parmi les cases vivantes : 11 tours de propagation déroulés (opérations élémentaires sur [S],
    # fusionnables ; un fori_loop impose une boucle séquentielle, un produit matriciel entier est lent sur CPU)
    reach = (1 << o) & alive
    for _ in range(11):
        for a in range(12):
            reach = reach | jnp.where(((reach >> a) & 1) != 0, M[a], 0)
        reach = reach & alive
    back = jnp.where(((pt[None, :] >> e[:, None]) & 1) != 0, 1 << j[None, :], 0).sum(1).astype(I32)  # [S] désignent e
    return e.astype(I32), jnp.where(pre, reach & back & alive, 0).astype(I32)


def link_table(bank, s, p):
    """Matériaux Link de p, par case de sa zone monstre et par sous-ensemble des 7 cases (128). Rend : cases → cartes
    [7, N] ; nombre de cartes [128] ; sommes possibles [128] (masque de bits : chaque monstre compte 1, un Link de lien
    L > 1 compte 1 ou L, Link.GetLinkCount) ; contient un Link [128] ; sous-ensemble de monstres recto, et une case
    reste pour le monstre Link une fois ces matériaux partis (GetLocationCountFromEx avec les matériaux : ni leurs cases
    ni leurs flèches) [128] ; Extra Link par sous-ensemble (extra_link). O(7 N + 128 × 7 × 5 + 128 × 12 × 11)."""
    ct = bank.ctype[s.cid]
    mine = (s.zone == Z_MZONE) & (s.ctrl == p)
    oh = mine[None, :] & (s.seq[None, :] == jnp.arange(7)[:, None])      # [7, N]
    used = oh.any(1)
    ok = (oh & (((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0))[None, :]).any(1)
    islk = (oh & ((ct & TYPE_LINK) != 0)[None, :]).any(1)
    r = jnp.where(oh, bank.level[s.cid][None, :], 0).sum(1)
    arw = link_arrows(bank, s, p)
    own = (oh[:, :, None] & arw[None, :, :]).any(1)                       # [7, 5] flèches du monstre de chaque case
    opp = (~mine[:, None] & arw).any(0)                                  # [5]
    sub = jnp.asarray(SUB7)                                               # [128, 7]
    lk = opp[None, :] | (~sub[:, :, None] & own[None]).any(1)             # [128, 5]
    left = used[None, :] & ~sub
    opu = zone_used(s, 1 - p, Z_MZONE)
    emz = ~left[:, 5] & ~left[:, 6] & (~opu[6] | ~opu[5])
    zok = (lk & ~left[:, :N_MAIN_ZONES]).any(1) | emz
    m = jnp.ones(128, I32)
    for z in range(7):
        alt = jnp.where(islk[z] & (r[z] > 1), m << jnp.clip(r[z], 0, 15), 0)
        m = jnp.where(sub[:, z], ((m << 1) | alt) & 0xFFFF, m)
    hasl = (sub & islk[None, :]).any(1)
    base = ~(sub & ~ok[None, :]).any(1)
    ee, ex = extra_link(bank, s, p, sub)
    # deux matériaux de même nom (noms différents exigés : ldiff)
    cc = jnp.where(oh, bank.code[s.cid][None, :], 0).sum(1)              # [7]
    pi, pj = jnp.asarray(PAIRS7[0]), jnp.asarray(PAIRS7[1])
    same = used[pi] & used[pj] & (cc[pi] == cc[pj])                       # [21]
    dup = (sub[:, pi] & sub[:, pj] & same[None, :]).any(1)
    return oh, sub.sum(1), m, hasl, zok & base, ee, jnp.where(base, ex, 0), dup


def link_valid(bank, s, c, tab):
    """[128] : ensembles de matériaux valides pour le monstre Link c (Link.CheckGoal) : filtre, nombre de matériaux,
    somme égale au lien, un Link parmi eux si demandé, une case pour c."""
    oh, cnt, m, hasl, zok, ee, ex, dup = tab
    k = s.cid[c]
    mm = (oh & mat_mask(bank, s, bank.lmat[k])[None, :]).any(1)          # [7]
    sub = jnp.asarray(SUB7)
    zok = zok | ((arrows_at(bank.lmark[k], ee) & ex) != 0)               # Extra Link
    return (zok & ~(sub & ~mm[None, :]).any(1) & (cnt >= bank.lmin[k]) & (cnt <= bank.lmax[k])
            & (((m >> jnp.clip(bank.level[k], 0, 15)) & 1) != 0) & ((bank.lneed[k] == 0) | hasl)
            & ((bank.ldiff[k] == 0) | ~dup))


def link_ok(bank, s):
    """[N] lz_c : monstres Link de l'Extra Deck du joueur qui décide invocables maintenant. O(EXP × (N + 128 × 7))."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (bank.lmax[s.cid] > 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    tab = link_table(bank, s, s.dec_player)

    def one(c, ok):
        return ok & link_valid(bank, s, c, tab).any()
    oks = jax.vmap(one)(idx, live)
    return jnp.zeros(n, bool).at[jnp.where(live, idx, n)].set(oks, mode="drop")


def link_picks(bank, s, c):
    """([N], bool) : matériaux qu'on peut encore choisir pour le monstre Link c (un ensemble valide contient les choix
    faits et ce candidat : Link.CheckRecursive) ; la sélection faite est elle-même valide (« terminer »)."""
    tab = link_table(bank, s, s.owner[c])
    v = link_valid(bank, s, c, tab)
    oh = tab[0]
    sel = (oh & ((s.st & ST_SEL) != 0)[None, :]).any(1)                  # [7]
    bits = 1 << jnp.arange(7, dtype=I32)
    smask = jnp.where(sel, bits, 0).sum()
    need = smask | bits
    q = jnp.arange(128, dtype=I32)
    can = ~sel & (v[None, :] & ((q[None, :] & need[:, None]) == need[:, None])).any(1)
    return (oh & can[:, None]).any(0), v[smask]


def has_link(bank, s):
    return ((s.zone == Z_EXTRA) & (bank.lmax[s.cid] > 0)).any()


def refresh_link(bank, s):
    return s._replace(lz_c=link_ok(bank, s))


def refresh_ex(bank, s):
    """Caches des invocations par procédure depuis l'Extra Deck (synchro, Xyz, Link) : seul legal les lit, pour le
    joueur qui décide et en décision libre (D_IDLE). Ils ne sont donc calculés qu'aux points de décision (fin de step /
    step_batch, new_game), seulement en D_IDLE et pour l'Extra Deck du joueur qui décide (faux ailleurs)."""
    idle = (s.dec == D_IDLE) & ~s.done
    s = lax.cond(idle & has_synchro(bank, s), lambda s: refresh_syn(bank, s), lambda s: s._replace(sz_c=s.sz_c & False), s)
    s = lax.cond(idle & has_xyz(bank, s), lambda s: refresh_xyz(bank, s), lambda s: s._replace(xz_c=s.xz_c & False), s)
    return lax.cond(idle & has_link(bank, s), lambda s: refresh_link(bank, s), lambda s: s._replace(lz_c=s.lz_c & False), s)


def refresh_ex_b(bank, s, mask, frac=8):
    """refresh_ex au niveau du lot, pour les parties mask seulement (regroupées dans un sous-lot de B/frac s'il y en a
    peu), chaque calcul seulement si l'une d'elles en a besoin."""
    nb = s.done.shape[0]
    k = max(1, nb // frac)

    def ref(s, m):
        for has, fn, fld in ((has_synchro, refresh_syn, "sz_c"), (has_xyz, refresh_xyz, "xz_c"), (has_link, refresh_link, "lz_c")):
            h = jax.vmap(lambda x: has(bank, x))(s) & (s.dec == D_IDLE) & ~s.done
            s = lax.cond((h & m).any(), lambda s: _where(h, jax.vmap(lambda x: fn(bank, x))(s), s), lambda s: s, s)
            s = s._replace(**{fld: getattr(s, fld) & h[:, None]})
        return s

    def sub(s):
        idx = jnp.nonzero(mask, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda x: x[jnp.minimum(idx, nb - 1)], s), idx < nb)
        return jax.tree.map(lambda x, y: x.at[idx].set(y, mode="drop"), s, part)
    full = lambda s: _where(mask, ref(s, mask), s)
    return lax.cond(mask.sum() > k, full, lambda s: lax.cond(mask.any(), sub, lambda s: s, s), s)


def refresh(bank, s, fus=True):
    """Recalcule les caches (stats, drapeaux, fusions réalisables ; pas ceux de refresh_ex). fus=False : sans fz_c (refresh_b le fait au
    niveau du lot, seulement si une partie a des monstres fusion dans son Extra Deck)."""
    s = drop_overlays(s)
    s = detach_fix(s)
    if fus:
        s = lax.cond(has_fusion(bank, s), lambda s: refresh_fus(bank, s), lambda s: s, s)
        s = lax.cond(has_ritual(bank, s), lambda s: refresh_rit(bank, s), lambda s: s._replace(rz_c=s.rz_c * 0), s)
    s = s._replace(cl_c=colinked(bank, s))
    atk, df, batk, fl, ib, pf, cav, cav_t, clk, tm_m, over = stats(bank, s)
    return s._replace(atk_c=atk, df_c=df, batk_c=batk, fl_c=fl, ib_c=ib, pf_c=pf, cav=cav, cav_t=cav_t, clk=clk, tm_m=tm_m,
                      ovf=s.ovf | jnp.where(over, 1, 0).astype(I32))


def _pure(bank, s, reg, ok, h, e, p, k, ins, count_asserts=False):
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
    # Les requêtes / Group:Filter incluent les détruits au combat encore sur le terrain.
    # Seul un ciblage les exclut ici (field::filter_matching_card, is_capable_be_effect_target).
    def filter_():
        row = (filter_mask(bank, s, bank.filt[s.cid[h], e, jnp.clip(b, 0, NF - 1)], p, h, bdes=False)
                                      & within & (((c & 8) == 0) | ((s.st & ST_BDES) == 0))
                                      & ~(((c & 8) != 0) & untargetable(bank, s, h)))
        row = jnp.where((c & (1 << 20)) != 0, within, row)  # copie intégrale du registre
        if count_asserts:
            count = row.sum()
            minimum, maximum = (c >> 4) & 255, (c >> 12) & 255
            return row, jnp.bool_(True), (count >= minimum) & ((maximum == 0) | (count < maximum))
        return put(row)
    branches[OP_FILTER] = filter_
    branches[OP_SELF] = lambda: put((jnp.arange(n) == h) & ((b == 0) | (((s.lk_flags[k] & CF_SRC_UNRELATED) == 0) & (s.mv[h] <= s.lk_t[k]))))
    def req():
        related = (c & 2) != 0
        count = (reg[ra] & (~related | (s.mv <= s.lk_tgt_t[jnp.clip(k, 0, CH - 1)]))).sum()
        return cond(jnp.where((c & 1) == 0, count >= b, count <= b))
    branches[OP_REQ] = req
    branches[OP_REQ_FREE] = lambda: cond(free_main(s, rel(p, a), jnp.where(b == Z_SZONE, Z_SZONE, Z_MZONE)) >= c)

    def minatk():
        original = (c & 2) != 0
        candidates = jnp.where(original, s.lk_reg[jnp.clip(k, 0, CH - 1), rb], reg[rb])
        value = jnp.where((c & 1) != 0, s.uid, s.atk_c)
        rank = jnp.where(candidates, value, BIG)
        return put(candidates & (rank == rank.min()))
    branches[OP_MINATK] = minatk
    branches[OP_CHOOSE] = lambda: (reg[rb], jnp.bool_(True), reg[rb].sum() >= (c & 15))
    lkind = bank.ekind[lc, s.lk_eff[lk]]
    lfield = (s.zone[jnp.maximum(s.lk_src[lk], 0)] & (Z_MZONE | Z_SZONE)) != 0
    lct = jnp.where((bank.ctype[lc] & TYPE_PENDULUM) != 0, P_SPELL, bank.ctype[lc])  # activation d'un pendule : magie
    branches[OP_COND_CHAIN] = lambda: cond((k > 0) & jnp.where(
        b == 0, ((s.lk_flags[lk] & CF_ACT) != 0) & ((lct & a) != 0),
        lfield & ((s.lk_flags[lk] & CF_ACT) == 0)))
    branches[OP_CHAINCARD] = lambda: put((jnp.arange(n) == s.lk_src[lk]) & (k > 0)
                                         & ((s.zone & jnp.where(b == 0, Z_SZONE, b)) != 0))
    branches[OP_PAY_LP] = lambda: cond(s.lp[p] >= a)
    evm = ((a >> jnp.arange(NEV)) & 1) != 0
    evb = ((b >> jnp.arange(NEV)) & 1) != 0
    branches[OP_EVENT] = lambda: put((jnp.where(c == 1, s.teg, s.ev_g) & evb[:, None]).any(axis=0))
    branches[OP_COND_TURN] = lambda: cond(s.tp == rel(p, a))
    branches[OP_COND_EVP] = lambda: cond((evm & s.ev_g.any(axis=1) & (s.ev_p == rel(p, b))).any())
    branches[OP_COND_SELF] = lambda: cond(((a == 0) | ((s.zone[h] & a) != 0)) & ((b == 0) | ((s.pzone[h] & b) != 0))
                                          & ((c == 0) | jnp.where(c < 0,
                                                jnp.where((-c & (1 << 30)) != 0, reason_matches(s.reason[h], -c & 0x1fffffff),
                                                          summon_matches(s.suminfo[h], c)), (s.st[h] & c) != 0)))
    branches[OP_COND_LABEL] = lambda: cond(s.ch_lab != 0)
    branches[OP_COND_PHASE] = lambda: cond((s.phase & a) != 0)
    branches[OP_COND_ATK] = lambda: cond(jnp.where(a == 0, s.atk, s.atk_tgt) == h)  # la carte attaque / est attaquée
    branches[OP_ATTACKER] = lambda: put((jnp.arange(n) == s.atk) & (s.atk >= 0) & ((s.st & ST_BDES) == 0)
                                        & ~((b != 0) & untargetable(bank, s, h)))  # b : ciblage
    row, wr, c_ok = _switch(op, [lambda _, f=f: f() for f in branches], None)
    return jnp.where(wr, reg.at[ra].set(row), reg), ok & c_ok


def test_ok(bank, s, h, e, p, enabled=None):
    """Mode test (chk==0 des scripts) : coût + cible exécutables par p avec la carte h, effet e ?"""
    n = s.cid.shape[0]
    progs = bank.tprog[s.cid[h], e]

    initial = (jnp.zeros((R, n), bool), jnp.bool_(True))
    length = (progs[:, 0] != OP_END).sum()
    if enabled is not None:
        length = jnp.where(enabled, length, 0)
    def body(i, carry):
        reg, ok = carry
        return _pure(bank, s, reg, ok, h, e, p, s.n_chain, progs[i], count_asserts=True)
    _, ok = lax.fori_loop(0, length, body, initial)
    return ok


def src_retarget(s, target, sel):
    """Duel.SetTargetCard recrée la relation de chaque cible avec l'effet : une source envoyée en coût puis prise pour
    cible (Ignition « envoyez cette carte au cimetière ; ciblez 1 monstre de votre cimetière ») reste liée à l'effet
    (IsRelateToEffect). On recale son horloge de début d'activation (CF_SRC_UNRELATED)."""
    src = jnp.maximum(s.ch_src, 0)
    hit = target & (s.ch_src >= 0) & sel[src]
    return jnp.where(hit, s.mv[src], s.ch_src_mv)


def sel_finish(bank, s):
    """Fin d'une sélection pendant un effet (D_SELECT) : les cartes choisies vont dans le registre de l'instruction."""
    sel = (s.st & ST_SEL) != 0
    ins = current_ins(bank, s)
    dst = jnp.clip(ins[1], 0, R - 1)
    target = (s.pg == PG_TARGET) & (ins[0] == OP_CHOOSE) & ((ins[3] & (1 << 10)) != 0)
    return s._replace(ch_tgt_t=jnp.where(target, s.clk, s.ch_tgt_t), ch_src_mv=src_retarget(s, target, sel),
                      reg=s.reg.at[dst].set(sel), st=s.st & ~ST_SEL, picks=_i(0), pc=s.pc + 1,
                      dec=_i(D_RUN), fsel=jnp.bool_(False), rsel=_i(0))


def sel_var(s):
    """Sélection à taille variable (CHOOSE, bit 9) en cours : A_PASS la termine dès que smin cartes sont prises."""
    return (s.dec == D_SELECT) & ~s.fsel & (s.rsel == 0) & (s.smin < s.picks + s.need)


def current_ins(bank, s):
    return bank.prog[s.cid[s.ch_src], s.ch_eff, s.pg, s.pc]


def use_count(bank, s, c, e, p, enabled=True):
    """Consomme la limite d'une activation ou d'une procédure, sans construire de maillon."""
    opt = bank.eopt[s.cid[c], e]
    oath = bank.erefund[s.cid[c], e] if bank.erefund.shape[1] else 0
    key = opt_key(s.cid[c], opt, oath)
    def add(_, table):
        return table.at[p].set(name_add(table[p], key))
    counts = lax.fori_loop(0, jnp.where(enabled & (key >= 0), 1, 0), add, s.opt_cid)
    bits = jnp.where(opt == OPT_CARD, ST_CNT << e, jnp.where(opt == OPT_CARD_NR, ST_CNTN << e, 0))
    return s._replace(opt_cid=counts, st=s.st.at[c].set(s.st[c] | jnp.where(enabled, bits, 0)))


def proc_ready(s, card):
    """Coûts de procédure payés : placement, sans maillon ni registre de groupe persistant."""
    return s._replace(pend=card, pend_set=jnp.bool_(False), picks=_i(0), psel=_i(0), need=_i(0),
                      zctx=_i(ZC_PROC), dec=_i(D_ZONE), ch_src=_i(-1), pc=_i(0), pg=_i(0),
                      ch_lab=_i(0), reg=jnp.zeros_like(s.reg))


def begin_act(bank, s, c, e, enabled=True):
    """Activation de l'effet e de c par le joueur qui décide : coût puis cible, puis mise en chaîne."""
    n = s.cid.shape[0]
    leave = (bank.ekind[s.cid[c], e] == K_ACTIVATE) & ((bank.ctype[s.cid[c]] & (TYPE_CONTINUOUS | TYPE_PENDULUM)) == 0)
    p = s.dec_player
    s = use_count(bank, s, c, e, p, enabled=enabled)
    # AddChain : une carte du terrain sous une annulation temporaire (Forbidden Chalice, Effect Veiler :
    # EFFECT_DISABLE_EFFECT) voit son maillon annulé, même si elle quitte ensuite le terrain (avant le coût)
    tdis = (s.tm_m[:, c] & (s.mv[c] <= s.tm_t) & (s.tm_code == MC_DISABLE)).any() & (s.zone[c] == Z_MZONE)
    card_activation = bank.ekind[s.cid[c], e] == K_ACTIVATE
    continuous = (s.zone[c] == Z_SZONE) & ((bank.ctype[s.cid[c]] & (TYPE_CONTINUOUS | TYPE_PENDULUM)) != 0)
    flags = jnp.where(card_activation, CF_ACT, 0) | jnp.where(continuous, CF_CONT, 0)
    s = s._replace(st=s.st.at[c].set(jnp.where(leave, s.st[c] | ST_LEAVE, s.st[c])), ch_dis=tdis, ch_lab=_i(0), ch_flags=flags,
                      ch_src=_i(c), ch_src_mv=s.mv[c], ch_tgt_t=s.clk, ch_eff=_i(e), ch_p=s.dec_player, ch_k=s.n_chain, pg=_i(PG_COST), pc=_i(0),
                      reg=jnp.zeros((R, n), bool), dec=_i(D_RUN), pend=_i(-1))
    if bank.einline.shape[1]:
        alternate = bank.einline[s.cid[c], e] - 1
        candidate = enabled & card_activation & (alternate >= 0) & (s.dstep == DS_NONE)
        available = act_ok(bank, s, c, jnp.maximum(alternate, 0), p, enabled=candidate)
        s = s._replace(dec=jnp.where(candidate & available, D_YESNO, s.dec),
                       zctx=jnp.where(candidate & available, ZC_INLINE, ZC_ACT))
    return s


def push_link(bank, s):
    s = brk(adjust(s))  # ygopro : break_effect après coût et cible (AddChain étape 8)
    k, src = s.n_chain, s.ch_src
    s = s._replace(lk_src=s.lk_src.at[k].set(s.ch_src), lk_eff=s.lk_eff.at[k].set(s.ch_eff),
                   lk_p=s.lk_p.at[k].set(s.ch_p), lk_reg=s.lk_reg.at[k].set(s.reg),
                   lk_neg=s.lk_neg.at[k].set(s.ch_dis), lk_t=s.lk_t.at[k].set(s.clk), lk_tgt_t=s.lk_tgt_t.at[k].set(s.ch_tgt_t), lk_lab=s.lk_lab.at[k].set(s.ch_lab),
                   lk_flags=s.lk_flags.at[k].set(s.ch_flags | jnp.where(s.mv[src] != s.ch_src_mv, CF_SRC_UNRELATED, 0)),
                   n_chain=k + 1, ch_src=_i(-1))
    return _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s._replace(tg=s.tg.at[src].set(False))),
                    lambda s: open_quick(s, 1 - s.ch_p), s)


def load_link(bank, s, k):
    """Résolution du maillon k ; un maillon nié saute directement à la fin de son programme, comme celui d'un
    monstre dont les effets sont annulés et qui est encore lié au maillon (solve_chain : STATUS_DISABLED)."""
    c = jnp.maximum(s.lk_src[k], 0)
    if bank.erefund.shape[1]:
        e, p = s.lk_eff[k], s.lk_p[k]
        key = opt_key(s.cid[c], bank.eopt[s.cid[c], e], bank.erefund[s.cid[c], e])
        refund = (bank.erefund[s.cid[c], e] != 0) & ((s.lk_flags[k] & CF_ACT_NEG) != 0) & (key >= 0)
        def restore(_, table):
            return table.at[p].set(jnp.where(table[p] == key + 1, 0, table[p]))
        s = s._replace(opt_cid=lax.fori_loop(0, refund.astype(I32), restore, s.opt_cid))
    dis = ((s.fl_c[c] & FL_DISABLED) != 0) & (s.zone[c] == Z_MZONE) & (s.mv[c] <= s.lk_t[k])
    continuous = (s.lk_flags[k] & CF_CONT) != 0
    dis |= continuous & ((s.zone[c] != Z_SZONE) | ((s.pos[c] & P_FACEUP) == 0) | (s.mv[c] > s.lk_t[k]))
    s = brk(s)  # break_effect avant chaque maillon (solve_chain)
    return s._replace(ch_k=k, ch_src=s.lk_src[k], ch_eff=s.lk_eff[k], ch_p=s.lk_p[k], reg=s.lk_reg[k] & (s.mv <= s.lk_t[k]),
                      ch_lab=s.lk_lab[k], ch_flags=s.lk_flags[k],
                      pg=_i(PG_RESOLVE), pc=jnp.where(s.lk_neg[k] | dis, P - 1, 0).astype(I32), dec=_i(D_RUN),
                      ev_g=jnp.zeros_like(s.ev_g))


def chain_end(bank, s):
    """Chaîne résolue : les magies/pièges activés encore sur le terrain partent ensemble au cimetière, puis
    nouvelle fenêtre pour le joueur du tour."""
    n = s.cid.shape[0]
    s = group_to_grave(bank, s, (s.st & ST_LEAVE) != 0, REASON_RULE)
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
        on &= ((s.lk_flags[s.ch_k] & CF_CONT) == 0) | ((s.zone[h] == Z_SZONE) &
               ((s.pos[h] & P_FACEUP) != 0) & (s.mv[h] <= s.lk_t[s.ch_k]))
        s = enable(s._replace(st=s.st.at[h].set(jnp.where(on, s.st[h] | ST_ON, s.st[h]))),
                   on & (jnp.arange(s.cid.shape[0]) == h))
        return _cond(s.ch_k > 0, lambda s: load_link(bank, s, s.ch_k - 1), lambda s: chain_end(bank, s), s)

    def end(s):
        return _switch(s.pg, [lambda s: s._replace(pg=_i(PG_TARGET), pc=_i(0)),
                                 lambda s: _cond(bank.ekind[s.cid[h], e] == K_PROC,
                                                 lambda s: proc_ready(s, h), lambda s: push_link(bank, s), s),
                                 next_link], s)

    def pure(s):
        reg, ok = _pure(bank, s, s.reg, jnp.bool_(True), h, e, p, s.ch_k, ins)
        return _cond(ok, adv, end, s._replace(reg=reg))  # condition fausse à la résolution : l'effet s'arrête

    def choose(s):  # au plus (c >> 4) & 15 cartes, moins s'il n'y a pas assez de candidats (rien : pas de décision)
        need = jnp.minimum((c >> 4) & 15, s.reg[jnp.clip(b, 0, R - 1)].sum()).astype(I32)
        # taille variable (bit 9) : au moins c & 15 cartes (autant que possible), A_PASS termine ensuite la sélection
        smin = jnp.where((c >> 9) & 1, jnp.minimum(c & 15, need), need).astype(I32)
        return _cond(need > 0,
                        lambda s: s._replace(dec=_i(D_SELECT), dec_player=rel(p, (c >> 8) & 1), need=need, picks=_i(0),
                                             smin=smin),
                        lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def move_(s):  # vers le cimetière (détruire, libérer, défausser, envoyer), la main du propriétaire ou les bannies
        # c = 1 : destruction (Duel.Destroy) : les cartes indestructibles par les effets restent ; le registre ne garde
        # que les cartes détruites (Destroy(...) ~= 0)
        destroying = (c & 1) != 0
        supplied = (c & MOVE_REASON_EXPLICIT) != 0
        raw_reason = (c >> 1) & 0x1fffffff
        base = jnp.where(supplied, raw_reason, jnp.where(s.pg == PG_COST, REASON_COST, REASON_EFFECT))
        reason = base | jnp.where(destroying, REASON_DESTROY, 0)
        reason |= jnp.where((c < 0) & (s.zone[h] == Z_MZONE), REASON_RELEASE, 0)
        t = target(s) & ~(destroying & ((s.fl_c & FL_INDES_EFF) != 0) & (s.zone == Z_MZONE))
        s = s._replace(reg=jnp.where(destroying, s.reg.at[ra].set(t), s.reg))
        return adv(_switch(jnp.where(b == Z_HAND, 1, jnp.where(b == Z_REMOVED, 2, 0)),
                              [lambda s: group_to_grave(bank, s, t, reason),
                               lambda s: group_to_hand(bank, s, t, reason),
                               lambda s: group_send(s, t, Z_REMOVED, P_GRAVE, reason=reason)], s))

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
        active = (op == OP_SPSUMMON) & s.reg[ra].any()
        typ = effect_summon_type(c)
        attempted = s.reg[ra] & active
        s = s._replace(suminfo=jnp.where(attempted, summon_info(typ, s.zone, p), s.suminfo))
        def prepare(_, carry):
            group, failed_space = carry
            lk = (bank.ctype[s.cid] & TYPE_LINK) != 0
            group &= special_allowed(bank, s, proper=c > 0) & ~(lk & ((b & P_FU_ATK) == 0))
            # Une réponse peut avoir rempli les cases ou interdit l'invocation après le ciblage.
            space = jnp.where(s.zone == Z_EXTRA, ex_zones(s, p).any(), free_main(s, p) > 0)
            faceup_pend = ((s.zone == Z_EXTRA) & ((bank.ctype[s.cid] & TYPE_PENDULUM) != 0)
                           & ((s.pos & P_FACEUP) != 0))
            pend_space = lax.fori_loop(0, (group & faceup_pend).any().astype(I32),
                lambda _, previous: pend_extra_zones(bank, s, p).any(), jnp.bool_(True))
            space = jnp.where(faceup_pend, pend_space, space)
            selected = jnp.argmax(group)
            need_link = group.any() & (s.zone[selected] == Z_EXTRA) & lk[selected]
            link_space = lax.fori_loop(0, need_link.astype(I32),
                lambda _, previous: link_zones(bank, s, p, selected).any(), jnp.bool_(True))
            available = space & (~lk | link_space)
            failed_space = group & ~available & (s.zone != Z_GRAVE)
            return group & available, failed_space
        group, failed_space = lax.fori_loop(0, active.astype(I32), prepare,
            (s.reg[ra], jnp.zeros_like(s.reg[ra])))
        # SpSummonStep / ss_tograve_set : un manque de place envoie la carte au
        # cimetière par règle ; une interdiction d'invoquer ne le fait pas.
        s = lax.fori_loop(0, failed_space.any().astype(I32),
            lambda _, state: group_to_grave(bank, state, failed_space, REASON_RULE), s)
        s = s._replace(reg=s.reg.at[ra].set(group))
        return _cond(group.any(), lambda s: s._replace(dec=_i(D_ZONE), zctx=_i(ZC_SPSUM), dec_player=p), adv, s)

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
                              tm_dur=s.tm_dur.at[j].set((b >> 9) & 15), tm_src=s.tm_src.at[j].set(h),
                              tm_n=jnp.minimum(s.tm_n + 1, TM), ovf=s.ovf | jnp.where(s.tm_n >= TM, 2, 0).astype(I32)))

    def damage_(s):
        mul = jnp.where(c > 0, s.reg[jnp.clip(c - 1, 0, R - 1)].sum(), 1)
        return adv(damage(s, rel(p, a), b * mul))

    branches = [lambda s: s] * N_OPS
    branches[OP_END] = end
    for o in (OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_COND_CHAIN, OP_CHAINCARD, OP_EVENT, OP_COND_TURN,
              OP_COND_EVP, OP_COND_SELF, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE, OP_COND_LABEL):
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
    def negate(s):  # negate_chain : la carte d'une activation encore liée en zone magie/piège part au cimetière
        k1 = jnp.clip(s.ch_k - 1, 0, CH - 1)
        lc = jnp.maximum(s.lk_src[k1], 0)
        hit = (((s.lk_flags[k1] & CF_ACT) != 0) & (s.zone[lc] == Z_SZONE) & (s.mv[lc] <= s.lk_t[k1])
               & (s.lk_src[k1] >= 0))
        return adv(s._replace(lk_neg=s.lk_neg.at[k1].set(True),
                              lk_flags=s.lk_flags.at[k1].set(s.lk_flags[k1] | CF_ACT_NEG),
                              st=s.st.at[lc].set(jnp.where(hit, s.st[lc] | ST_LEAVE | ST_ACTDIS, s.st[lc]))))
    branches[OP_NEGATE] = negate
    # coût en LP d'une activation : défaite constatée par adjust à la mise en chaîne (push_link), après les cibles ;
    # procédure (K_PROC, sans maillon) : tout de suite
    branches[OP_PAY_LP] = lambda s: adv(_cond(bank.ekind[s.cid[h], e] == K_PROC, lambda s: damage(s, p, a),
                                              lambda s: s._replace(lp=s.lp.at[p].set(jnp.maximum(0, s.lp[p] - a))), s))
    branches[OP_LABEL] = lambda s: adv(s._replace(
        ch_lab=jnp.where(b == 0, s.reg[ra].any().astype(I32), s.ch_lab),
        ch_tgt_t=jnp.where(b == 1, s.clk, s.ch_tgt_t), ch_src_mv=src_retarget(s, b == 1, s.reg[ra])))
    branches[OP_YESNO] = lambda s: s._replace(dec=_i(D_YESNO), dec_player=p)
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
    us = zone_used(s, p, Z_SZONE)
    pend_hand = in_hand & ((ct & TYPE_PENDULUM) != 0) & (~us[0] | ~us[4])  # vers une zone pendule libre
    faceup_mon = my_mon & ((s.pos & P_FACEUP) != 0)
    in_grave = mine & (s.zone == Z_GRAVE)
    faceup_st = mine & (s.zone == Z_SZONE) & ((s.pos & P_FACEUP) != 0) & ((s.st & ST_ON) != 0)
    trig_w = chn & (s.wk == W_TRIG)
    # main ajustée en End Phase (hand_adjusted) : pas de fenêtre libre, seulement les réponses à une chaîne
    fast = bat | (chn & ~trig_w & ~((s.wk == W_QUICK) & (s.ret == R_HAND) & (s.n_chain == 0)))  # vitesse 2 minimum
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
        # zones d'un effet rapide de monstre / d'ignition (erng ; 0 pour une ignition : zone monstres)
        rng = jnp.where((kind == K_IGNITION) & (rng == 0), Z_MZONE, rng)
        inr = ((in_hand & ((rng & Z_HAND) != 0)) | (faceup_mon & ((rng & Z_MZONE) != 0))
               | (in_grave & ((rng & Z_GRAVE) != 0)) | (faceup_st & ((rng & Z_SZONE) != 0)))
        qm = (kind == K_QUICK) & inr
        where = ((idle & spell & ~quick & act & (from_hand | set_st))
                 | (idle & act & pend_hand)
                 | ((idle | fast) & qm)
                 | ((idle | fast) & quick & act & ((from_hand & (p == s.tp)) | ready))
                 | ((idle | fast) & trap & act & ready & ((s.pf_c[p] & PF_NO_TRAP) == 0))
                 | (idle & inr & (kind == K_IGNITION)))
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
        # Procédure inhérente depuis la main : ni vitesse de chaîne, ni activation de monstre.
        where |= idle & in_hand & (kind == K_PROC) & (free_main(s, p) > 0) & ((s.fl_c & FL_NO_SS) == 0)
        cand.append(where)
    return jnp.stack(cand)


def auto_pass(bank, s):
    """Fenêtre de réponse où le joueur ne peut rien activer : passer est la seule action (jouée par settle, sans
    passer par legal + apply). Les fenêtres W_TRIG ne sont ouvertes que s'il y a un déclencheur activable.
    Avec au plus KA candidats, leur coût et leur cible sont testés (act_ok, comme legal) : en milieu de partie, la
    plupart des fenêtres ont un ou deux candidats qui ne passent pas act_ok, et la passe, seule action légale, prenait
    sinon un tour complet de step (apply, settle, legal, forced). Même partie : A_PASS joue aussi pass_priority."""
    cand = act_cand(bank, s)
    nc = cand.sum()
    ce, ch = jnp.nonzero(cand, size=KA, fill_value=0)
    valid = (jnp.arange(KA) < nc) & (nc <= KA) & (s.dec == D_CHAIN) & (s.wk != W_TRIG) & ~s.done
    okc = jax.vmap(lambda h, e, active: act_ok(bank, s, h, e, s.dec_player, enabled=active))(ch, ce, valid) & valid
    return (s.dec == D_CHAIN) & (s.wk != W_TRIG) & ~s.done & ((nc == 0) | ((nc <= KA) & ~okc.any()))


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
              & ((ct & (TYPE_RITUAL | TYPE_EXTRA | TYPE_SPSUMMON)) == 0))
    can_repos = (my_mon & ((s.st & (ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN)) == 0)
                 & ((ct & TYPE_LINK) == 0))  # un Link ne change pas de position
    unsel = (s.st & ST_SEL) == 0
    # get_attack_target (select_target vrai par défaut) : au moins une cible possible, ou attaque directe
    atk_ok = ~op_mon.any() | (op_mon & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    # sans attaque directe (FL_NO_DIRECT) : il faut un monstre adverse qui peut être attaqué
    tgt_ok = (op_mon & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    can_atk = (my_mon & (s.pos == P_FU_ATK) & ((s.st & ST_ATTACKED) == 0) & ((s.fl_c & FL_NO_ATTACK) == 0)
               & jnp.where((s.fl_c & FL_NO_DIRECT) != 0, tgt_ok, atk_ok))
    ins = current_ins(bank, s)
    eff_cand = s.reg[jnp.clip(ins[2], 0, R - 1)]
    if fus:
        src = jnp.argmax(eff_cand)
        eff_cand = jnp.where(s.fsel, fus_picks(bank, s, src),
                             jnp.where(s.rsel > 0, rit_picks(bank, s, src, s.rsel), eff_cand))
        syn_c = syn_picks(bank, s, jnp.maximum(s.pend, 0))
        xyz_c = xyz_mats(bank, s, jnp.maximum(s.pend, 0)) & unsel
        lnk_c, lgoal = link_picks(bank, s, jnp.maximum(s.pend, 0))
    else:
        syn_c = xyz_c = lnk_c = jnp.zeros(n, bool)
        lgoal = jnp.bool_(False)
    # invocation pendule : V_SUMMON sur l'échelle gauche ; sélection (PS_PEND) : boucle de Pendulum.Operation
    pl, can_pend, pch, pcx, ft1, ft2, ft, plz, _ = pend_info(bank, s, p)
    psel_n = (~unsel & pch).sum(), (~unsel & pcx).sum()
    pen_c = unsel & ((pch & (ft1 > psel_n[0])) | (pcx & (ft2 > psel_n[1]))) & (ft > psel_n[0] + psel_n[1])

    # libérations : sans case principale libre, l'une d'elles doit libérer une case principale (pas une zone extra)
    mainz = my_mon & (s.seq < N_MAIN_ZONES)
    trib_ok = ((free_m > 0) | mainz | (mainz & ~unsel).any()
               | ((s.need >= 2) & (mainz & unsel).any()))
    idle, bat, chn = s.dec == D_IDLE, s.dec == D_BATTLE, s.dec == D_CHAIN
    # invocation par procédure (synchro, Xyz, Link) : V_SUMMON sur le monstre de l'Extra Deck, sauf interdiction (FL_NO_SS)
    can_syn = (s.zone == Z_EXTRA) & (s.ctrl == p) & (s.sz_c | s.xz_c | s.lz_c) & ((s.fl_c & FL_NO_SS) == 0)
    verbs = [
        idle & (can_ns | can_syn | ((jnp.arange(n) == pl) & can_pend)),
        idle & can_ns,
        idle & can_repos,
        bat & can_atk,
        ((s.dec == D_TRIBUTE) & jnp.where(s.psel == PS_SYN, syn_c, jnp.where(s.psel == PS_XYZ, xyz_c, jnp.where(s.psel == PS_LINK, lnk_c, jnp.where(s.psel == PS_PEND, pen_c, my_mon & unsel & trib_ok))))) | ((s.dec == D_TARGET) & op_mon & ((s.fl_c & FL_NO_BTARGET) == 0))
        | ((s.dec == D_DISCARD) & in_hand & unsel) | ((s.dec == D_SELECT) & eff_cand & unsel),
        idle & in_hand & (spell | trap) & (free_s > 0),
    ]
    trig_w = chn & (s.wk == W_TRIG)
    cand = act_cand(bank, s)
    # Tous les candidats, en blocs de KC ; 16 blocs au maximum dans l'arène bornée.
    # Le noyau reste KC × N, et les parties ordinaires ne font qu'un bloc.
    def more(carry):
        remaining, _, rounds = carry
        return remaining.any() & (rounds < ACT_ROUNDS)
    def batch(carry):
        remaining, accepted, rounds = carry
        ce, ch = jnp.nonzero(remaining, size=KC, fill_value=0)
        valid = jnp.arange(KC) < remaining.sum()
        passed = jax.vmap(lambda h, e, active: act_ok(bank, s, h, e, p, enabled=active))(ch, ce, valid) & valid
        accepted |= jnp.zeros((E, n), I32).at[ce, ch].add(passed.astype(I32)) > 0
        return remaining.at[ce, ch].set(False), accepted, rounds + 1
    _, ok, _ = lax.while_loop(more, batch, (cand, jnp.zeros((E, n), bool), _i(0)))
    verbs = jnp.concatenate([jnp.stack(verbs), cand & ok])
    sp = jnp.zeros(N_SPECIAL, bool)
    sp = sp.at[A_TO_BP].set(idle & (s.phase == PH_MAIN1) & (s.turn > 1))
    sp = sp.at[A_TO_M2].set(bat)
    sp = sp.at[A_TO_EP].set(idle | bat)
    mz = (s.zctx == ZC_SUMMON) | (s.zctx == ZC_SPSUM) | (s.zctx == ZC_PROC) | (s.zctx == ZC_PEND)
    used = jnp.where(mz, zone_used(s, p, Z_MZONE), zone_used(s, p, Z_SZONE))[:7]
    spc = jnp.argmax(s.reg[jnp.clip(ins[1], 0, R - 1)])             # carte invoquée spécialement (ZC_SPSUM)
    fromex = ((s.zctx == ZC_SPSUM) & (s.zone[spc] == Z_EXTRA)) | ((s.zctx == ZC_PROC) & (s.zone[jnp.maximum(s.pend, 0)] == Z_EXTRA))
    pc = jnp.where((s.zctx == ZC_PROC) | (s.zctx == ZC_PEND), jnp.maximum(s.pend, 0), spc)  # carte invoquée
    is_lk = (ct[pc] & TYPE_LINK) != 0
    exz = lax.cond(is_lk & (s.dec == D_ZONE), lambda s: link_zones(bank, s, p, pc), lambda s: ex_zones(s, p), s)
    faceup_pend = ((ct[pc] & TYPE_PENDULUM) != 0) & ((s.pos[pc] & P_FACEUP) != 0) & ~is_lk
    exz = jnp.where(faceup_pend, jnp.concatenate([exz[:N_MAIN_ZONES] & plz, exz[N_MAIN_ZONES:]]), exz)
    zok = jnp.where(fromex, exz, ~used & (jnp.arange(7) < N_MAIN_ZONES))
    zok = jnp.where(s.zctx == ZC_PEND, pend_zones(bank, s, p, pc, plz), zok)
    # activation d'un pendule : zones pendule (cases 0 et 4)
    pact = (s.zctx == ZC_ACT) & ((ct[jnp.maximum(s.pend, 0)] & TYPE_PENDULUM) != 0)
    zok &= ~pact | (jnp.arange(7) == 0) | (jnp.arange(7) == 4)
    sp = sp.at[A_ZONE0:A_ZONE0 + 7].set((s.dec == D_ZONE) & zok)
    pos_ok = jnp.where(s.zctx == ZC_SPSUM, current_ins(bank, s)[2], P_FU_ATK | P_FU_DEF)  # positions permises
    pos_ok &= jnp.where(is_lk, P_FU_ATK, -1)  # un Link est toujours en position d'attaque
    sp = sp.at[A_POS_ATK].set((s.dec == D_POS) & ((pos_ok & P_FU_ATK) != 0))
    sp = sp.at[A_POS_DEF].set((s.dec == D_POS) & ((pos_ok & P_FU_DEF) != 0))
    sp = sp.at[A_PASS].set((chn & ~(trig_w & (s.tst < 2)))  # les déclencheurs obligatoires ne se refusent pas
                           | ((s.dec == D_TRIBUTE) & (s.psel == PS_LINK) & lgoal)  # Link : « terminer » la sélection
                           | ((s.dec == D_TRIBUTE) & (s.psel == PS_PEND) & ~unsel.all())  # pendule : au moins une carte
                           | (s.dec == D_YESNO)
                           | (sel_var(s) & (s.picks >= s.smin)))  # sélection à taille variable : « terminer »
    sp = sp.at[A_YES].set(s.dec == D_YESNO)
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
        info = summon_info(jnp.where(need > 0, SUMMON_TRIBUTE, SUMMON_NORMAL), _i(Z_HAND), s.tp)
        s = s._replace(suminfo=s.suminfo.at[c].set(jnp.where(s.zone[c] == Z_HAND, info, s.suminfo[c])))
        s = s._replace(pend=c, pend_set=jnp.bool_(is_set), need=need, picks=_i(0), zctx=_i(ZC_SUMMON))
        s = s._replace(dec=jnp.where(need > 0, D_TRIBUTE, D_ZONE).astype(I32))
        # depuis l'Extra Deck : invocation synchro (matériaux choisis un par un en D_TRIBUTE, puis case et position)
        ps = jnp.where(bank.syn[s.cid[c]] != 0, PS_SYN, jnp.where(bank.xct[s.cid[c]] > 0, PS_XYZ, PS_LINK)).astype(I32)
        # depuis l'échelle gauche : invocation pendule (cartes choisies une par une, puis placées une par une)
        return _switch(jnp.where(s.zone[c] == Z_EXTRA, 1, jnp.where(s.zone[c] == Z_SZONE, 2, 0)), [
            lambda s: s,
            lambda s: s._replace(psel=ps, need=_i(99), zctx=_i(ZC_PROC), dec=_i(D_TRIBUTE)),
            lambda s: s._replace(psel=_i(PS_PEND), need=_i(99), zctx=_i(ZC_PEND), dec=_i(D_TRIBUTE))], s)

    def repos(s):
        new = jnp.where(s.pos[c] == P_FU_ATK, P_FU_DEF, P_FU_ATK)
        fd = (jnp.arange(n) == c) & (s.pos[c] == P_FD_DEF)  # verso → recto : invocation par flip
        s = raise_ev(raise_ev(s, EV_FLIP_SUMMON, fd, s.tp), EV_FLIP, fd, s.tp)
        info = (s.suminfo[c] & 0x7fffffff) | (s.tp << 31)
        s = s._replace(suminfo=s.suminfo.at[c].set(jnp.where(fd[c], info, s.suminfo[c])))
        s = enable(s._replace(pos=s.pos.at[c].set(new),
                              st=s.st.at[c].set(s.st[c] | ST_FORM_CHANGED | jnp.where(fd[c], ST_FLIP_TURN, 0))), fd)
        return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

    def attack(s):
        s = s._replace(st=s.st.at[c].set(s.st[c] | ST_ATTACKED), atk=c, atk_tgt=_i(-1), ds_t=s.clk, rpl=jnp.bool_(False))
        opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
        has_target = opp.any()
        # aucune cible possible (CANNOT_BE_BATTLE_TARGET) : attack_announce_failed, l'attaque compte, pas de fenêtre
        ok = (opp & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
        return _cond(has_target, lambda s: _cond(ok, lambda s: s._replace(dec=_i(D_TARGET)), to_battle, s),
                            lambda s: trig_point(bank, declare(s, _i(-1), jnp.bool_(False))), s)

    def select(s):
        def pick(s):
            return s._replace(st=s.st.at[c].set(s.st[c] | ST_SEL), need=s.need - 1, picks=s.picks + 1)

        def tribute(s):
            s = pick(s)
            sel = (s.st & ST_SEL) != 0
            xdone = sel.sum() >= bank.xct[s.cid[jnp.maximum(s.pend, 0)]]
            # Link : la sélection se termine par A_PASS (joué d'office quand il ne reste aucun candidat)
            fin = jnp.where(s.psel == PS_SYN, syn_done(bank, s),
                            jnp.where(s.psel == PS_XYZ, xdone, (s.psel == 0) & (s.need == 0)))
            done = lambda s: _cond(s.psel == PS_XYZ,
                                   lambda s: attach(s, sel, s.pend)._replace(picks=_i(0)),  # matériaux Xyz attachés
                                   lambda s: selected_to_grave(bank, s, jnp.where(s.psel == PS_SYN,
                                       REASON_SYNCHRO | REASON_MATERIAL, REASON_SUMMON | REASON_MATERIAL | REASON_RELEASE)), s)._replace(dec=_i(D_ZONE), psel=_i(0))
            return _cond(fin, done, lambda s: s, s)

        def target(s):
            return trig_point(bank, declare(s, c, s.rpl))

        def discard(s):
            s = pick(s)
            return _cond(s.need == 0, lambda s: trig_point(bank, open_quick(selected_to_grave(bank, s, REASON_RULE | REASON_DISCARD | 0x100), s.tp, R_HAND,
                                                                            chained=False)), lambda s: s, s)

        def eff_select(s):
            done = lambda s: sel_finish(bank, s)
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
        active = verb == V_ACT0 + e
        flip = lambda s: begin_act(bank, s._replace(pos=s.pos.at[c].set(P_SFU)), c, e, enabled=active)
        card = bank.ekind[s.cid[c], e] == K_ACTIVATE  # activation de carte (pas un effet de monstre)
        return _switch(jnp.where(card & (s.zone[c] == Z_HAND), 0, jnp.where(card & (s.zone[c] == Z_SZONE), 1, 2)),
                          [from_hand, flip, lambda s: _cond((bank.ekind[s.cid[c], e] == K_PROC) &
                              (bank.prog[s.cid[c], e, PG_TARGET, 0, 0] == OP_END),
                              lambda s: proc_ready(s, c), lambda s: s, begin_act(bank, s, c, e, enabled=active))], s)

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
            lambda s: begin_act(bank, to_szone(s, P_SFU), s.pend, s.ch_eff,
                                 enabled=(verb == NV) & (s.dec == D_ZONE) & (s.zctx == ZC_ACT)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
        ], s)

    def proc_place(s):  # fin d'une invocation par procédure (synchro) : pas de chaîne, fenêtre comme une invocation
        c = s.pend
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        ct = bank.ctype[s.cid[c]]
        typ = jnp.where((ct & TYPE_LINK) != 0, SUMMON_LINK,
                        jnp.where((ct & TYPE_XYZ) != 0, SUMMON_XYZ,
                                  jnp.where((ct & TYPE_SYNCHRO) != 0, SUMMON_SYNCHRO, SUMMON_SPECIAL)))
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(typ,s.zone[c],s.tp)))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.tp, pos)
        s = raise_ev(brk(s), EV_SPSUMMON, jnp.arange(n) == c, s.tp)  # break_effect après les matériaux
        lk = (bank.ctype[s.cid[c]] & TYPE_LINK) != 0
        s = s._replace(proc=s.proc.at[c].set(True), st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM | jnp.where(lk, ST_LINKSUM, 0)),
                       pend=_i(-1))
        return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

    def pend_next(s):
        """Invocation pendule : prochaine carte choisie à placer (plus petit uid : card_sort), ou fin : EVENT_SPSUMMON
        du groupe, puis fenêtre comme une invocation."""
        sel = (s.st & ST_SEL) != 0

        def go(s):
            return s._replace(pend=jnp.argmin(jnp.where(sel, s.uid, BIG)).astype(I32), zctx=_i(ZC_PEND), dec=_i(D_ZONE),
                              dec_player=s.tp, psel=_i(0))

        def fin(s):
            g = (s.st & ST_PSTEP) != 0
            s = raise_ev(brk(s), EV_SPSUMMON, g, s.tp)
            s = s._replace(st=s.st & ~ST_PSTEP, psel=_i(0), pend=_i(-1), ps_used=jnp.bool_(True))
            return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))
        return _cond(sel.any(), go, fin, s)

    def pend_place(s):
        c = s.pend
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(SUMMON_PENDULUM,s.zone[c],s.tp)))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.tp, pos)
        s = s._replace(proc=s.proc.at[c].set(True), st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM | ST_PSTEP))
        return pend_next(s)

    def position(s):
        return _switch(jnp.where(s.zctx == ZC_PROC, 1, jnp.where(s.zctx == ZC_PEND, 2, 0)),
                       [eff_position, proc_place, pend_place], s)

    def eff_position(s):
        c = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[1], 0, R - 1)])
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        # Seule une invocation de procédure explicite est correctement accomplie.
        proper = current_ins(bank, s)[3]
        completed = proper > 0
        typ = effect_summon_type(proper)
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(typ,s.zone[c],s.ch_p)))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.ch_p, pos)
        # Duel.SpecialSummon : EVENT_SPSUMMON_SUCCESS avec ep = PLAYER_NONE (operations.cpp, special_summon, étape 4) ;
        # une procédure (proc_place, pend_next) donne ep = le joueur qui invoque
        s = raise_ev(s, EV_SPSUMMON, jnp.arange(n) == c, _i(2))
        s = s._replace(proc=s.proc.at[c].set(s.proc[c] | completed))
        return s._replace(st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM), pc=s.pc + 1, dec=_i(D_RUN))

    def to_ep(s):
        return open_quick(s, 1 - s.tp, jnp.where(s.dec == D_BATTLE, R_TO_EPB, R_TO_EP), passed=True, chained=False)

    def spec(s):
        yn = s.dec == D_YESNO  # oui : l'effet continue ; non (A_PASS) : il s'arrête (dernière instruction, OP_END)
        rq = s.zctx == ZC_REPLAY  # relance d'attaque : oui → nouvelle cible, non → fin de l'attaque
        inline = s.zctx == ZC_INLINE
        def inline_yes(s):
            if not bank.einline.shape[1]:
                return s
            e = bank.einline[s.cid[s.ch_src], s.ch_eff] - 1
            s = use_count(bank, s, s.ch_src, e, s.ch_p, enabled=yn & inline & (k == A_YES))
            return s._replace(ch_eff=e, pc=_i(0), pg=_i(PG_COST), dec=_i(D_RUN), zctx=_i(ZC_ACT))
        return _switch(jnp.where(yn, jnp.where(inline, jnp.where(k == A_YES, 10, 11),
                                 jnp.where(rq, jnp.where(k == A_YES, 8, 9), jnp.where(k == A_YES, 6, 7))),
                                 jnp.where(k < A_ZONE0, k, jnp.where(k < A_POS_ATK, 3, jnp.where(k < A_PASS, 4, 5)))), [
            lambda s: open_quick(s, 1 - s.tp, R_TO_BP, passed=True, chained=False),
            lambda s: open_quick(s, 1 - s.tp, R_TO_M2B, passed=True, chained=False),
            to_ep, zone, position,
            lambda s: _cond(s.dec == D_TRIBUTE,  # fin de la sélection des matériaux Link : au cimetière, puis la case
                            lambda s: _cond(s.psel == PS_PEND, pend_next,  # pendule : placement carte par carte
                                            lambda s: selected_to_grave(bank, s, REASON_LINK | REASON_MATERIAL)._replace(dec=_i(D_ZONE), psel=_i(0)), s),
                            lambda s: _cond(s.dec == D_SELECT, lambda s: sel_finish(bank, s), lambda s: _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s._replace(tst=s.tst + 1)),
                                            lambda s: pass_priority(bank, s), s), s), s),
            lambda s: s._replace(pc=s.pc + 1, dec=_i(D_RUN)),
            lambda s: s._replace(pc=_i(P - 1), dec=_i(D_RUN)),
            replay_go, battle_over,
            inline_yes, lambda s: s._replace(pg=_i(PG_TARGET), pc=_i(P - 1), dec=_i(D_RUN), zctx=_i(ZC_ACT)),
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
    sel = ((s.dec == D_TRIBUTE) & (s.psel == 0)) | (s.dec == D_DISCARD) | ((s.dec == D_SELECT) & ~s.fsel & (s.rsel == 0)
                                                                            & ~sel_var(s))
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
        s = refresh_ex(bank, settle(bank, refresh_if(bank, s, apply(bank, s, a))))
        return s, forced(bank, s), i + 1

    # pas de settle initial : s est toujours à une décision (sortie de step ou de new_game), et une seconde copie de
    # settle doublerait presque la taille du programme compilé (≈ 40 % du HLO de step)
    s, _, _ = lax.while_loop(lambda c: (c[1] >= 0) & ~c[0].done & (c[2] < MAX_ITER), body, (s, _i(a), _i(0)))
    return s


def _where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


def step_batch(bank, s, a, frac=0.0, stats=False):
    """Version lot de step, pour l'entraînement : a[b] = action de la partie b (politique, ou l'action forcée rendue
    par l'appel précédent). Rend (états, masques légaux, actions forcées en attente ou -1).

    Les boucles et les calculs rares (trig_eval) sont décidés au niveau du lot. La boucle des actions forcées s'arrête
    dès que la part des parties qui en ont encore une est <= frac : sinon tout le lot attend la plus lente. Les parties
    arrêtées en route reprennent à l'appel suivant avec leur action forcée. De même, les boucles d'exécution et de
    settle s'arrêtent dès qu'il ne reste en cours que la part frac des parties qui jouent (sinon tout le lot attend
    la traîne des longues résolutions, qui s'allonge avec B) : ces parties sont rendues en cours de résolution, avec
    l'action forcée n_actions(N) (« continuer », sans apply). frac = 0 : même résultat que vmap(step).
    stats : rend aussi un compteur int32[11] (diagnostic du débit) : tours de la boucle externe, de settle_b, de run_b,
    trig_eval pris, passes prises, secondes passes prises, legal avec fusion ; puis la somme sur les tours du nombre de
    parties actives : run_b, trig_eval, première passe, seconde passe.

    Après le premier tour (lot complet), les tours des actions forcées ne portent que sur un sous-lot de B/4 parties
    (celles qui ont une action forcée, les autres attendent le tour suivant). Dans settle, quand le premier joueur passe
    sans pouvoir rien faire, la seconde passe (adversaire qui ne peut rien faire non plus) est jouée dans le même tour."""
    nb = s.done.shape[0]
    na = n_actions(s.cid.shape[1])

    def apb(s, mask, prev):
        """auto_pass recalculé seulement pour les parties mask (sous-lot de B/8, ou tout le lot s'il y en a plus),
        prev ailleurs : auto_pass teste act_ok, trop cher pour tout le lot à chaque tour de settle."""
        n = s.done.shape[0]
        ks = max(1, n // 8)
        full = lambda s: jax.vmap(lambda s: auto_pass(bank, s))(s)

        def sub(s):
            idx = jnp.nonzero(mask, size=ks, fill_value=n)[0]
            part = full(jax.tree.map(lambda x: x[jnp.minimum(idx, n - 1)], s))
            return jnp.zeros(n, bool).at[idx].set(part, mode="drop")
        r = lax.cond(mask.sum() > ks, full, lambda s: lax.cond(mask.any(), sub, lambda s: jnp.zeros(n, bool), s), s)
        return jnp.where(mask, r, prev)

    def todo(s, ap):  # ap : auto_pass(s) à jour pour toutes les parties
        return ((s.dec == D_RUN) | (s.dec == D_TRIG) | ap) & ~s.done

    def run_b(s, thr):  # boucle d'exécution au niveau du lot : refresh seulement pour les parties modifiées
        def body(c):
            s, i, n = c
            r = (s.dec == D_RUN) & ~s.done
            return refresh_b(bank, s, _where(r, jax.vmap(lambda s: exec_one(bank, s))(s), s)), i + 1, n + r.sum()
        c = lax.while_loop(lambda c: (((c[0].dec == D_RUN) & ~c[0].done).sum() > thr) & (c[1] < MAX_ITER), body,
                           (s, _i(0), _i(0)))
        return c[0], c[1], c[2]

    def settle_body(c):  # invariant : ap == auto_pass(s) pour toutes les parties, à l'entrée et à la sortie
        s, i, thr, k, ap = c
        r0 = ((s.dec == D_RUN) | (s.dec == D_TRIG)) & ~s.done   # seules parties que run_b et trig_eval modifient
        s, nr, ar = run_b(s, thr)
        need = s.dec == D_TRIG
        k = k + jnp.array([0, 1, nr, need.any(), 0, 0, 0, ar, need.sum(), 0, 0], I32)
        s = lax.cond(need.any(), lambda s: _where(need, jax.vmap(lambda s: trig_eval(bank, s))(s), s),
                     lambda s: s, s)
        ap = apb(s, r0 | need, ap)

        def passes(s):  # première passe, puis la seconde tout de suite si l'adversaire ne peut rien faire non plus
            s1 = _where(ap, jax.vmap(lambda s: pass_priority(bank, s))(s), s)
            ap2 = ap & ~s.cw_passed & apb(s1, ap & ~s.cw_passed, jnp.zeros_like(ap))
            s2 = lax.cond(ap2.any(), lambda s1: _where(ap2, jax.vmap(lambda s: pass_priority(bank, s))(s1), s1), lambda s1: s1, s1)
            s2 = refresh_b(bank, s, s2)
            return s2, apb(s2, ap, ap & False), ap2.sum()
        s, ap_new, p2 = lax.cond(ap.any(), passes, lambda s: (s, ap, _i(0)), s)
        return s, i + 1, thr, k + jnp.array([0, 0, 0, 0, ap.any(), p2 > 0, 0, 0, 0, ap.sum(), p2], I32), ap_new

    def settle_b(s, thr, k, ap):
        c = lax.while_loop(lambda c: (todo(c[0], c[4]).sum() > thr) & (c[1] < MAX_ITER), settle_body,
                           (s, _i(0), thr, k, ap))
        return c[0], c[3], c[4]

    def rnd(s, m, a, k):  # un tour : actions a (>= 0) appliquées, résolution, masques et actions forcées suivantes
        act = (a >= 0) & ~s.done
        s1 = jax.vmap(lambda s, a: apply(bank, s, a))(s, jnp.clip(a, 0, na - 1))
        s2 = refresh_b(bank, s, _where(act & (a < na), s1, s))
        s2, k, ap = settle_b(s2, frac * act.sum(), k, apb(s2, act, jnp.zeros_like(act)))
        s = _where(act, s2, s)
        left = act & todo(s, ap)                  # arrêtées en cours de résolution : « continuer » à l'appel suivant
        s = refresh_ex_b(bank, s, act & ~left)
        fl = (s.fsel | (s.rsel > 0) | (s.psel > 0)).any()
        lg = lax.cond(fl, lambda s: jax.vmap(lambda s: legal(bank, s))(s),
                      lambda s: jax.vmap(lambda s: legal(bank, s, fus=False))(s), s)
        m = jnp.where(act[:, None], lg, m)
        f = jnp.where(left, na, jax.vmap(lambda s, m: forced_m(bank, s, m))(s, m))
        return s, m, jnp.where(act, f, -1), k + jnp.array([1, 0, 0, 0, 0, 0, fl, 0, 0, 0, 0], I32)

    ks = max(1, nb // 4)

    def body(carry):  # tours suivants : seulement les parties qui ont une action forcée, dans un sous-lot de B/4 (les
        s, m, f, i, k = carry  # autres attendent le tour suivant ; après le premier tour, ~30 % des parties en ont une)
        idx = jnp.nonzero((f >= 0) & ~s.done, size=ks, fill_value=nb)[0]
        g = lambda x: x[jnp.minimum(idx, nb - 1)]
        ss, ms, fs, k = rnd(jax.tree.map(g, s), g(m), jnp.where(idx < nb, g(f), -1), k)
        put = lambda x, y: x.at[idx].set(y, mode="drop")
        return jax.tree.map(put, s, ss), put(m, ms), put(f, fs), i + 1, k

    def more(c):
        pend = ((c[2] >= 0) & ~c[0].done).sum()
        return (pend > frac * nb) & (c[3] < MAX_ITER)

    # pas de settle_b initial (les états d'entrée sont à une décision). rnd est compilé deux fois (lot complet, sous-lot) :
    # compilation ~2× plus longue et pic de mémoire ~2× (5,8 Go sur CPU), pour ~+30 % de débit sur L4
    s, m, f, k = rnd(s, jnp.zeros((nb, n_actions(s.cid.shape[1])), bool), a.astype(I32), jnp.zeros(11, I32))
    s, m, f, _, k = lax.while_loop(more, body, (s, m, f, _i(1), k))
    return (s, m, f, k) if stats else (s, m, f)

def new_game(bank, deck0, deck1, key=None, extra0=None, extra1=None, extra_counts=None):
    """deck0/deck1 : [D] index dans la banque, dans l'ordre d'ajout (dernier = dessus du deck).
    extra0/extra1 : [X] Extra Deck (même capacité pour les deux joueurs ; None : aucun), dans l'ordre d'ajout.
    extra_counts : [2] tailles réelles optionnelles (0..X) ; les emplacements restants ont zone=0.
    Permet une seule compilation pour des Extra Decks de tailles différentes, sans cartes fictives en jeu.
    Rend l'état au début du premier tour, dont la première décision peut être à option unique : l'appelant la joue
    avec step (forced(bank, s) si >= 0), ou tire une action légale (alors la seule).

    key=None : ordre fixe, slots dans l'ordre des decks (pour coller à l'oracle en DUEL_PSEUDO_SHUFFLE).
    Avec une clé : decks mélangés et cartes réparties au hasard dans les slots, pour que ni l'ordre du deck
    ni le numéro de slot ne trahissent une carte cachée.
    """
    d = deck0.shape[0]
    x = 0 if extra0 is None else extra0.shape[0]
    nd = 2 * d
    n = nd + 2 * x
    if n > MAX_SLOTS:
        raise ValueError(f"capacité d'arène dépassée : {n} > {MAX_SLOTS}")
    cid = jnp.concatenate([deck0, deck1] + ([] if x == 0 else [extra0, extra1])).astype(I32)
    owner = jnp.concatenate([jnp.repeat(jnp.arange(2, dtype=I32), d), jnp.repeat(jnp.arange(2, dtype=I32), x)])
    seq = jnp.concatenate([jnp.tile(jnp.arange(d, dtype=I32), 2), jnp.tile(jnp.arange(x, dtype=I32), 2)])
    zone0 = jnp.concatenate([jnp.full(nd, Z_DECK, I32), jnp.full(2 * x, Z_EXTRA, I32)])
    if extra_counts is not None and x:
        active_extra = jnp.tile(jnp.arange(x, dtype=I32), 2) < jnp.repeat(jnp.asarray(extra_counts, I32), x)
        zone0 = zone0.at[nd:].set(jnp.where(active_extra, Z_EXTRA, 0))
    ar = lambda k: jnp.arange(k, dtype=I32)
    uid = jnp.concatenate([ar(d), d + x + ar(d), d + ar(x), 2 * d + x + ar(x)])
    if key is not None:
        k1, k2 = jax.random.split(key)
        own = owner[:nd]
        order = jnp.lexsort((jax.random.uniform(k1, (nd,)), own))  # mélange à l'intérieur de chaque deck
        seq = seq.at[:nd].set(jnp.zeros(nd, I32).at[order].set(jnp.arange(nd, dtype=I32)) - own * d)
        perm = jax.random.permutation(k2, n)
        cid, owner, seq, zone0, uid = cid[perm], owner[perm], seq[perm], zone0[perm], uid[perm]
    z = jnp.zeros(n, I32)
    f = jnp.bool_(False)
    s = State(reason=z, suminfo=z, cid=cid, zone=zone0,
              seq=seq, ctrl=owner, owner=owner, pos=jnp.full(n, P_FD_DEF, I32),
              st=z, lp=jnp.full(2, 8000, I32), turn=_i(0), tp=_i(1), phase=_i(PH_DRAW), ns_used=f,
              dec=_i(D_NONE), dec_player=_i(0), pend=_i(-1), pend_set=f, need=_i(0), picks=_i(0), smin=_i(0),
              zctx=_i(ZC_SUMMON), zsel=_i(0), ch_src=_i(-1), ch_src_mv=_i(0), ch_tgt_t=_i(0), ch_eff=_i(0), ch_p=_i(0), ch_k=_i(0), ch_dis=f, ch_lab=_i(0), ch_flags=_i(0), pg=_i(0),
              pc=_i(0), reg=jnp.zeros((R, n), bool), done=f, odraw=f, winner=_i(-1),
              lk_src=jnp.full(CH, -1, I32), lk_eff=jnp.zeros(CH, I32), lk_p=jnp.zeros(CH, I32),
              lk_reg=jnp.zeros((CH, R, n), bool), lk_neg=jnp.zeros(CH, bool), lk_t=jnp.zeros(CH, I32), lk_tgt_t=jnp.zeros(CH, I32), lk_lab=jnp.zeros(CH, I32), lk_flags=jnp.zeros(CH, I32), mv=z, clk=_i(0), n_chain=_i(0),
              wk=_i(W_QUICK), cw_passed=f, ret=_i(R_IDLE), chained=f, ph_restart=_i(0), wph=_i(WP_DRAW),
              bend_ep=f, atk=_i(-1), atk_tgt=_i(-1), ev_g=jnp.zeros((NEV, n), bool), ev_p=jnp.zeros(NEV, I32),
              pzone=zone0, tev=jnp.zeros((NEV, n), bool), tew=jnp.zeros((NEV, n), bool), teg=jnp.zeros((NEV, n), bool), tg=jnp.zeros(n, bool), tg_e=z,
              tst=_i(0), pt_p=_i(0), pt_passed=f, dstep=_i(DS_NONE), ds_t=_i(0), ds_flip=f, rb_m=jnp.zeros(n, bool), rb_t=_i(0), rpl=f, opt_cid=jnp.zeros((2, n * E), I32),
              lock_cid=jnp.zeros((2, n), I32), atk_c=bank.atk[cid], df_c=bank.df[cid], batk_c=bank.atk[cid], tm_code=jnp.zeros(TM, I32),
              tm_m=jnp.zeros((TM, n), bool), tm_v=jnp.zeros((TM, n), I32), tm_t=jnp.zeros(TM, I32), tm_dur=jnp.zeros(TM, I32), tm_rd=jnp.zeros(TM, bool), tm_src=jnp.full(TM, -1, I32), tm_n=_i(0),
              ovf=_i(0), en_t=z, fl_c=z, ib_c=jnp.full(n, BIG, I32), pf_c=jnp.zeros(2, I32), cav=jnp.zeros((n, E), bool), cav_t=jnp.zeros((n, E), I32),
              proc=jnp.zeros(n, bool), fz_c=jnp.zeros(n, bool), fsel=f, rz_c=z, rsel=_i(0), sz_c=jnp.zeros(n, bool),
              xz_c=jnp.zeros(n, bool), lz_c=jnp.zeros(n, bool), cl_c=jnp.zeros(n, bool), psel=_i(0), ov_par=jnp.full(n, -1, I32), ps_used=f, uid=uid)
    for p in (0, 1):
        for _ in range(5):
            s = draw(s, p)
    # refresh après la pioche : fz_c / rz_c dépendent de la main ; avant, ils restaient périmés jusqu'au premier
    # changement d'état (step les gardait, step_batch les recalculait en rafraîchissant tout le lot)
    s = refresh(bank, s)
    # pas de step ici : il recompilerait tout le moteur une seconde fois (≈ 2 Go de compilation en plus). La première
    # décision peut être à option unique (fenêtre de la Draw Phase) : l'appelant la joue avec son propre step
    # (forced, ou une action légale tirée au hasard, qui est alors la seule)
    return refresh_ex(bank, start_turn(s))
