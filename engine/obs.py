"""Observation d'un joueur : ce qu'il a le droit de voir, sous forme de tokens de taille fixe.

- Un token par carte visible : sa main, tout le terrain (les cartes verso adverses sans leur identité),
  les cimetières et les bannies. Les decks et la main adverse ne sont que des compteurs (token global).
- L'identité (cid, ATK/DEF/niveau) n'est donnée que si le joueur peut la connaître.
- L'ordre des tokens ne dépend que d'informations visibles (zone, contrôleur, seq), jamais du slot.
- `slot` sert uniquement à convertir le choix du réseau (verbe × token) en action du moteur. Ce n'est
  pas une entrée du réseau : le slot est tiré au hasard à chaque partie et ne doit rien apprendre.
- Au plus T tokens. Les cartes visées par une action légale passent en premier, puis main et terrain, puis
  cimetières et bannies (les plus récents d'abord). Les cartes en trop ne sont résumées que par les compteurs.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import core
from core import I32, NV

T = 48
ST_PUBLIC = core.ST_SUMMON_TURN | core.ST_ATTACKED | core.ST_FORM_CHANGED | core.ST_SET_TURN | core.ST_LEAVE
ZONE_RANK = {core.Z_HAND: 0, core.Z_MZONE: 1, core.Z_SZONE: 2, core.Z_GRAVE: 3, core.Z_REMOVED: 4}


class Obs(NamedTuple):
    valid: jnp.ndarray     # [T] bool
    known: jnp.ndarray     # [T] bool : identité visible
    cid: jnp.ndarray       # [T] index dans la banque, -1 si inconnue
    atk: jnp.ndarray       # [T] 0 si inconnue
    df: jnp.ndarray
    level: jnp.ndarray
    zone: jnp.ndarray      # [T]
    seq: jnp.ndarray
    mine: jnp.ndarray      # [T] bool : contrôlée par l'observateur
    pos: jnp.ndarray
    st: jnp.ndarray        # [T] statuts publics (+ ST_SEL pour ses propres choix en cours)
    pending: jnp.ndarray   # [T] bool : carte en cours (monstre à invoquer, attaquant, effet en résolution)
    chain: jnp.ndarray     # [T] rang du maillon de chaîne dont elle est la source (1 = premier), 0 sinon
    negated: jnp.ndarray   # [T] bool : ce maillon est nié
    act: jnp.ndarray       # [T, NV] bool : actions légales verbe × token
    special: jnp.ndarray   # [N_SPECIAL] bool : actions spéciales légales
    glob: jnp.ndarray      # [G] int : LP, tour, phase, décision, compteurs
    slot: jnp.ndarray      # [T] PAS une entrée du réseau : sert à reconstruire l'action


def _count(s, zone, p):
    return ((s.zone == zone) & (s.ctrl == p)).sum().astype(I32)


def observe(bank, s, p=None):
    """Observation du joueur p (par défaut celui qui doit décider)."""
    p = s.dec_player if p is None else jnp.asarray(p, I32)
    n = s.cid.shape[0]
    mine = s.ctrl == p
    faceup = (s.pos & (core.P_FU_ATK | core.P_FU_DEF)) != 0
    field = (s.zone == core.Z_MZONE) | (s.zone == core.Z_SZONE)
    public = (s.zone == core.Z_GRAVE) | (s.zone == core.Z_REMOVED)
    shown = (mine & (s.zone == core.Z_HAND)) | field | public
    known = shown & ((mine & ((s.zone == core.Z_HAND) | field)) | (field & faceup) | public)

    m = core.legal(bank, s)
    act_slot = m[:NV * n].reshape(NV, n).T            # [N, NV]
    acted = act_slot.any(1)
    # p n'observe ses actions que si c'est lui qui décide
    acted = acted & (s.dec_player == p)
    act_slot = act_slot & (s.dec_player == p)

    zrank = jnp.zeros(n, I32)
    for z, r in ZONE_RANK.items():
        zrank = jnp.where(s.zone == z, r, zrank)
    cls = jnp.where(acted, 0, jnp.where(public, 2, 1))
    seqkey = jnp.where(public, 999 - s.seq, s.seq)
    score = cls * 1_000_000 + (~mine) * 100_000 + zrank * 1_000 + seqkey
    score = jnp.where(shown, score, 10_000_000)
    _, idx = jax.lax.top_k(-score, T)                 # ordre fixé par des informations visibles uniquement
    valid = shown[idx]
    kn = known[idx] & valid

    c = s.cid[idx]
    own_sel = jnp.where(mine[idx], s.st[idx] & core.ST_SEL, 0)
    pend = (((s.pend >= 0) & (idx == s.pend)) | ((s.ch_src >= 0) & (idx == s.ch_src))
            | ((s.atk >= 0) & (idx == s.atk)) | ((s.atk_tgt >= 0) & (idx == s.atk_tgt))) & valid
    live = jnp.arange(core.CH) < s.n_chain
    link = jnp.zeros(n, I32).at[jnp.where(live, s.lk_src, n)].set(jnp.arange(1, core.CH + 1, dtype=I32), mode="drop")
    neg = jnp.zeros(n, bool).at[jnp.where(live, s.lk_src, n)].set(s.lk_neg, mode="drop")
    glob = jnp.stack([
        s.lp[p], s.lp[1 - p], s.turn, (s.tp == p).astype(I32), s.phase, s.ns_used.astype(I32),
        jnp.where(s.dec_player == p, s.dec, 0), jnp.where(s.dec_player == p, s.need, 0),
        jnp.where(s.dec_player == p, s.zctx, 0), s.n_chain,
        *[_count(s, z, q) for q in (p, 1 - p)
          for z in (core.Z_HAND, core.Z_DECK, core.Z_MZONE, core.Z_SZONE, core.Z_GRAVE, core.Z_REMOVED)],
    ]).astype(I32)
    zero = lambda x: jnp.where(kn, x, 0)
    return Obs(valid=valid, known=kn, cid=jnp.where(kn, c, -1), atk=zero(bank.atk[c]), df=zero(bank.df[c]),
               level=zero(bank.level[c]), zone=jnp.where(valid, s.zone[idx], 0), seq=jnp.where(valid, s.seq[idx], 0),
               mine=mine[idx] & valid, pos=jnp.where(valid, s.pos[idx], 0),
               st=jnp.where(valid, (s.st[idx] & ST_PUBLIC) | own_sel, 0), pending=pend,
               chain=jnp.where(valid, link[idx], 0), negated=neg[idx] & valid,
               act=act_slot[idx] & valid[:, None], special=m[NV * n:] & (s.dec_player == p), glob=glob,
               slot=jnp.where(valid, idx, -1).astype(I32))


def to_action(obs, n, verb, tok):
    """Choix du réseau (verbe, token) → action du moteur."""
    return (verb * n + obs.slot[tok]).astype(I32)


def special_action(n, k):
    return jnp.asarray(NV * n + k, I32)
