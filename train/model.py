"""Réseau de politique et de valeur (jalon 6, étape 3 ; ETAPE3.md).

Deux étages :
1. `encode_bank` : chaque carte de la banque est lue depuis ses briques (en-tête, effets, instructions de ses
   programmes, filtres), jamais depuis un identifiant. Un petit transformer par carte (≈ 90 tokens : en-tête, un
   token par effet, un par instruction, un par filtre, un par filtre de matériaux) donne un vecteur [d] par carte.
   Calculé une fois par rollout (paramètres fixes), recalculé à chaque minibatch de la mise à jour (gradient).
2. `policy` : les T tokens de l'observation (engine/obs.py) = vecteur de la carte (ou « inconnue ») + son état visible
   (zone, position, ATK / DEF courantes, statuts, chaîne…), plus un token global. Transformer, puis
   logits verbe × token, logits des actions spéciales (sur le token global), valeur (token global).

Les entiers des briques (paramètres a, b, c, champs des filtres) sont donnés par symlog(x) et leurs 16 bits de poids
faible : la plupart sont des champs de bits (zones, types, positions, événements) ou des codes (archétype, nom), que le
réseau doit pouvoir comparer entre cartes (même archétype) sans identifiant appris.
Paramètres en pytree de dicts ; jax pur (pas de flax), optax pour l'optimiseur.
"""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import core  # noqa: E402
from core import E, P, NF, FW, MW, NSET, FK, NV, N_SPECIAL, CH  # noqa: E402

NBITS = 16
N_OPS = 32
N_KIND = 8
N_DEC = 13
N_PHASE = 16


# ---------------------------------------------------------------------------------------------------- briques

def ifeat(x):
    """Entier(s) → [..., 1 + NBITS] : symlog et bits de poids faible."""
    x = x.astype(jnp.int32)
    sl = jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0
    bits = ((jnp.abs(x)[..., None] >> jnp.arange(NBITS)) & 1).astype(jnp.float32)
    return jnp.concatenate([sl[..., None], bits], -1)


def bits(x, n):
    return ((x.astype(jnp.int32)[..., None] >> jnp.arange(n)) & 1).astype(jnp.float32)


def onehot(x, n):
    return jax.nn.one_hot(jnp.clip(x, 0, n - 1), n)


def dense_init(k, i, o, scale=1.0):
    return dict(w=jax.random.normal(k, (i, o)) * scale / jnp.sqrt(i), b=jnp.zeros(o))


CDT = jnp.float32   # type de calcul (jnp.bfloat16 : précision mixte, paramètres en f32) ; voir set_compute


def set_compute(bf16):
    global CDT
    CDT = jnp.bfloat16 if bf16 else jnp.float32


def dense(p, x):
    return x.astype(CDT) @ p["w"].astype(CDT) + p["b"].astype(CDT)


def ln_init(d):
    return dict(g=jnp.ones(d), b=jnp.zeros(d))


def ln(p, x):
    x = x.astype(jnp.float32)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return ((x - m) * jax.lax.rsqrt(v + 1e-5) * p["g"] + p["b"]).astype(CDT)


def block_init(k, d, h):
    ks = jax.random.split(k, 4)
    return dict(ln1=ln_init(d), qkv=dense_init(ks[0], d, 3 * d), o=dense_init(ks[1], d, d, 0.5),
                ln2=ln_init(d), f1=dense_init(ks[2], d, 4 * d), f2=dense_init(ks[3], 4 * d, d, 0.5))


def block(p, x, mask, h):
    """Bloc pré-LN. x [..., L, d], mask [..., L] (les tokens masqués ne sont pas lus)."""
    d = x.shape[-1]
    y = ln(p["ln1"], x)
    q, k, v = jnp.split(dense(p["qkv"], y), 3, -1)
    sh = lambda t: t.reshape(t.shape[:-1] + (h, d // h))
    q, k, v = sh(q), sh(k), sh(v)
    att = jnp.einsum("...qhc,...khc->...hqk", q, k) / jnp.sqrt(d // h)
    att = jnp.where(mask[..., None, None, :], att.astype(jnp.float32), -1e9)
    y = jnp.einsum("...hqk,...khc->...qhc", jax.nn.softmax(att, -1).astype(CDT), v).reshape(x.shape)
    x = x + dense(p["o"], y)
    return x + dense(p["f2"], jax.nn.gelu(dense(p["f1"], ln(p["ln2"], x))))


# ---------------------------------------------------------------------------------------------------- cartes

F_INS = 3 * (1 + NBITS) + 16                       # a, b, c
F_FILT = FW * (1 + NBITS) + 16
F_EFF = 32 + 3 * 8 + MW * (1 + NBITS) + 3     # etrig, eopt, erng, edmg ; emod ; inline, remboursement
F_HEAD = 3 + 32 + 8 + 32 + NSET * NBITS + 9 + 2 + 8 + FK   # stats, ctype, attr, race, setcode, flèches, échelles, procédures
N_MATF = FK + 2 + 1 + 1                        # filtres de matériaux : fusion, synchro (2), Xyz, Link


def card_tokens(bank):
    """Banque → (features par type de token, masques). Formes fixes pour toutes les banques procédurales."""
    sl = lambda x: (jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0)[..., None]
    head = jnp.concatenate([
        sl(bank.atk), sl(bank.df), sl(bank.level), bits(bank.ctype, 32), bits(bank.attr, 8), bits(bank.race, 32),
        bits(bank.setcode, NBITS).reshape(bank.setcode.shape[0], -1), bits(bank.lmark, 9), sl(bank.lsc), sl(bank.rsc),
        sl(bank.fmn), sl(bank.xct), sl(bank.lmin), sl(bank.lmax), sl(bank.lneed), sl(bank.ldiff), sl(bank.syn), sl(bank.erl), bits(bank.fsub, FK),
    ], -1)
    inline = bank.einline if bank.einline.shape[1] else jnp.zeros_like(bank.ekind)
    refund = bank.erefund if bank.erefund.shape[1] else jnp.zeros_like(bank.ekind)
    eff = jnp.concatenate([bits(bank.etrig, 32), bits(bank.eopt, 8), bits(bank.erng, 8), bits(bank.edmg, 8),
                           ifeat(bank.emod).reshape(bank.emod.shape[:2] + (-1,)), bits(inline, 2), bits(refund, 1)], -1)
    prog = bank.prog                                                                          # [C, E, 3, P, 4]
    ins_op = prog[..., 0]
    ins = jnp.concatenate([ifeat(prog[..., 1:]).reshape(prog.shape[:-1] + (-1,)), bits(prog[..., 3], 32)[..., 16:]], -1)                               # [C, E, 3, P, F_INS]
    filt = jnp.concatenate([ifeat(bank.filt[..., :-1]).reshape(bank.filt.shape[:-1] + (-1,)), bits(jnp.abs(bank.filt[..., core.F_REASON]), 32)[..., 16:], ifeat(bank.filt[..., -1])], -1)                             # [C, E, NF, F_FILT]
    mats = jnp.concatenate([bank.fmat, bank.smat, bank.xmat[:, None], bank.lmat[:, None]], 1)   # [C, N_MATF, FW]
    matf = jnp.concatenate([ifeat(mats[..., :-1]).reshape(mats.shape[:-1] + (-1,)), bits(jnp.abs(mats[..., core.F_REASON]), 32)[..., 16:], ifeat(mats[..., -1])], -1)
    has = jnp.concatenate([jnp.arange(FK)[None] < bank.fmn[:, None], jnp.repeat(bank.syn[:, None] != 0, 2, 1),
                           bank.xct[:, None] > 0, bank.lmax[:, None] > 0], 1)
    emask = bank.ekind != core.K_NONE                                                         # [C, E]
    imask = (ins_op != core.OP_END) & emask[:, :, None, None]
    fmask = (bank.filt != 0).any(-1) & emask[:, :, None]
    return dict(head=head, eff=eff, ekind=bank.ekind, ins=ins, ins_op=ins_op, filt=filt, matf=matf), \
        dict(eff=emask, ins=imask, filt=fmask, mat=has)


def card_enc_init(k, d, layers):
    ks = jax.random.split(k, 12 + layers)
    return dict(head=dense_init(ks[0], F_HEAD, d), eff=dense_init(ks[1], F_EFF, d), kind=jax.random.normal(ks[2], (N_KIND, d)) * 0.1,
                ins=dense_init(ks[3], F_INS, d), op=jax.random.normal(ks[4], (N_OPS, d)) * 0.1,
                filt=dense_init(ks[5], F_FILT, d), matf=dense_init(ks[6], F_FILT, d),
                pos_e=jax.random.normal(ks[7], (E, d)) * 0.1, pos_g=jax.random.normal(ks[8], (3, d)) * 0.1,
                pos_i=jax.random.normal(ks[9], (P, d)) * 0.1, pos_f=jax.random.normal(ks[10], (NF, d)) * 0.1,
                pos_m=jax.random.normal(ks[11], (N_MATF, d)) * 0.1,
                blocks=[block_init(ks[12 + i], d, 4) for i in range(layers)], out=ln_init(d))


LC = 48   # tokens utiles gardés par carte (en-tête compris) ; ~8 en moyenne, 33 au plus sur 30 banques, 26 vraies cartes


def n_tokens(msk):
    """Tokens utiles par carte (en-tête compris), [C] : doit rester <= LC (voir `encode_cards`)."""
    C = msk["eff"].shape[0]
    return 1 + sum(v.reshape(C, -1).sum(1) for v in msk.values())


def encode_cards(p, tok, msk, heads=4):
    """[C, d] : un vecteur par carte (sortie du token d'en-tête).
    Sur les 91 tokens possibles, ~8 sont utiles : ils sont regroupés en tête (tri stable, l'en-tête reste premier) et la
    séquence est coupée à LC avant le transformer. Les positions sont ajoutées avant ; le résultat est le même que sur
    les 91 tokens tant qu'aucune carte n'a plus de LC tokens utiles."""
    C = tok["head"].shape[0]
    pe = p["pos_e"]
    # Les anciens checkpoints précèdent les FK permissions de substitution.
    head = tok["head"]
    if p["head"]["w"].shape[0] == F_HEAD - FK:
        head = head[:, :-FK]  # même produit matriciel qu'avant ; aucun poids inventé
    elif p["head"]["w"].shape[0] != F_HEAD:
        raise ValueError("dimension d'en-tête de checkpoint incompatible")
    x_head = dense(p["head"], head)[:, None]
    eff = tok["eff"]
    if p["eff"]["w"].shape[0] == F_EFF - 3:
        eff = eff[..., :-3]
    elif p["eff"]["w"].shape[0] != F_EFF:
        raise ValueError("dimension d'effet de checkpoint incompatible")
    x_eff = dense(p["eff"], eff) + p["kind"][jnp.clip(tok["ekind"], 0, N_KIND - 1)] + pe
    if p["ins"]["w"].shape[0] not in (F_INS, F_INS - 16):
        raise ValueError("dimension d'instruction de checkpoint incompatible")
    if any(p[key]["w"].shape[0] not in (F_FILT, F_FILT - 17, F_FILT - 50) for key in ("filt", "matf")):
        raise ValueError("dimension de filtre de checkpoint incompatible")
    ins = tok["ins"][..., :p["ins"]["w"].shape[0]]
    filt = tok["filt"][..., :p["filt"]["w"].shape[0]]
    matf = tok["matf"][..., :p["matf"]["w"].shape[0]]
    x_ins = (dense(p["ins"], ins) + p["op"][jnp.clip(tok["ins_op"], 0, N_OPS - 1)]
             + pe[:, None, None] + p["pos_g"][:, None] + p["pos_i"])
    x_filt = dense(p["filt"], filt) + pe[:, None] + p["pos_f"]
    x_mat = dense(p["matf"], matf) + p["pos_m"]
    x = jnp.concatenate([x_head, x_eff, x_ins.reshape(C, -1, x_head.shape[-1]), x_filt.reshape(C, -1, x_head.shape[-1]),
                         x_mat], 1)
    m = jnp.concatenate([jnp.ones((C, 1), bool), msk["eff"], msk["ins"].reshape(C, -1), msk["filt"].reshape(C, -1),
                         msk["mat"]], 1)
    order = jnp.argsort(~m, axis=1, stable=True)[:, :LC]
    x = jnp.take_along_axis(x, order[..., None], 1)
    m = jnp.take_along_axis(m, order, 1)
    for b in p["blocks"]:
        x = block(b, x, m, heads)
    return ln(p["out"], x[:, 0]).astype(jnp.float32)


# ---------------------------------------------------------------------------------------------------- politique

F_DYN = 3 + 8 + 16 + 4 + 17 + 1 + (CH + 1) + 1 + 1 + 1 + 64
F_GLOB = 2 + 2 + N_PHASE + 3 + N_DEC + 1 + 8 + 1 + 12  # LP, tour, phase, flags, décision, need, zctx, chaîne, compteurs


def dyn_feats(o):
    sl = lambda x: (jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0)[..., None]
    zone = bits(o.zone, 8)
    return jnp.concatenate([sl(o.atk), sl(o.df), sl(o.level), zone, onehot(o.seq, 16), bits(o.pos, 4), bits(o.st, 17),
                            o.mine[..., None].astype(jnp.float32), onehot(o.chain, CH + 1),
                            o.negated[..., None].astype(jnp.float32), o.pending[..., None].astype(jnp.float32),
                            o.card_activation[..., None].astype(jnp.float32), bits(o.suminfo, 32), bits(o.reason, 32)], -1)


def glob_feats(o):
    g = o.glob
    f = lambda x: x.astype(jnp.float32)[..., None]
    return jnp.concatenate([f(g[..., 0]) / 8000, f(g[..., 1]) / 8000, f(jnp.minimum(g[..., 2], 100)) / 50, f(g[..., 3]),
                            bits(g[..., 4], N_PHASE), f(g[..., 5]), f(g[..., 6]), f(g[..., 3]) * 0,
                            onehot(g[..., 7], N_DEC), f(g[..., 8]) / 4, onehot(g[..., 9], 8), f(g[..., 10]) / CH,
                            g[..., 11:23].astype(jnp.float32) / 20], -1)


def init(k, d=128, card_layers=2, layers=3):
    ks = jax.random.split(k, 8 + layers)
    return dict(cards=card_enc_init(ks[0], d, card_layers), unk=jax.random.normal(ks[1], (d,)) * 0.1,
                dyn=dense_init(ks[2], F_DYN, d), glob=dense_init(ks[3], F_GLOB, d),
                blocks=[block_init(ks[8 + i], d, 4) for i in range(layers)], out=ln_init(d),
                act=dense_init(ks[4], d, NV, 0.01), spec=dense_init(ks[5], d, N_SPECIAL, 0.01),
                v1=dense_init(ks[6], d, d), v2=dense_init(ks[7], d, 1, 0.01))


def policy(p, cemb, o, heads=4):
    """o : Obs (engine/obs.py) d'un lot [..]. Rend (logits [.., T*NV + N_SPECIAL] masqués, valeur [..])."""
    known = o.known[..., None]
    dyn = dyn_feats(o)
    if p["dyn"]["w"].shape[0] in (F_DYN - 32, F_DYN - 64, F_DYN - 65):
        dyn = dyn[..., :p["dyn"]["w"].shape[0]]
    elif p["dyn"]["w"].shape[0] != F_DYN:
        raise ValueError("dimension dynamique de checkpoint incompatible")
    x = jnp.where(known, cemb[jnp.clip(o.cid, 0, cemb.shape[0] - 1)], p["unk"]) + dense(p["dyn"], dyn)
    g = dense(p["glob"], glob_feats(o))[..., None, :]
    x = jnp.concatenate([g, x], -2)
    m = jnp.concatenate([jnp.ones(o.valid.shape[:-1] + (1,), bool), o.valid], -1)
    for b in p["blocks"]:
        x = block(b, x, m, heads)
    x = ln(p["out"], x)
    la = dense(p["act"], x[..., 1:, :])                               # [.., T, NV]
    ls = dense(p["spec"], x[..., 0, :])                               # [.., N_SPECIAL]
    logits = jnp.concatenate([la.reshape(la.shape[:-2] + (-1,)), ls], -1)
    legal = jnp.concatenate([o.act.reshape(o.act.shape[:-2] + (-1,)), o.special], -1)
    logits = jnp.where(legal, logits.astype(jnp.float32), -1e9)
    v = jnp.tanh(dense(p["v2"], jax.nn.gelu(dense(p["v1"], x[..., 0, :])))[..., 0].astype(jnp.float32))
    return logits, v


def to_engine(o, n, idx):
    """Index de la sortie du réseau → action du moteur (verbe × slot, ou action spéciale)."""
    T = o.slot.shape[-1]
    tok, verb = idx // NV, idx % NV
    slot = jnp.take_along_axis(o.slot, jnp.clip(tok, 0, T - 1)[..., None], -1)[..., 0]
    return jnp.where(idx < T * NV, verb * n + slot, NV * n + (idx - T * NV)).astype(jnp.int32)
