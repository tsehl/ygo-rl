"""Tests de l'observation (parties mélangées, slots aléatoires) :
1. toute action légale correspond à un token (rien n'est tronqué) ;
2. l'observation ne dépend pas de l'information cachée : on mélange les identités des cartes que
   l'observateur ne peut pas connaître et on renumérote les slots, l'observation doit rester identique
   (sauf `slot`) ;
3. contrôle positif : changer une carte visible change bien l'observation.
"""
import jax
import jax.numpy as jnp
import numpy as np

import cards
import core
import obs as O

B, D, C, TT = 32, 40, 670, 300
key = jax.random.PRNGKey(0)
bank = cards.test_bank(key, C)
CT = bank.code.shape[0]
kd = jax.random.fold_in(key, 3)
decks = jnp.where(jax.random.uniform(kd, (B, 2, D)) < 0.3,          # 30 % de cartes à effet
                  jax.random.randint(jax.random.fold_in(kd, 1), (B, 2, D), C, CT),
                  jax.random.randint(jax.random.fold_in(kd, 2), (B, 2, D), 0, C))
N = 2 * D


def rollout(d, k):
    s = core.new_game(bank, d[0], d[1], jax.random.fold_in(k, 7))

    def body(s, kt):
        m = core.legal(bank, s)
        a = jax.random.categorical(kt, jnp.where(m, 0.0, -1e9))
        s2 = core.step(bank, s, a)
        s2 = jax.tree.map(lambda x, y: jnp.where(s.done, x, y), s, s2)
        return s2, s
    return jax.lax.scan(body, s, jax.random.split(k, TT))[1]


hist = jax.jit(jax.vmap(rollout))(decks, jax.random.split(jax.random.fold_in(key, 4), B))
hist = jax.tree.map(np.asarray, hist)
live = ~hist.done
print("états vivants :", int(live.sum()))

observe = jax.jit(O.observe)


def state_at(b, t):
    return core.State(*[jnp.asarray(x[b, t]) for x in hist])


# 1. aucune action légale perdue
legal_cnt = jax.jit(jax.vmap(jax.vmap(lambda s: core.legal(bank, s)[:core.NV * N].sum())))(hist)
tok_cnt = jax.jit(jax.vmap(jax.vmap(lambda s: O.observe(bank, s).act.sum())))(hist)
bad = live & (np.asarray(legal_cnt) != np.asarray(tok_cnt))
assert not bad.any(), f"{bad.sum()} états avec des actions légales sans token"
print("1. ok : toutes les actions légales ont un token")

# 2. invariance à l'information cachée
rng = np.random.default_rng(0)
FIELDS = [f for f in O.Obs._fields if f != "slot"]


legal_j = jax.jit(lambda s: core.legal(bank, s))


def perturb(s, p):
    s = jax.tree.map(np.array, s)
    faceup = (s.pos & (core.P_FU_ATK | core.P_FU_DEF)) != 0
    # pendant une recherche dans son deck, p voit tout son deck
    search = (int(s.dec_player) == p) & np.asarray(legal_j(core.State(*[jnp.asarray(x) for x in s]))[:core.NV * N]
                                                    .reshape(core.NV, N).any(0) & (s.zone == core.Z_DECK)).any()
    hidden = (((s.zone == core.Z_DECK) & ~(search & (s.ctrl == p)))
              | ((s.zone == core.Z_HAND) & (s.ctrl != p))
              | (((s.zone == core.Z_MZONE) | (s.zone == core.Z_SZONE)) & (s.ctrl != p) & ~faceup))
    h = np.where(hidden)[0]
    cid = s.cid.copy()
    # identités cachées arbitraires (seule la zone d'une carte posée est publique : magie/piège ou monstre)
    spells = np.where(np.asarray(bank.ctype) & (core.TYPE_SPELL | core.TYPE_TRAP))[0]
    mons = np.where(np.asarray(bank.ctype) & core.TYPE_MONSTER)[0]
    cid[h] = np.select([s.zone[h] == core.Z_SZONE, s.zone[h] == core.Z_MZONE],
                       [rng.choice(spells, len(h)), rng.choice(mons, len(h))], rng.integers(0, CT, len(h)))
    s = s._replace(cid=cid)
    perm = rng.permutation(N)                      # renumérotation des slots
    inv = np.argsort(perm)
    sl = lambda x: np.where(x >= 0, inv[x], -1)
    s = s._replace(**{f: getattr(s, f)[perm] for f in ("cid", "zone", "seq", "ctrl", "owner", "pos", "st", "pzone", "tg", "tg_e",
                                                  "atk_c", "df_c", "batk_c", "en_t", "cav", "cav_t", "fl_c", "ib_c")}, tm_m=s.tm_m[:, perm], tm_v=s.tm_v[:, perm],
                   pend=sl(s.pend), ch_src=sl(s.ch_src), atk=sl(s.atk), atk_tgt=sl(s.atk_tgt), lk_src=sl(s.lk_src),
                   reg=s.reg[:, perm], lk_reg=s.lk_reg[:, :, perm], mv=s.mv[perm], ev_g=s.ev_g[:, perm], tev=s.tev[:, perm])
    return core.State(*[jnp.asarray(x) for x in s]), len(h)


nchk = 0
samples = [tuple(x) for x in np.argwhere(live)]
for i in rng.choice(len(samples), 400, replace=False):
    b, t = samples[i]
    s = state_at(b, t)
    for p in (0, 1):
        o1 = observe(bank, s, p)
        s2, nh = perturb(s, p)
        o2 = observe(bank, s2, p)
        diff = [f for f in FIELDS if not np.array_equal(getattr(o1, f), getattr(o2, f))]
        assert not diff, f"fuite d'information (b={b}, t={t}, p={p}) : {diff}"
        # les slots diffèrent mais désignent les mêmes cartes
        v = np.asarray(o1.valid)
        assert np.array_equal(np.asarray(s.zone)[np.asarray(o1.slot)[v]], np.asarray(s2.zone)[np.asarray(o2.slot)[v]])
        nchk += 1
print(f"2. ok : {nchk} observations invariantes à l'information cachée et aux slots")

# 3. contrôle positif : changer une carte de sa propre main change l'observation
changed = 0
for i in rng.choice(len(samples), 100, replace=False):
    b, t = samples[i]
    s = state_at(b, t)
    p = int(s.dec_player)
    hand = np.where((np.asarray(s.zone) == core.Z_HAND) & (np.asarray(s.ctrl) == p))[0]
    if not len(hand):
        continue
    cid = np.array(s.cid)
    cid[hand[0]] = (cid[hand[0]] + 1) % CT
    o1, o2 = observe(bank, s, p), observe(bank, s._replace(cid=jnp.asarray(cid)), p)
    changed += not np.array_equal(o1.cid, o2.cid)
print(f"3. ok : {changed} contrôles positifs détectés" if changed else "3. ÉCHEC : rien détecté")
