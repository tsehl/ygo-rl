"""Test de faisabilité : interpréteur d'effets façon Yu-Gi-Oh vectorisé en JAX.

Pas les vraies règles, mais la même forme de calcul :
- 80 cartes par partie, chacune avec jusqu'à E effets = programmes de L opcodes (cartes procédurales aléatoires)
- 32 opcodes exécutés via lax.switch (sous vmap -> toutes les branches calculées)
- pile de chaîne de profondeur CH, réponses adverses, masque d'actions légales avec conditions
- tours, OPT, pioche, fin de partie + reset

Mesure le débit en coups/s : politique aléatoire (moteur seul) puis politique transformer.
Usage : python interp_bench.py [--quick]
"""
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

N, E, L, CH, NOPS, MZ = 80, 4, 8, 4, 32, 5
Z_DECK, Z_HAND, Z_MZ, Z_SZ, Z_GY, Z_BAN = 0, 1, 2, 3, 4, 5
NA = N * E + 1  # actions : (carte, effet) ou passer
MAX_TURNS = 200


# ---------------------------------------------------------------- helpers

def rank(mask, order):
    """Rang de chaque carte du masque selon `order` (0 = dessus)."""
    return jnp.sum(mask[None, :] & (order[None, :] < order[:, None]), axis=1)


def match(M, S, me, zsel, osel, thr):
    rel = jnp.where(osel % 3 == 0, S["owner"] == me,
                    jnp.where(osel % 3 == 1, S["owner"] != me, True))
    af = jnp.where(thr == 0, True, M["atk"] <= thr * 500)
    return (M["zone"] == zsel % 6) & rel & af


def move(M, sel, to):
    return {**M, "zone": jnp.where(sel, to, M["zone"])}


def move_first(M, mask, k, to):
    return move(M, mask & (rank(mask, M["order"]) < k), to)


def deck_of(M, S, p):
    return (M["zone"] == Z_DECK) & (S["owner"] == p)


def count(m):
    return jnp.sum(m.astype(jnp.int32))


def add_lp(M, p, v):
    return {**M, "lp": M["lp"].at[p].add(v)}


# ---------------------------------------------------------------- opcodes
# signature : (M, S, me, link, a0, a1, a2) -> (M, ctrl)  ctrl: 0 continue, 1 abort, 2 skip next

C, A, K = 0, 1, 2


def op_nop(M, S, me, l, a0, a1, a2): return M, C
def op_draw(M, S, me, l, a0, a1, a2): return move_first(M, deck_of(M, S, me), a0 % 3 + 1, Z_HAND), C
def op_damage(M, S, me, l, a0, a1, a2): return add_lp(M, 1 - me, -(a0 + 1) * 100), C
def op_recover(M, S, me, l, a0, a1, a2): return add_lp(M, me, (a0 + 1) * 100), C


def op_destroy(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2) & (M["flags"] & 1 == 0)
    return move(M, sel, Z_GY), C


def op_banish(M, S, me, l, a0, a1, a2): return move(M, match(M, S, me, a0, a1, a2), Z_BAN), C
def op_bounce(M, S, me, l, a0, a1, a2): return move(M, match(M, S, me, Z_MZ, a1, a2), Z_HAND), C


def op_spsummon(M, S, me, l, a0, a1, a2):
    free = MZ - count((M["zone"] == Z_MZ) & (S["owner"] == me))
    src = jnp.where(a0 % 2 == 0, Z_HAND, Z_GY)
    return move_first(M, match(M, S, me, src, 0, a2), jnp.minimum(1, free), Z_MZ), C


def _atk(M, sel, v): return {**M, "atk": jnp.where(sel, v, M["atk"])}


def op_atk_up(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return _atk(M, sel, M["atk"] + (a0 + 1) * 100), C


def op_atk_down(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return _atk(M, sel, jnp.maximum(0, M["atk"] - (a0 + 1) * 100)), C


def op_set_flag(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return {**M, "flags": jnp.where(sel, M["flags"] | (1 << (a0 % 4)), M["flags"])}, C


def op_clear_flag(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return {**M, "flags": jnp.where(sel, M["flags"] & ~(1 << (a0 % 4)), M["flags"])}, C


def op_search(M, S, me, l, a0, a1, a2):
    return move_first(M, deck_of(M, S, me) & (S["arch"] == a0), 1, Z_HAND), C


def op_mill(M, S, me, l, a0, a1, a2):
    p = jnp.where(a1 % 2 == 0, me, 1 - me)
    return move_first(M, deck_of(M, S, p), a0 % 3 + 1, Z_GY), C


def op_negate(M, S, me, l, a0, a1, a2):
    neg = M["chain_neg"].at[jnp.maximum(l - 1, 0)].set(True)
    return {**M, "chain_neg": jnp.where(l > 0, neg, M["chain_neg"])}, C


def op_cond_count(M, S, me, l, a0, a1, a2):
    return M, jnp.where(count(match(M, S, me, a0, a1, 0)) < a2 % 3, A, C)


def op_cond_lp(M, S, me, l, a0, a1, a2): return M, jnp.where(M["lp"][me] < a0 * 1000, A, C)


def op_discard(M, S, me, l, a0, a1, a2):
    hand = (M["zone"] == Z_HAND) & (S["owner"] == me)
    return move_first(M, hand, a0 % 2 + 1, Z_GY), C


def op_tribute(M, S, me, l, a0, a1, a2):
    mine = (M["zone"] == Z_MZ) & (S["owner"] == me)
    return move_first(M, mine, 1, Z_GY), jnp.where(mine.any(), C, A)


def op_swap(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return {**M, "atk": jnp.where(sel, M["def"], M["atk"]), "def": jnp.where(sel, M["atk"], M["def"])}, C


def op_double(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return _atk(M, sel, jnp.minimum(M["atk"] * 2, 10000)), C


def op_set_atk(M, S, me, l, a0, a1, a2): return _atk(M, match(M, S, me, Z_MZ, a1, a2), a0 * 500), C
def op_gain_per(M, S, me, l, a0, a1, a2): return add_lp(M, me, 100 * count(match(M, S, me, a0, a1, a2))), C
def op_dmg_per(M, S, me, l, a0, a1, a2): return add_lp(M, 1 - me, -100 * count(match(M, S, me, a0, a1, a2))), C


def op_to_deck(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, a0, a1, a2)
    return {**move(M, sel, Z_DECK), "order": jnp.where(sel, M["order"] + 1000, M["order"])}, C


def op_shuffle(M, S, me, l, a0, a1, a2):
    return {**M, "order": (M["order"] * 7 + a0 * 13 + 1) % 997}, C


def op_copy(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    best = jnp.max(jnp.where(sel, M["atk"], 0))
    return _atk(M, sel, best), C


def op_add_counter(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return {**M, "counters": jnp.where(sel, M["counters"] + a0 + 1, M["counters"])}, C


def op_remove_counter(M, S, me, l, a0, a1, a2):
    mine = (M["zone"] == Z_MZ) & (S["owner"] == me)
    ok = jnp.sum(jnp.where(mine, M["counters"], 0)) >= a0
    first = mine & (rank(mine, M["order"]) == 0)
    cnt = jnp.where(ok & first, jnp.maximum(0, M["counters"] - a0), M["counters"])
    return {**M, "counters": cnt}, jnp.where(ok, C, A)


def op_change_pos(M, S, me, l, a0, a1, a2):
    sel = match(M, S, me, Z_MZ, a1, a2)
    return {**M, "flags": jnp.where(sel, M["flags"] ^ 16, M["flags"])}, C


def op_skip_if(M, S, me, l, a0, a1, a2):
    return M, jnp.where(count(match(M, S, me, a0, a1, a2)) == 0, K, C)


def op_end(M, S, me, l, a0, a1, a2): return M, A


OPS = [op_nop, op_draw, op_damage, op_recover, op_destroy, op_banish, op_bounce, op_spsummon,
       op_atk_up, op_atk_down, op_set_flag, op_clear_flag, op_search, op_mill, op_negate,
       op_cond_count, op_cond_lp, op_discard, op_tribute, op_swap, op_double, op_set_atk,
       op_gain_per, op_dmg_per, op_to_deck, op_shuffle, op_copy, op_add_counter,
       op_remove_counter, op_change_pos, op_skip_if, op_end]
assert len(OPS) == NOPS
OP_COND = 15


def wrap(f):
    def g(M, S, me, l, a0, a1, a2):
        M2, ctrl = f(M, S, me, l, a0, a1, a2)
        return M2, jnp.asarray(ctrl, jnp.int32)
    return g


BRANCHES = [wrap(f) for f in OPS]


# ---------------------------------------------------------------- environnement

def draw(M, S, p, n): return move_first(M, deck_of(M, S, p), n, Z_HAND)


def reset(key):
    k = jax.random.split(key, 8)
    # S : définitions des cartes, statiques pendant la partie (cartes procédurales)
    S = dict(owner=jnp.arange(N) // (N // 2),
             arch=jax.random.randint(k[4], (N,), 0, 8),
             level=jax.random.randint(k[3], (N,), 1, 13),
             nfx=jax.random.randint(k[5], (N,), 1, E + 1),
             ops=jax.random.randint(k[6], (N, E, L), 1, NOPS),
             args=jax.random.randint(k[7], (N, E, L, 3), 0, 8))
    # M : état mutable touché par les opcodes
    M = dict(zone=jnp.full(N, Z_DECK, jnp.int32),
             order=jax.random.permutation(k[0], N).astype(jnp.int32),
             atk=jax.random.randint(k[1], (N,), 0, 31) * 100,
             **{"def": jax.random.randint(k[2], (N,), 0, 31) * 100},
             flags=jnp.zeros(N, jnp.int32), counters=jnp.zeros(N, jnp.int32),
             lp=jnp.array([8000, 8000], jnp.int32), chain_neg=jnp.zeros(CH, bool))
    M = draw(draw(M, S, 0, 5), S, 1, 5)
    T = dict(player=jnp.int32(0), tstep=jnp.int32(0), turns=jnp.int32(0), opt=jnp.zeros((N, E), bool))
    return dict(M=M, S=S, T=T)


def legal(M, S, opt, p):
    z = M["zone"]
    ok_card = (S["owner"] == p) & ((z == Z_HAND) | (z == Z_MZ) | (z == Z_SZ))
    ok = ok_card[:, None] & (jnp.arange(E)[None, :] < S["nfx"][:, None]) & ~opt
    op0, a = S["ops"][:, :, 0], S["args"][:, :, 0]
    zsel, osel = a[..., 0] % 6, a[..., 1] % 3
    own = S["owner"][None, None, :]
    rel = jnp.where((osel == 0)[..., None], own == p, jnp.where((osel == 1)[..., None], own != p, True))
    cnt = jnp.sum((z[None, None, :] == zsel[..., None]) & rel, axis=-1)
    cond = jnp.where(op0 == OP_COND, cnt >= a[..., 2] % 3, True)
    m = (ok & cond).reshape(-1)
    return jnp.concatenate([m, jnp.array([True])])  # passer toujours légal


def step(env, action, key):
    M, S, T = env["M"], env["S"], env["T"]
    p = T["player"]
    passed = action == N * E
    c0, e0 = action // E, action % E
    ch_c = jnp.zeros(CH, jnp.int32).at[0].set(c0)
    ch_e = jnp.zeros(CH, jnp.int32).at[0].set(e0)
    ch_p = jnp.zeros(CH, jnp.int32).at[0].set(p)
    depth = jnp.where(passed, 0, 1)
    opt = T["opt"].at[c0, e0].set(T["opt"][c0, e0] | ~passed)

    # réponses en chaîne (joueurs alternés, politique aléatoire interne)
    for j in range(1, CH):
        q = jnp.where(j % 2 == 1, 1 - p, p)
        mask = legal(M, S, opt, q)[:-1]
        kj = jax.random.fold_in(key, j)
        want = jax.random.bernoulli(kj, 0.5) & mask.any() & (depth == j)
        pick = jax.random.categorical(jax.random.fold_in(kj, 1), jnp.where(mask, 0.0, -1e9))
        ch_c = ch_c.at[j].set(jnp.where(want, pick // E, ch_c[j]))
        ch_e = ch_e.at[j].set(jnp.where(want, pick % E, ch_e[j]))
        ch_p = ch_p.at[j].set(q)
        opt = opt.at[pick // E, pick % E].set(opt[pick // E, pick % E] | want)
        depth = depth + want.astype(jnp.int32)

    # résolution de la chaîne (dernier maillon d'abord), CH*L cycles d'interpréteur
    M = {**M, "chain_neg": jnp.zeros(CH, bool)}

    def cycle(_, carry):
        M, link, pc = carry
        active = link >= 0
        li = jnp.maximum(link, 0)
        c, e = ch_c[li], ch_e[li]
        op = jnp.where(active & ~M["chain_neg"][li], S["ops"][c, e, pc], 0)
        a = S["args"][c, e, pc]
        M2, ctrl = lax.switch(op, BRANCHES, M, S, ch_p[li], li, a[0], a[1], a[2])
        npc = pc + 1 + (ctrl == K).astype(jnp.int32)
        fin = (ctrl == A) | (npc >= L)
        return M2, jnp.where(active & fin, link - 1, link), jnp.where(fin, 0, npc)

    M, _, _ = lax.fori_loop(0, CH * L, cycle, (M, depth - 1, jnp.int32(0)))

    # tours
    tstep = T["tstep"] + 1
    end_turn = passed | (tstep >= 8)
    np_ = jnp.where(end_turn, 1 - p, p)
    M = jax.tree.map(lambda a, b: jnp.where(end_turn, a, b), draw(M, S, np_, 1), M)
    T = dict(player=np_, tstep=jnp.where(end_turn, 0, tstep),
             turns=T["turns"] + end_turn.astype(jnp.int32),
             opt=jnp.where(end_turn, jnp.zeros_like(opt), opt))

    done = (M["lp"] <= 0).any() | ~deck_of(M, S, np_).any() | (T["turns"] >= MAX_TURNS)
    new = reset(jax.random.fold_in(key, 999))
    env = jax.tree.map(lambda a, b: jnp.where(done, a, b), new, dict(M=M, S=S, T=T))
    return env, done


# ---------------------------------------------------------------- politique transformer

D, H, NL = 256, 4, 4
FEAT = 6 + 1 + 3 + 8 + 5 + 1 + E


def init_params(key):
    ks = iter(jax.random.split(key, 64))

    def w(shape, scale=None):
        return jax.random.normal(next(ks), shape) * (scale if scale else shape[0] ** -0.5)

    return dict(feat=w((FEAT, D)), prog=w((NOPS + 24, D)), out=w((D, E)), pass_=w((D, 1)),
                layers=[dict(qkv=w((D, 3 * D)), o=w((D, D)), f1=w((D, 4 * D)), f2=w((4 * D, D)))
                        for _ in range(NL)])


DT = jnp.bfloat16 if "--bf16" in sys.argv else jnp.float32


def ln(x):
    x32 = x.astype(jnp.float32)
    return ((x32 - x32.mean(-1, keepdims=True)) * lax.rsqrt(x32.var(-1, keepdims=True) + 1e-5)).astype(DT)


def policy_logits(params, env):
    M, S, T = env["M"], env["S"], env["T"]
    p = T["player"]
    f = jnp.concatenate([
        jax.nn.one_hot(M["zone"], 6), (S["owner"] == p)[:, None].astype(jnp.float32),
        M["atk"][:, None] / 3000, M["def"][:, None] / 3000, S["level"][:, None] / 12,
        jax.nn.one_hot(S["arch"], 8), ((M["flags"][:, None] >> jnp.arange(5)) & 1).astype(jnp.float32),
        M["counters"][:, None] / 5, T["opt"].astype(jnp.float32)], axis=-1)
    # encodeur d'effets léger : sac d'opcodes + sac d'arguments
    bag_ops = jax.nn.one_hot(S["ops"], NOPS).sum((1, 2))
    bag_args = jax.nn.one_hot(S["args"], 8).sum((1, 2)).reshape(N, 24)
    params = jax.tree.map(lambda w: w.astype(DT), params)
    bags = jnp.concatenate([bag_ops, bag_args], -1) / (E * L)
    x = f.astype(DT) @ params["feat"] + bags.astype(DT) @ params["prog"]
    for lp in params["layers"]:
        h = ln(x)
        q, k, v = jnp.split(h @ lp["qkv"], 3, -1)
        q, k, v = (t.reshape(N, H, D // H).transpose(1, 0, 2) for t in (q, k, v))
        att = jax.nn.softmax((q @ k.transpose(0, 2, 1)).astype(jnp.float32) / (D // H) ** 0.5, -1).astype(DT)
        x = x + (att @ v).transpose(1, 0, 2).reshape(N, D) @ lp["o"]
        x = x + jax.nn.gelu(ln(x) @ lp["f1"]) @ lp["f2"]
    x = ln(x)
    return jnp.concatenate([(x @ params["out"]).reshape(-1), x.mean(0) @ params["pass_"]]).astype(jnp.float32)


# ---------------------------------------------------------------- benchmark

def make_rollout(B, T_STEPS, use_nn):
    def one_step(carry, _):
        envs, key, params = carry
        key, ka, ks = jax.random.split(key, 3)
        masks = jax.vmap(lambda e: legal(e["M"], e["S"], e["T"]["opt"], e["T"]["player"]))(envs)
        if use_nn:
            logits = jax.vmap(policy_logits, (None, 0))(params, envs)
        else:
            logits = jnp.zeros((B, NA))
        actions = jax.random.categorical(ka, jnp.where(masks, logits, -1e9))
        envs, done = jax.vmap(step)(envs, actions, jax.random.split(ks, B))
        return (envs, key, params), done.sum()

    @jax.jit
    def rollout(envs, key, params):
        (envs, key, _), dones = lax.scan(one_step, (envs, key, params), None, length=T_STEPS)
        return envs, dones.sum()

    return rollout


def bench(B, use_nn, T_STEPS=50, reps=3):
    key = jax.random.PRNGKey(0)
    envs = jax.jit(jax.vmap(reset))(jax.random.split(key, B))
    params = init_params(jax.random.PRNGKey(1))
    rollout = make_rollout(B, T_STEPS, use_nn)
    print(f"  ... compilation {'NN' if use_nn else 'random'} B={B}", flush=True)
    t0 = time.perf_counter()
    envs, d = rollout(envs, key, params)
    d.block_until_ready()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    games = 0
    for r in range(reps):
        envs, d = rollout(envs, jax.random.fold_in(key, r), params)
        games += int(d)
    dt = time.perf_counter() - t0
    sps = B * T_STEPS * reps / dt
    print(f"{'NN' if use_nn else 'random':6s} B={B:6d}  {sps:12,.0f} coups/s   "
          f"(compile {compile_s:5.1f}s, {games} parties finies)", flush=True)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print("devices:", jax.devices(), "| jax", jax.__version__, "| dtype NN", DT.__name__, flush=True)
    sizes_env = [64, 256] if quick else [4096, 16384]
    sizes_nn = [64] if quick else [512, 1024, 2048]
    for B in sizes_env:
        try:
            bench(B, False, T_STEPS=10 if quick else 50)
        except Exception as ex:  # OOM etc.
            print(f"random B={B}: échec ({type(ex).__name__})", flush=True)
    for B in sizes_nn:
        try:
            bench(B, True, T_STEPS=10 if quick else 50)
        except Exception as ex:
            print(f"NN     B={B}: échec ({type(ex).__name__})", flush=True)
