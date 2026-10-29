"""Test de faisabilité v2 : même jeu que interp_bench.py, version optimisée.

Optimisations (toutes transposables au vrai moteur) :
- état en int8/int16 au lieu d'int32
- plus de lax.switch qui renvoie l'état complet : chaque opcode = (sélecteur, action) décrit par des
  tables ; le sélecteur (zone, propriétaire, filtre, k premières) est calculé UNE fois par cycle, puis
  chaque champ est mis à jour par une sélection élément par élément sur l'opcode
- conditions du masque légal via une table de comptes (zone, propriétaire) calculée une fois par coup
- encodage des effets (sac d'opcodes) précalculé au reset, le réseau ne le recalcule pas
- attention via jax.nn.dot_product_attention, réseau en bf16
Usage : python interp_bench_v2.py [--quick]
"""
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

N, E, L, CH, NOPS, MZ = 80, 4, 8, 4, 32, 5
Z_DECK, Z_HAND, Z_MZ, Z_SZ, Z_GY, Z_BAN = 0, 1, 2, 3, 4, 5
NA = N * E + 1
MAX_TURNS = 200
I8, I16 = jnp.int8, jnp.int16
DT = jnp.bfloat16
C, A, K = 0, 1, 2

# ---------------------------------------------------------------- tables d'opcodes
# 0 nop 1 draw 2 damage 3 recover 4 destroy 5 banish 6 bounce 7 spsummon 8 atk_up 9 atk_down
# 10 set_flag 11 clear_flag 12 search 13 mill 14 negate 15 cond_count 16 cond_lp 17 discard
# 18 tribute 19 swap 20 double 21 set_atk 22 gain_per 23 dmg_per 24 to_deck 25 shuffle 26 copy
# 27 add_counter 28 remove_counter 29 change_pos 30 skip_if 31 end
OP_COND = 15
# zone source : -1 -> a0 % 6, -2 -> main (a0 pair) ou cimetière
SRC = jnp.array([2, 0, 2, 2, 2, -1, 2, -2, 2, 2, 2, 2, 0, 0, 2, -1,
                 2, 1, 2, 2, 2, 2, -1, -1, -1, 2, 2, 2, 2, 2, -1, 2], I8)
# propriétaire : 0 moi, 1 adversaire, 3 -> a1 % 3, 4 -> moi/adversaire selon a1 pair
OSEL = jnp.array([3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 4, 3, 3,
                  3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3], I8)
USE_THR = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], bool)
# combien de cartes : 0 toutes, 1 a0%3+1, 2 une, 3 a0%2+1, 4 une si place en zone monstre
KMODE = jnp.array([0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 1, 0, 0,
                   0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], I8)
DEST = jnp.array([-1, 1, -1, -1, 4, 5, 1, 2, -1, -1, -1, -1, 1, 4, -1, -1,
                  -1, 4, 4, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1], I8)


def rank(mask, order):
    return jnp.sum(mask[None, :] & (order[None, :] < order[:, None]), axis=1)


def count(m):
    return jnp.sum(m.astype(jnp.int32))


def exec_op(M, S, op, a0, a1, a2, me, li):
    """Un cycle d'interpréteur, sans branchement : sélecteur commun puis mise à jour champ par champ."""
    a0, a1, a2 = a0.astype(jnp.int32), a1.astype(jnp.int32), a2.astype(jnp.int32)
    zone, owner = M["zone"], S["owner"]
    atk, df = M["atk"].astype(jnp.int32), M["def"].astype(jnp.int32)
    fl, ctr = M["flags"].astype(jnp.int32), M["counters"].astype(jnp.int32)

    # --- sélecteur
    src = SRC[op]
    zs = jnp.where(src == -1, a0 % 6, jnp.where(src == -2, jnp.where(a0 % 2 == 0, Z_HAND, Z_GY), src))
    oc = OSEL[op]
    osel = jnp.where(oc == 3, a1 % 3, jnp.where(oc == 4, a1 % 2, oc))
    thr = jnp.where(USE_THR[op], a2, 0)
    rel = jnp.where(osel == 0, owner == me, jnp.where(osel == 1, owner != me, True))
    m = (zone == zs) & rel & jnp.where(thr == 0, True, atk <= thr * 500)
    m = m & jnp.where(op == 12, S["arch"] == a0, True) & jnp.where(op == 4, (fl & 1) == 0, True)
    cnt = count(m)
    free = MZ - count((zone == Z_MZ) & (owner == me))
    km = KMODE[op]
    k = jnp.select([km == 1, km == 2, km == 3, km == 4], [a0 % 3 + 1, 1, a0 % 2 + 1, jnp.minimum(1, free)], N)
    sel = m & (rank(m, M["order"]) < k)

    # --- actions
    dest = DEST[op]
    zone2 = jnp.where(sel & (dest >= 0), dest, zone).astype(I8)

    v = (a0 + 1) * 100
    best = jnp.max(jnp.where(sel, atk, 0))
    new_atk = jnp.select([op == 8, op == 9, op == 20, op == 21, op == 26, op == 19],
                         [jnp.minimum(atk + v, 10000), jnp.maximum(atk - v, 0), jnp.minimum(atk * 2, 10000),
                          jnp.broadcast_to(a0 * 500, atk.shape), jnp.broadcast_to(best, atk.shape), df], atk)
    atk2 = jnp.where(sel, new_atk, atk).astype(I16)
    def2 = jnp.where(sel & (op == 19), atk, df).astype(I16)

    bit = 1 << (a0 % 4)
    new_fl = jnp.select([op == 10, op == 11, op == 29], [fl | bit, fl & ~bit, fl ^ 16], fl)
    fl2 = jnp.where(sel, new_fl, fl).astype(I8)

    ok_ctr = jnp.sum(jnp.where(m, ctr, 0)) >= a0
    ctr2 = jnp.where(sel & (op == 27), ctr + a0 + 1, ctr)
    ctr2 = jnp.where(sel & (op == 28) & ok_ctr, jnp.maximum(0, ctr - a0), ctr2).astype(I16)

    order = M["order"].astype(jnp.int32)
    order2 = jnp.where(op == 25, (order * 7 + a0 * 13 + 1) % 997,
                       jnp.where(sel & (op == 24), (order + 1000) % 30000, order)).astype(I16)

    d_me = jnp.select([op == 3, op == 22], [v, 100 * cnt], 0)
    d_opp = jnp.select([op == 2, op == 23], [-v, -100 * cnt], 0)
    lp2 = M["lp"].at[me].add(d_me).at[1 - me].add(d_opp)

    neg = M["chain_neg"]
    neg2 = jnp.where((op == 14) & (li > 0), neg.at[jnp.maximum(li - 1, 0)].set(True), neg)

    ctrl = jnp.select([(op == 15) & (cnt < a2 % 3), (op == 16) & (M["lp"][me] < a0 * 1000),
                       (op == 18) & (cnt == 0), (op == 28) & ~ok_ctr, (op == 30) & (cnt == 0), op == 31],
                      [A, A, A, A, K, A], C)
    M2 = dict(zone=zone2, order=order2, atk=atk2, flags=fl2, counters=ctr2, lp=lp2, chain_neg=neg2,
              **{"def": def2})
    return M2, ctrl


# ---------------------------------------------------------------- environnement

def draw(M, S, p, n):
    m = (M["zone"] == Z_DECK) & (S["owner"] == p)
    return {**M, "zone": jnp.where(m & (rank(m, M["order"]) < n), Z_HAND, M["zone"]).astype(I8)}


def reset(key):
    k = jax.random.split(key, 8)
    ops = jax.random.randint(k[6], (N, E, L), 1, NOPS).astype(I8)
    args = jax.random.randint(k[7], (N, E, L, 3), 0, 8).astype(I8)
    bag = jnp.concatenate([jax.nn.one_hot(ops, NOPS).sum((1, 2)),
                           jax.nn.one_hot(args, 8).sum((1, 2)).reshape(N, 24)], -1) / (E * L)
    S = dict(owner=(jnp.arange(N) // (N // 2)).astype(I8),
             arch=jax.random.randint(k[4], (N,), 0, 8).astype(I8),
             level=jax.random.randint(k[3], (N,), 1, 13).astype(I8),
             nfx=jax.random.randint(k[5], (N,), 1, E + 1).astype(I8),
             ops=ops, args=args, bag=bag.astype(DT))
    M = dict(zone=jnp.full(N, Z_DECK, I8),
             order=jax.random.permutation(k[0], N).astype(I16),
             atk=(jax.random.randint(k[1], (N,), 0, 31) * 100).astype(I16),
             **{"def": (jax.random.randint(k[2], (N,), 0, 31) * 100).astype(I16)},
             flags=jnp.zeros(N, I8), counters=jnp.zeros(N, I16),
             lp=jnp.array([8000, 8000], jnp.int32), chain_neg=jnp.zeros(CH, bool))
    M = draw(draw(M, S, 0, 5), S, 1, 5)
    T = dict(player=jnp.int32(0), tstep=jnp.int32(0), turns=jnp.int32(0), opt=jnp.zeros((N, E), bool))
    return dict(M=M, S=S, T=T)


def zone_counts(M, S):
    """Table [zone, propriétaire] des comptes, calculée une fois par coup."""
    return (jax.nn.one_hot(M["zone"], 6, dtype=jnp.int32)[:, :, None]
            * jax.nn.one_hot(S["owner"], 2, dtype=jnp.int32)[:, None, :]).sum(0)


def legal(M, S, opt, p, zc):
    z = M["zone"]
    ok_card = (S["owner"] == p) & ((z == Z_HAND) | (z == Z_MZ) | (z == Z_SZ))
    ok = ok_card[:, None] & (jnp.arange(E)[None, :] < S["nfx"][:, None]) & ~opt
    op0, a = S["ops"][:, :, 0], S["args"][:, :, 0].astype(jnp.int32)
    zsel, osel = a[..., 0] % 6, a[..., 1] % 3
    cnt = jnp.where(osel == 0, zc[zsel, p], jnp.where(osel == 1, zc[zsel, 1 - p], zc[zsel, 0] + zc[zsel, 1]))
    cond = jnp.where(op0 == OP_COND, cnt >= a[..., 2] % 3, True)
    return jnp.concatenate([(ok & cond).reshape(-1), jnp.array([True])])


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
    zc = zone_counts(M, S)

    for j in range(1, CH):
        q = jnp.where(j % 2 == 1, 1 - p, p)
        mask = legal(M, S, opt, q, zc)[:-1]
        kj = jax.random.fold_in(key, j)
        want = jax.random.bernoulli(kj, 0.5) & mask.any() & (depth == j)
        pick = jax.random.categorical(jax.random.fold_in(kj, 1), jnp.where(mask, 0.0, -1e9))
        ch_c = ch_c.at[j].set(jnp.where(want, pick // E, ch_c[j]))
        ch_e = ch_e.at[j].set(jnp.where(want, pick % E, ch_e[j]))
        ch_p = ch_p.at[j].set(q)
        opt = opt.at[pick // E, pick % E].set(opt[pick // E, pick % E] | want)
        depth = depth + want.astype(jnp.int32)

    M = {**M, "chain_neg": jnp.zeros(CH, bool)}

    def cycle(_, carry):
        M, link, pc = carry
        active = link >= 0
        li = jnp.maximum(link, 0)
        c, e = ch_c[li], ch_e[li]
        op = jnp.where(active & ~M["chain_neg"][li], S["ops"][c, e, pc].astype(jnp.int32), 0)
        a = S["args"][c, e, pc]
        M2, ctrl = exec_op(M, S, op, a[0], a[1], a[2], ch_p[li], li)
        npc = pc + 1 + (ctrl == K).astype(jnp.int32)
        fin = (ctrl == A) | (npc >= L)
        return M2, jnp.where(active & fin, link - 1, link), jnp.where(fin, 0, npc)

    M, _, _ = lax.fori_loop(0, CH * L, cycle, (M, depth - 1, jnp.int32(0)))

    tstep = T["tstep"] + 1
    end_turn = passed | (tstep >= 8)
    np_ = jnp.where(end_turn, 1 - p, p)
    M = jax.tree.map(lambda a, b: jnp.where(end_turn, a, b), draw(M, S, np_, 1), M)
    T = dict(player=np_, tstep=jnp.where(end_turn, 0, tstep),
             turns=T["turns"] + end_turn.astype(jnp.int32),
             opt=jnp.where(end_turn, jnp.zeros_like(opt), opt))

    deck_left = ((M["zone"] == Z_DECK) & (S["owner"] == np_)).any()
    done = (M["lp"] <= 0).any() | ~deck_left | (T["turns"] >= MAX_TURNS)
    new = reset(jax.random.fold_in(key, 999))
    env = jax.tree.map(lambda a, b: jnp.where(done, a, b), new, dict(M=M, S=S, T=T))
    return env, done


# ---------------------------------------------------------------- politique transformer

D, H, NL = 256, 4, 4
FEAT = 6 + 1 + 3 + 8 + 5 + 1 + E


def init_params(key):
    ks = iter(jax.random.split(key, 64))

    def w(shape, scale=None):
        return (jax.random.normal(next(ks), shape) * (scale if scale else shape[0] ** -0.5)).astype(DT)

    return dict(feat=w((FEAT, D)), prog=w((NOPS + 24, D)), out=w((D, E)), pass_=w((D, 1)),
                layers=[dict(qkv=w((D, 3 * D)), o=w((D, D)), f1=w((D, 4 * D)), f2=w((4 * D, D)))
                        for _ in range(NL)])


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
    x = f.astype(DT) @ params["feat"] + S["bag"] @ params["prog"]
    for lp in params["layers"]:
        q, k, v = jnp.split(ln(x) @ lp["qkv"], 3, -1)
        q, k, v = (t.reshape(N, H, D // H) for t in (q, k, v))
        x = x + jax.nn.dot_product_attention(q, k, v).reshape(N, D) @ lp["o"]
        x = x + jax.nn.gelu(ln(x) @ lp["f1"]) @ lp["f2"]
    x = ln(x)
    return jnp.concatenate([(x @ params["out"]).reshape(-1), x.mean(0) @ params["pass_"]]).astype(jnp.float32)


# ---------------------------------------------------------------- benchmark

def make_rollout(B, T_STEPS, use_nn):
    def one_step(carry, _):
        envs, key, params = carry
        key, ka, ks = jax.random.split(key, 3)

        def mask_of(e):
            return legal(e["M"], e["S"], e["T"]["opt"], e["T"]["player"], zone_counts(e["M"], e["S"]))

        masks = jax.vmap(mask_of)(envs)
        logits = jax.vmap(policy_logits, (None, 0))(params, envs) if use_nn else jnp.zeros((B, NA))
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
    print(f"{'NN' if use_nn else 'random':6s} B={B:6d}  {B * T_STEPS * reps / dt:12,.0f} coups/s   "
          f"(compile {compile_s:5.1f}s, {games} parties finies)", flush=True)



# ================================================================ ablation (collé après interp_bench_v2.py sans son main)
# Usage : python ablation.py [--quick]   (ablation.py = v2 sans main + ce fichier, construit par build_ablation.sh)
# Questions : (1) le réseau seul est-il limité par le calcul ou par la mémoire / l'overhead ?
#             (2) dans le moteur, qu'est-ce qui coûte : les 32 cycles, le tri O(N²), le reset à chaque coup ?

CFG = dict(ncyc=CH * L, fast_rank=False, reset=True)
_rank_exact = rank


def rank(mask, order):  # noqa: F811 — remplace la version de v2 au moment du traçage
    if CFG["fast_rank"]:
        return jnp.cumsum(mask.astype(jnp.int32)) - 1  # O(N), ordre d'index au lieu de l'ordre du deck
    return _rank_exact(mask, order)


_step_v2 = step


def step(env, action, key):  # noqa: F811 — copie de v2 avec nombre de cycles et reset réglables
    M, S, T = env["M"], env["S"], env["T"]
    p = T["player"]
    passed = action == N * E
    c0, e0 = action // E, action % E
    ch_c = jnp.zeros(CH, jnp.int32).at[0].set(c0)
    ch_e = jnp.zeros(CH, jnp.int32).at[0].set(e0)
    ch_p = jnp.zeros(CH, jnp.int32).at[0].set(p)
    depth = jnp.where(passed, 0, 1)
    opt = T["opt"].at[c0, e0].set(T["opt"][c0, e0] | ~passed)
    zc = zone_counts(M, S)
    for j in range(1, CH):
        q = jnp.where(j % 2 == 1, 1 - p, p)
        mask = legal(M, S, opt, q, zc)[:-1]
        kj = jax.random.fold_in(key, j)
        want = jax.random.bernoulli(kj, 0.5) & mask.any() & (depth == j)
        pick = jax.random.categorical(jax.random.fold_in(kj, 1), jnp.where(mask, 0.0, -1e9))
        ch_c = ch_c.at[j].set(jnp.where(want, pick // E, ch_c[j]))
        ch_e = ch_e.at[j].set(jnp.where(want, pick % E, ch_e[j]))
        ch_p = ch_p.at[j].set(q)
        opt = opt.at[pick // E, pick % E].set(opt[pick // E, pick % E] | want)
        depth = depth + want.astype(jnp.int32)
    M = {**M, "chain_neg": jnp.zeros(CH, bool)}

    def cycle(_, carry):
        M, link, pc = carry
        active = link >= 0
        li = jnp.maximum(link, 0)
        c, e = ch_c[li], ch_e[li]
        op = jnp.where(active & ~M["chain_neg"][li], S["ops"][c, e, pc].astype(jnp.int32), 0)
        a = S["args"][c, e, pc]
        M2, ctrl = exec_op(M, S, op, a[0], a[1], a[2], ch_p[li], li)
        npc = pc + 1 + (ctrl == K).astype(jnp.int32)
        fin = (ctrl == A) | (npc >= L)
        return M2, jnp.where(active & fin, link - 1, link), jnp.where(fin, 0, npc)

    M, _, _ = lax.fori_loop(0, CFG["ncyc"], cycle, (M, depth - 1, jnp.int32(0)))
    tstep = T["tstep"] + 1
    end_turn = passed | (tstep >= 8)
    np_ = jnp.where(end_turn, 1 - p, p)
    M = jax.tree.map(lambda a, b: jnp.where(end_turn, a, b), draw(M, S, np_, 1), M)
    T = dict(player=np_, tstep=jnp.where(end_turn, 0, tstep),
             turns=T["turns"] + end_turn.astype(jnp.int32),
             opt=jnp.where(end_turn, jnp.zeros_like(opt), opt))
    deck_left = ((M["zone"] == Z_DECK) & (S["owner"] == np_)).any()
    done = (M["lp"] <= 0).any() | ~deck_left | (T["turns"] >= MAX_TURNS)
    env = dict(M=M, S=S, T=T)
    if CFG["reset"]:
        new = reset(jax.random.fold_in(key, 999))
        env = jax.tree.map(lambda a, b: jnp.where(done, a, b), new, env)
    return env, done


def timed(fn, args, n_items, label, flops_per_item=None, reps=3):
    print(f"  ... compilation {label}", flush=True)
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    c = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    rate = n_items * reps / (time.perf_counter() - t0)
    tf = f"  {rate * flops_per_item / 1e12:5.1f} TFLOPS" if flops_per_item else ""
    print(f"{label:42s} {rate:12,.0f} /s{tf}   (compile {c:4.1f}s)", flush=True)
    return rate


# ---------------------------------------------------------------- (1) réseau seul

def nn_real(B, T_STEPS):
    """Vraie politique v2 sur des états figés (pas de moteur). La dépendance via acc empêche XLA de sortir le calcul de la boucle."""
    envs = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(0), B))
    params = init_params(jax.random.PRNGKey(1))

    @jax.jit
    def run(envs, params):
        def body(acc, _):
            e = {**envs, "S": {**envs["S"], "bag": envs["S"]["bag"] + (acc * 1e-6).astype(DT)[:, None, None]}}
            lg = jax.vmap(policy_logits, (None, 0))(params, e)
            return acc + lg[:, 0], None
        return lax.scan(body, jnp.zeros(B), None, length=T_STEPS)[0]

    flops = 2 * NL * (N * 12 * D * D + 2 * N * N * D)
    return timed(run, (envs, params), B * T_STEPS, f"NN réel (états figés) B={B}", flops)


def nn_synth(B, T_STEPS, d, nl, ntok, attn=True, ffn=True):
    """Transformer pur sur des tokens aléatoires : même structure que policy_logits, tailles réglables."""
    ks = iter(jax.random.split(jax.random.PRNGKey(2), 4 * nl + 2))
    w = lambda s: (jax.random.normal(next(ks), s) * s[0] ** -0.5).astype(DT)
    params = [dict(qkv=w((d, 3 * d)), o=w((d, d)), f1=w((d, 4 * d)), f2=w((4 * d, d))) for _ in range(nl)]
    x0 = jax.random.normal(next(ks), (B, ntok, d)).astype(DT)
    h = max(1, d // 64)

    def fwd(params, x):
        for lp in params:
            if attn:
                q, k, v = jnp.split(ln(x) @ lp["qkv"], 3, -1)
                q, k, v = (t.reshape(ntok, h, d // h) for t in (q, k, v))
                x = x + jax.nn.dot_product_attention(q, k, v).reshape(ntok, d) @ lp["o"]
            if ffn:
                x = x + jax.nn.gelu(ln(x) @ lp["f1"]) @ lp["f2"]
        return ln(x).mean(0)[0].astype(jnp.float32)

    @jax.jit
    def run(params, x0):
        def body(acc, _):
            return acc + jax.vmap(fwd, (None, 0))(params, x0 + (acc * 1e-6).astype(DT)[:, None, None]), None
        return lax.scan(body, jnp.zeros(B), None, length=T_STEPS)[0]

    flops = 2 * nl * ntok * ((4 * d * d + 2 * ntok * d) * attn + 8 * d * d * ffn)
    tag = f"d={d} L={nl} tok={ntok}" + ("" if attn else " sans attn") + ("" if ffn else " sans FFN")
    return timed(run, (params, x0), B * T_STEPS, f"NN synth {tag} B={B}", flops)


# ---------------------------------------------------------------- (2) moteur seul

def engine(B, T_STEPS, label, **cfg):
    CFG.update(dict(ncyc=CH * L, fast_rank=False, reset=True), **cfg)
    envs = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(0), B))
    rollout = make_rollout(B, T_STEPS, False)
    return timed(lambda e, k: rollout(e, k, None)[1], (envs, jax.random.PRNGKey(3)), B * T_STEPS,
                 f"moteur {label} B={B}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print("ablation | devices:", jax.devices(), "| jax", jax.__version__, flush=True)
    T_ = 4 if quick else 20
    Bn = 32 if quick else 2048
    Be = 64 if quick else 16384
    jobs = [
        lambda: nn_real(Bn, T_),
        lambda: nn_synth(Bn, T_, 256, 4, 80),
        lambda: nn_synth(Bn, T_, 256, 4, 80, attn=False),
        lambda: nn_synth(Bn, T_, 256, 4, 80, ffn=False),
        lambda: nn_synth(Bn, T_, 128, 4, 80),
        lambda: nn_synth(Bn, T_, 512, 4, 80),
        lambda: nn_synth(Bn, T_, 256, 4, 32),
        lambda: nn_synth(4 * Bn, T_, 256, 4, 80),
        lambda: engine(Be, T_, "base (32 cycles)"),
        lambda: engine(Be, T_, "8 cycles", ncyc=8),
        lambda: engine(Be, T_, "rank O(N)", fast_rank=True),
        lambda: engine(Be, T_, "sans reset", reset=False),
        lambda: engine(Be, T_, "8 cycles + rank O(N) + sans reset", ncyc=8, fast_rank=True, reset=False),
    ]
    for j in jobs:
        try:
            j()
        except Exception as ex:
            print(f"échec ({type(ex).__name__}: {str(ex)[:200]})", flush=True)
    print("FIN", flush=True)
