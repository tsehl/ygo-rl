
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
