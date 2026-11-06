"""Affectation exacte avec au plus un remplacement par substitut. Pas de matrice carte × carte.

Hall sur les sous-ensembles des FK emplacements. Après réservation d'un substitut,
les ensembles serrés (|voisinage| = |ensemble|) interdisent certains profils au
substitut réservé. Un histogramme des 2**FK profils permet de les compter en O(N).
"""
import jax
import jax.numpy as jnp


def completions(cand, substitutes, field, eligible, excluded):
    """Quatre tables [2**FK, M] : sans/avec substitut, sans/avec matériau du terrain.
    `excluded[M]` : candidat déjà réservé, -1 pour aucun. Les cartes déjà choisies
    doivent avoir été retirées de cand et substitutes par l'appelant.
    Coût O(FK² * 2**FK * M + FK * (N + FK * 2**FK)), mémoire O(FK² * 2**FK * M).
    FK=4 est une borne statique ; M=N pour les choix, M=1 pour la disponibilité.
    """
    fk, n = cand.shape
    q = jnp.arange(1 << fk, dtype=jnp.int32)
    slot = jnp.arange(fk)
    membership = (q[:, None] & (1 << slot)) != 0
    sizes = membership.sum(1)
    present = excluded >= 0
    ix = jnp.maximum(excluded, 0)
    allowed = (eligible & (1 << slot)) != 0

    def forced(l):
        candidates = cand & ((slot[:, None] != l) | field)
        bits = (candidates * (1 << slot[:, None])).sum(0)
        nq = 1 << fk
        # Histograms + transformée zêta : aucun tenseur [Q,Q,N], ni [N,N].
        weights = jnp.stack([jnp.ones(n, jnp.int32), substitutes.astype(jnp.int32),
                             (substitutes & field).astype(jnp.int32)])
        hist = jnp.zeros((3,nq),jnp.int32).at[:,bits].add(weights)
        for j in range(fk):
            hist = hist + jnp.where((q & (1 << j))[None, :] != 0, hist[:, q ^ (1 << j)], 0)
        neighbors_excluded = (bits[ix][None, :] & q[:,None]) != 0
        counts = (hist[0,-1] - hist[0,(~q) & (nq-1)])[:,None] - (neighbors_excluded & present)
        bad = counts < sizes[:,None]
        forbidden = jnp.where(counts == sizes[:,None], q[:,None], 0)
        for j in range(fk):
            use = (q & (1 << j))[:,None] != 0
            bad = bad | (use & bad[q ^ (1 << j)])
            forbidden = forbidden | jnp.where(use, forbidden[q ^ (1 << j)], 0)
        plain = ~bad

        def wildcard(j):
            rest = q & ~(1 << j)
            ok = plain[rest]
            blocked = forbidden[rest]
            stock = jnp.where(l == j, hist[2], hist[1])[(~blocked) & (nq-1)]
            taken = substitutes[ix] & present & ((bits[ix][None, :] & blocked) == 0) & ((l != j) | field[ix])
            return ok & (stock - taken > 0) & allowed[j] & ((q & (1 << j)) != 0)[:, None]
        enhanced = plain | jax.vmap(wildcard)(slot).any(0)
        return plain, enhanced

    plain, enhanced = forced(jnp.int32(-1))
    fp, fs = jax.vmap(forced)(slot)
    use_field = membership.T[:, :, None]
    return plain, enhanced, (fp & use_field).any(0), (fs & use_field).any(0)
