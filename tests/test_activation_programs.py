"""Le programme dérivé conserve exactement les conditions du bytecode original."""
from pathlib import Path
import sys, unittest
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'gen')]
import core, cards, cardgen


def reference(bank, state, effect):
    instructions = bank.prog[state.cid[0], effect, :core.PG_RESOLVE, :core.PT].reshape(-1, 4)
    def step(carry, instruction):
        registers, ok = carry
        return core._pure(bank, state, registers, ok, jnp.int32(0), effect,
                          jnp.int32(0), state.n_chain, instruction), None
    (_, ok), _ = lax.scan(step, (jnp.zeros_like(state.reg), jnp.bool_(True)), instructions)
    return ok


class ActivationProgramTests(unittest.TestCase):
    def test_compaction_preserves_bounds_aliases_and_program_boundaries(self):
        f = cards.F(me=core.Z_HAND)
        cases = [
            ([cards.FILTER(0, f), cards.REQ(0, 2), cards.REQ(0, 4, True)],
             [cards.CHOOSE(0, 0, 3, mn=1)]),
            ([cards.FILTER(0, f)], [cards.REQ(0, 1)]),
            ([cards.FILTER(0, f), cards.REQ(0, 5), cards.REQ(0, 2, True)], []),
            ([cards.FILTER(0, f), cards.REQ(0, 0, True), cards.CHOOSE(0, 0, 3, upto=True)], []),
            ([cards.FILTER(0, f), cards.CHOOSE(1, 0, 1), cards.REQ(1, 2)], []),
            ([cards.FILTER(0, f), cards.REQ(0, 256), cards.REQ(0, 255, True)], []),
        ]
        codes = np.arange(1, len(cases)+1)
        bank = core.make_bank(codes, [1000]*len(codes), [1000]*len(codes), [4]*len(codes),
            **cards.assemble(codes, cards={c: [cards.act(cost=cost, target=target)]
                for c, (cost, target) in zip(codes, cases)}, ritual={}, fusion={}, synchro={}, xyz={}, link={}))
        original = np.asarray(bank.prog).copy()
        state = jax.jit(core.new_game)(bank, jnp.zeros(40, jnp.int32), jnp.zeros(40, jnp.int32))
        expected = jax.jit(reference)
        actual = jax.jit(lambda bank, state: core.test_ok(bank, state, jnp.int32(0), jnp.int32(0), jnp.int32(0)))
        for card in range(len(codes)):
            for count in range(11):
                zones = jnp.where(jnp.arange(80) < count, core.Z_HAND, core.Z_GRAVE)
                trial = state._replace(cid=state.cid.at[0].set(card), zone=zones,
                                       ctrl=jnp.zeros(80, jnp.int32), dec_player=jnp.int32(0))
                self.assertEqual(bool(actual(bank, trial)), bool(expected(bank, trial, jnp.int32(0))), (card, count))
        np.testing.assert_array_equal(bank.prog, original)

    def test_generated_banks_keep_fixed_shapes(self):
        banks = [cardgen.to_bank(cardgen.Gen(seed).make_set(80, 10)) for seed in (0, 1)]
        self.assertEqual(jax.tree.map(lambda a: a.shape, banks[0]), jax.tree.map(lambda a: a.shape, banks[1]))
        self.assertEqual(banks[0].tprog.shape[2], 2*core.PT)


if __name__ == '__main__':
    unittest.main()
