"""Les compteurs et interdictions par nom ne doivent jamais oublier une entrée active."""
from pathlib import Path
import sys, unittest
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine')]
import core,cards

class NameCapacityTests(unittest.TestCase):
    def test_five_distinct_once_per_turn_effects_remain_consumed(self):
        codes=np.arange(1,7)
        bank=core.make_bank(codes,[500]*6,[500]*6,[4]*6,
            **cards.assemble(codes,cards={int(c):[cards.ignition(opt=core.OPT_NAME)] for c in codes},
                             ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.arange(40,dtype=jnp.int32)%6
        state=jax.jit(core.new_game)(bank,deck,deck)
        consume=jax.jit(core.use_count)
        for h in range(5):state=consume(bank,state,jnp.int32(h),jnp.int32(0),jnp.int32(0))
        for h in range(5):
            self.assertFalse(bool(core.act_ok(bank,state,jnp.int32(h),jnp.int32(0),jnp.int32(0))),h)
        self.assertTrue(bool(core.act_ok(bank,state,jnp.int32(5),jnp.int32(0),jnp.int32(0))))

    def test_duplicate_names_do_not_consume_capacity(self):
        table=jnp.zeros(5,jnp.int32)
        add=jax.jit(core.name_add)
        for key in (0,0,1,0,2,3,0,4,4):table=add(table,jnp.int32(key))
        np.testing.assert_array_equal(np.sort(np.asarray(table)),[1,2,3,4,5])

    def test_activation_candidates_beyond_the_first_block_are_tested(self):
        bank=core.make_bank(np.array([1]),[0],[0],[0],ctype=[core.TYPE_SPELL],
            **cards.assemble([1],cards={1:[cards.act()]},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.zeros(40,jnp.int32)
        state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(zone=jnp.where(jnp.arange(80)<35,core.Z_HAND,core.Z_DECK),
            dec=jnp.int32(core.D_IDLE),dec_player=jnp.int32(0),phase=jnp.int32(core.PH_MAIN1))
        mask=np.asarray(jax.jit(core.legal)(bank,state))
        activations=mask[core.V_ACT0*80:(core.V_ACT0+1)*80]
        self.assertEqual(int(activations.sum()),35)
        self.assertTrue(activations[:35].all())

    def test_arena_bound_is_checked_before_allocating_state(self):
        bank=core.make_bank(np.array([1]),[500],[500],[4])
        deck=jnp.zeros(core.MAX_SLOTS//2+1,jnp.int32)
        with self.assertRaisesRegex(ValueError,'arène'):
            core.new_game(bank,deck,deck)

    def test_trigger_after_a_full_block_of_locked_names_is_offered(self):
        codes=np.arange(1,27)
        bank=core.make_bank(codes,[500]*26,[500]*26,[4]*26,
            **cards.assemble(codes,cards={int(c):[cards.trigger(core.EV_SUMMON,optional=True)] for c in codes},
                             ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.arange(40,dtype=jnp.int32)%26
        state=jax.jit(core.new_game)(bank,deck,deck)
        pending=jnp.arange(80)<26
        locks=state.lock_cid.at[0,:24].set(jnp.arange(1,25))
        state=state._replace(tg=pending,dec=jnp.int32(core.D_TRIG),lock_cid=locks)
        result=jax.jit(core.trig_eval)(bank,state)
        self.assertEqual(int(result.dec),core.D_CHAIN)
        self.assertEqual(int(result.wk),core.W_TRIG)
        self.assertEqual(int(result.tst),2)

if __name__=='__main__':unittest.main()
