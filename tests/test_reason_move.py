from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
import core,cards,jax
import jax.numpy as jnp
import numpy as np

class ReasonMoveTests(unittest.TestCase):
    def state_and_bank(self, instruction, phase=core.PG_RESOLVE):
        bank=core.make_bank(np.array([1]),[500],[500],[4],
            **cards.assemble([1],cards={1:[cards.act(resolve=[instruction])]},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.zeros(40,jnp.int32);state=jax.jit(core.new_game)(bank,deck,deck)
        state=state._replace(dec=jnp.int32(core.D_RUN),ch_src=jnp.int32(0),ch_eff=jnp.int32(0),ch_p=jnp.int32(0),
            pg=jnp.int32(phase),pc=jnp.int32(0),zone=state.zone.at[1].set(core.Z_MZONE),
            reg=state.reg.at[0,1].set(True),reason=state.reason.at[2].set(0x123))
        if phase!=core.PG_RESOLVE:
            bank=bank._replace(prog=bank.prog.at[0,0,phase,0].set(jnp.array(instruction)))
        return bank,state
    def test_moves_keep_the_same_combinations_as_native_oracle(self):
        cases=[(cards.DESTROY(0,reason=core.REASON_EFFECT),core.PG_RESOLVE,65),
               (cards.MOVE(0,reason=core.REASON_EFFECT),core.PG_RESOLVE,64),
               (cards.MOVE(0,reason=core.REASON_COST|core.REASON_RELEASE),core.PG_COST,130),
               (cards.MOVE(0,reason=core.REASON_EFFECT),core.PG_COST,64),
               (cards.MOVE(0,reason=0),core.PG_RESOLVE,0)]
        run=jax.jit(core.exec_one)
        for instruction,phase,expected in cases:
            with self.subTest(instruction=instruction,phase=phase):
                bank,state=self.state_and_bank(instruction,phase)
                result=run(bank,state)
                self.assertEqual(int(result.reason[1]),expected)
                self.assertEqual(int(result.reason[2]),0x123)
                self.assertEqual(int(result.zone[1]),core.Z_GRAVE)
    def test_self_cost_reason_depends_on_source_zone(self):
        run=jax.jit(core.exec_one)
        instruction=cards.MOVE(0,reason=core.REASON_COST,release_if_field=True)
        for zone,expected in ((core.Z_HAND,core.REASON_COST),
                              (core.Z_MZONE,core.REASON_COST|core.REASON_RELEASE)):
            bank,state=self.state_and_bank(instruction,core.PG_COST)
            state=state._replace(zone=state.zone.at[0].set(zone),
                reg=state.reg.at[0].set(jnp.arange(80)==0))
            result=run(bank,state)
            self.assertEqual(int(result.reason[0]),expected)
if __name__=='__main__':unittest.main()
