"""Une attaque ouvre immédiatement le point des effets déclenchés."""
from pathlib import Path
import sys,unittest
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
import core,cards
class AttackTriggerWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        effect=cards.trigger(cards.ev(core.EV_ATTACK)|core.TRIG_WHEN,optional=True,rng=core.Z_MZONE)
        cls.bank=core.make_bank(np.array([1,2]),[1000,1000],[1000,1000],[4,4],
            **cards.assemble([1,2],cards={1:[effect]},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.ones(40,jnp.int32).at[0].set(0)
        s=jax.jit(core.new_game)(cls.bank,deck,deck)
        cls.state=s._replace(zone=jnp.full(80,core.Z_DECK,jnp.int32).at[0].set(core.Z_MZONE).at[1].set(core.Z_MZONE),
            pos=s.pos.at[0].set(core.P_FU_ATK).at[1].set(core.P_FU_ATK),
            seq=s.seq.at[0].set(0).at[1].set(1),dec=jnp.int32(core.D_BATTLE),dec_player=jnp.int32(0),
            tp=jnp.int32(0),phase=jnp.int32(core.PH_BATTLE))
        cls.apply=staticmethod(jax.jit(core.apply))
    def test_direct_attack_opens_trigger(self):
        s=self.apply(self.bank,self.state,jnp.int32(core.V_ATTACK*80+1))
        self.assertEqual(int(s.dec),core.D_TRIG)
        self.assertTrue(bool(s.tg[0]))
    def test_targeted_attack_opens_trigger_after_selection(self):
        s=self.state._replace(zone=self.state.zone.at[40].set(core.Z_MZONE),
                              pos=self.state.pos.at[40].set(core.P_FU_ATK),seq=self.state.seq.at[40].set(0))
        s=self.apply(self.bank,s,jnp.int32(core.V_ATTACK*80+1))
        self.assertEqual(int(s.dec),core.D_TARGET)
        s=self.apply(self.bank,s,jnp.int32(core.V_SELECT*80+40))
        self.assertEqual(int(s.dec),core.D_TRIG)
        self.assertTrue(bool(s.tg[0]))
if __name__=='__main__':unittest.main()
