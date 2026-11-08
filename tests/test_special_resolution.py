from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
import core,cards,jax
import jax.numpy as jnp
import numpy as np
class SpecialResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank=core.make_bank(np.array([1]),[500],[500],[4],
            **cards.assemble([1],cards={1:[cards.act(resolve=[cards.SPSUMMON(0,core.P_FU_ATK)])]},
                             ritual={},fusion={},synchro={},xyz={},link={}))
        d=jnp.zeros(40,jnp.int32)
        s=jax.jit(core.new_game)(cls.bank,d,d)
        cls.state=s._replace(dec=jnp.int32(core.D_RUN),ch_src=jnp.int32(0),ch_eff=jnp.int32(0),
            ch_p=jnp.int32(0),pg=jnp.int32(core.PG_RESOLVE),pc=jnp.int32(0),
            zone=s.zone.at[5].set(core.Z_GRAVE),reg=s.reg.at[0,5].set(True))
        cls.execute=staticmethod(jax.jit(core.exec_one))
    def test_legal_special_summon_still_opens_placement(self):
        self.assertEqual(int(self.execute(self.bank,self.state).dec),core.D_ZONE)
    def test_a_later_prohibition_cancels_special_summon(self):
        s=self.state._replace(fl_c=self.state.fl_c.at[5].set(core.FL_NO_SS))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.pc),1);self.assertFalse(bool(result.reg[0].any()))
    def test_filled_main_zones_do_not_leave_a_decision_without_actions(self):
        s=self.state._replace(zone=self.state.zone.at[:5].set(core.Z_MZONE),seq=self.state.seq.at[:5].set(jnp.arange(5)))
        self.assertEqual(int(self.execute(self.bank,s).pc),1)
    def test_a_monster_already_on_field_is_not_special_summonable(self):
        s=self.state._replace(zone=self.state.zone.at[5].set(core.Z_MZONE),seq=self.state.seq.at[5].set(0))
        self.assertEqual(int(self.execute(self.bank,s).pc),1)
        mask=core.filter_mask(self.bank,s,jnp.array(cards.F(me=core.Z_MZONE,spsum=1)),0,0)
        self.assertFalse(bool(mask[5]))
    def test_a_failed_attempt_replaces_native_summon_information(self):
        s=self.state._replace(fl_c=self.state.fl_c.at[5].set(core.FL_NO_SS),
            suminfo=self.state.suminfo.at[5].set(core.SUMMON_TRIBUTE))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.suminfo[5]),int(core.summon_info(core.SUMMON_SPECIAL,core.Z_GRAVE,0)))
        self.assertEqual(int(result.zone[5]),core.Z_GRAVE)
    def test_a_space_failure_sends_a_hand_target_to_grave_by_rule(self):
        s=self.state._replace(zone=self.state.zone.at[:5].set(core.Z_MZONE).at[5].set(core.Z_HAND),
            seq=self.state.seq.at[:5].set(jnp.arange(5)).at[5].set(0))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.zone[5]),core.Z_GRAVE)
        self.assertEqual(int(result.suminfo[5]),int(core.summon_info(core.SUMMON_SPECIAL,core.Z_HAND,0)))
if __name__=='__main__':unittest.main()
