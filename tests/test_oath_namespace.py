"""Deux compteurs de même nom et de drapeaux différents restent indépendants."""
from pathlib import Path
import sys,unittest,tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'translate')]
import core,cards,cardgen,translate as T,jax
import jax.numpy as jnp
import numpy as np

class OathNamespaceTests(unittest.TestCase):
    def test_distinct_keys_and_refund(self):
        regular=cards.ignition(opt=core.OPT_NAME)
        oath=(*regular[:7],{'refund':1})
        bank=core.make_bank(np.array([1]),[1000],[1000],[4],
            **cards.assemble([1],cards={1:[regular,oath]},ritual={},fusion={},synchro={},xyz={},link={}))
        deck=jnp.zeros(40,jnp.int32);state=jax.jit(core.new_game)(bank,deck,deck)
        used=jax.jit(core.use_count)(bank,state,jnp.int32(0),jnp.int32(1),jnp.int32(0))
        regular_key=core.opt_key(jnp.int32(0),jnp.int32(core.OPT_NAME))
        oath_key=core.opt_key(jnp.int32(0),jnp.int32(core.OPT_NAME),1)
        self.assertNotEqual(int(regular_key),int(oath_key))
        self.assertFalse(bool(core.name_in(used.opt_cid[0],regular_key)))
        self.assertTrue(bool(core.name_in(used.opt_cid[0],oath_key)))
        used=used._replace(lk_src=used.lk_src.at[0].set(0),lk_eff=used.lk_eff.at[0].set(1),
            lk_p=used.lk_p.at[0].set(0),lk_flags=used.lk_flags.at[0].set(core.CF_ACT_NEG))
        refunded=jax.jit(core.load_link)(bank,used,jnp.int32(0))
        self.assertFalse(bool(core.name_in(refunded.opt_cid[0],oath_key)))
    def test_translated_procedure_records_oath(self):
        path=Path(T.OFFICIAL,'c70095154.lua')
        source=path.read_text().replace('e1:SetRange(LOCATION_HAND)',
            'e1:SetRange(LOCATION_HAND)\n\te1:SetCountLimit(1,id,EFFECT_COUNT_CODE_OATH)')
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'c70095154.lua').write_text(source)
            with patch.object(T,'OFFICIAL',folder):effects,*_=T.translate(70095154)
        self.assertEqual(effects[0][0],core.K_PROC)
        self.assertEqual(effects[0][7]['refund'],1)

    def test_procedural_oath_is_recorded(self):
        spec={'kind':'hand_proc','opt':core.OPT_NAME,'counts':[],'cost':None}
        result=cardgen.bricks(spec)
        self.assertEqual(result[7]['refund'],1)

if __name__=='__main__':unittest.main()
