"""Permissions lisibles par le réseau ; compatibilité des checkpoints existants."""
import sys
from pathlib import Path
import gzip,pickle,unittest
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'train')]
import core,cards,model


FSUB=slice(-core.FK-model.NUM,-model.NUM)  # permissions de substitution, avant les NUM features


class EncodingTests(unittest.TestCase):
    def bank(self):
        return core.make_bank(np.array([100,200,300]),[1000]*3,[1000]*3,[4]*3,ctype=[33,65,65],
                              **cards.assemble([100,200,300],cards={100:[cards.cont(core.Z_HAND|core.Z_MZONE,core.MC_FUS_SUB,0)]},
                                               fusion={200:cards.FusionRecipe([cards.MAT(code=100)]*2,True),
                                                       300:cards.FusionRecipe([cards.MAT(code=100)]*2,False)},
                                               ritual={},synchro={},xyz={},link={}))

    def test_permissions_and_passive_range_are_visible(self):
        bank=self.bank();tok,mask=model.card_tokens(bank)
        self.assertEqual(tok['head'].shape[-1],model.F_HEAD)
        np.testing.assert_array_equal(np.asarray(tok['head'][1,FSUB]),[1,1,0,0])
        np.testing.assert_array_equal(np.asarray(tok['head'][2,FSUB]),[0,0,0,0])
        self.assertEqual(int(bank.erng[0,0]),core.Z_HAND|core.Z_MZONE)

    def test_old_checkpoint_encoder_matches_zero_extension(self):
        with gzip.open(ROOT/'train/ckpt/run3_it120_f16.pkl.gz','rb') as f: params=pickle.load(f)
        params=jax.tree.map(lambda x:jnp.asarray(x,jnp.float32),params)
        bank=self.bank()
        tok,mask=model.card_tokens(bank)
        # Référence : migration manuelle des anciens poids, zéro sur les nouveaux bits.
        migrated=dict(params['cards'])
        old_head=params['cards']['head']
        migrated['head']=dict(old_head,w=jnp.pad(old_head['w'],((0,core.FK),(0,0))))
        expected=model.encode_cards(migrated,tok,mask)
        actual=model.encode_cards(params['cards'],tok,mask)
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
