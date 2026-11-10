from pathlib import Path
import sys, unittest
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine'),str(ROOT/'train')]
import translate as T, core, cards, model, jax
import jax.numpy as jnp
import numpy as np

class NegativeTypeTests(unittest.TestCase):
    def test_material_filter_can_include_and_exclude_types(self):
        bank=core.make_bank(np.array([1,2,3]),[500,500,0],[500,500,0],[4,4,0],
            ctype=[core.TYPE_MONSTER|core.TYPE_NORMAL,core.TYPE_MONSTER|cards.TYPE_EFFECT,core.TYPE_SPELL])
        state=SimpleNamespace(cid=jnp.array([0,1,2]))
        actual=core.mat_mask(bank,state,jnp.asarray(cards.MAT(typ=core.TYPE_MONSTER,not_typ=cards.TYPE_EFFECT)))
        np.testing.assert_array_equal(actual,[True,False,False])
    def test_not_type_combines_with_positive_and_other_exclusions(self):
        tr=T.Translator.__new__(T.Translator); fl=T.Filt()
        for text in ('c:IsType(TYPE_MONSTER)','not c:IsType(TYPE_EFFECT)','not c:IsType(TYPE_FUSION)'):
            node=T.LA.parse('return '+text).body.body[0].values[0]
            tr.filter_term(fl,node,{},'c')
        result=fl.build(core.Z_GRAVE,0)
        self.assertEqual(result[core.F_TYPE],core.TYPE_MONSTER)
        self.assertEqual(result[core.F_NOTTYPE],cards.TYPE_EFFECT|core.TYPE_FUSION)
    def test_disjunction_with_negation_is_not_mistaken_for_type_union(self):
        node=T.LA.parse('return (not c:IsType(TYPE_EFFECT)) or c:IsType(TYPE_SPELL)').body.body[0].values[0]
        tr=T.Translator.__new__(T.Translator)
        with self.assertRaises(T.Unsupported):tr.filter_term(T.Filt(),node,{},'c')
    def test_previous_reason_checkpoint_keeps_its_feature_order(self):
        effect=cards.act(resolve=[cards.FILTER(0,cards.F(me=core.Z_GRAVE,
                    reason=core.REASON_LINK,not_typ=cards.TYPE_EFFECT))])
        bank=core.make_bank(np.array([1]),[500],[500],[4],
                **cards.assemble([1],cards={1:[effect]},ritual={},fusion={},synchro={},xyz={},link={}))
        tokens,mask=model.card_tokens(bank)
        self.assertTrue(bool(jnp.any(tokens['filt'][...,model.F_FILT-33:model.F_FILT-17])))
        params=model.card_enc_init(jax.random.PRNGKey(19),16,1)
        old=dict(params)
        for name in ('filt','matf'):
            old[name]=dict(params[name],w=params[name]['w'][:-17])
        padded=dict(old)
        for name in ('filt','matf'):
            padded[name]=dict(old[name],w=jnp.pad(old[name]['w'],((0,17),(0,0))))
        actual=model.encode_cards(old,tokens,mask)
        expected=model.encode_cards(padded,tokens,mask)
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)

    def test_invalid_exclusion_masks_are_rejected(self):
        fl=T.Filt()
        for value in (0,-1,1<<31):
            with self.subTest(value=value),self.assertRaises(T.Unsupported):fl.set('not_typ',value)

if __name__=='__main__':unittest.main()
