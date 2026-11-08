from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T
class RawSummonTypeTests(unittest.TestCase):
    def translate_source(self,old,new):
        source=Path(T.OFFICIAL,'c69015963.lua').read_text().replace(old,new)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'c69015963.lua').write_text(source)
            with patch.object(T,'OFFICIAL',folder):return T.translate(69015963)
    def test_custom_procedure_type_is_not_silently_compiled_as_generic(self):
        with self.assertRaisesRegex(T.Unsupported,'type non représenté'):
            self.translate_source('Duel.SpecialSummon(g,0,','Duel.SpecialSummon(g,SUMMON_TYPE_FUSION,')
    def test_custom_summon_condition_is_not_silently_checked_as_generic(self):
        with self.assertRaisesRegex(T.Unsupported,'type non représenté'):
            self.translate_source('IsCanBeSpecialSummoned(e,0,','IsCanBeSpecialSummoned(e,SUMMON_TYPE_FUSION,')
    def test_explicit_special_type_retains_generic_semantics(self):
        actual=self.translate_source('Duel.SpecialSummon(g,0,','Duel.SpecialSummon(g,SUMMON_TYPE_SPECIAL,')
        self.assertEqual(actual,T.translate(69015963))
    def test_native_generic_markers_preserve_the_previous_four_cards(self):
        import core,cards
        for code,tag in ((83764718,1010),(3283679,156),(28877602,152),(81873903,151)):
            effects,*_=T.translate(code)
            operations=[ins for effect in effects for ins in effect[6] if ins[0]==core.OP_SPSUMMON]
            self.assertTrue(operations,code)
            self.assertEqual(operations[-1][3],-(tag+1))
        self.assertEqual(T.K['SUMMON_WITH_MONSTER_REBORN'],1010)
        self.assertEqual(cards.CARDS[83764718][0][6][-1][3],-1011)
if __name__=='__main__':unittest.main()
