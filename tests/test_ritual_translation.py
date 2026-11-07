"""Rituels : propriétés sans slot d'effet et limites exactes des procédures standard."""
from pathlib import Path
import sys,pickle,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T
import cards,core

class RitualTranslationTests(unittest.TestCase):
    def test_vanilla_ritual_keeps_its_revive_limit_without_an_effect_slot(self):
        effects,*_=T.translate(5405694)
        self.assertEqual(effects,[])
        self.assertEqual(effects.revive,1)
        effects=pickle.loads(pickle.dumps(effects))
        bank=cards.assemble([5405694],cards={5405694:effects},ritual={})
        self.assertEqual(int(bank['erl'][0]),1)
        self.assertFalse(bank['ekind'].any())

    def test_standard_fixed_and_equal_ritual_procedures(self):
        for code,mode in ((55761792,core.RIT_GREATER),(8198712,core.RIT_EQUAL)):
            with self.subTest(code=code):
                effects,_,_,descs=T.translate(code)
                self.assertEqual(descs,[1171])
                program=effects[0][6]
                self.assertEqual([ins[3] for ins in program if ins[0]==core.OP_RITMAT],[mode])
                self.assertEqual(program[-1],cards.SPSUMMON(1,core.P_FACEUP,True))

    def test_cost_check_with_else_compiles_the_actual_payment(self):
        effects,*_=T.translate(72426662)
        self.assertEqual(effects.revive,1)
        self.assertEqual(effects[0][4],[cards.PAY_LP(2000)])

    def test_different_fixed_level_and_custom_materials_are_refused(self):
        for call in ('Ritual.AddProcGreaterCode(c,7,nil,5405694)',
                     'Ritual.AddProcGreater(c,nil,nil,nil,function() return Group.CreateGroup() end)'):
            source='local s,id=GetID()\nfunction s.initial_effect(c)\n'+call+'\nend\n'
            with self.subTest(call=call),tempfile.TemporaryDirectory() as tmp:
                Path(tmp,'c999999975.lua').write_text(source)
                with patch.object(T,'OFFICIAL',tmp),self.assertRaises(T.Unsupported):T.translate(999999975)

if __name__=='__main__':unittest.main()
