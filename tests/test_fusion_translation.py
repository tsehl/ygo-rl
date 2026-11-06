"""Recettes exactes et refus explicites des variantes non représentables."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'translate'), str(ROOT/'engine')]
import translate as T
import cards
from luaparser import ast


class FusionTranslationTests(unittest.TestCase):
    def recipe(self, expression):
        call = ast.parse(expression).body.body[0]
        translator = T.Translator.__new__(T.Translator)
        translator.code = 1
        return translator.procedure(T.dotted(call.func), call.args[1:], {})

    def test_distinct_named_materials(self):
        self.assertEqual(self.recipe('Fusion.AddProcMix(c,false,true,34460851,44287299)'),
                         ('fusion', [cards.MAT(code=34460851), cards.MAT(code=44287299)], False))

    def test_repeated_materials(self):
        self.assertEqual(self.recipe('Fusion.AddProcMixN(c,false,true,89631139,3)'),
                         ('fusion', [cards.MAT(code=89631139)] * 3, False))

    def test_filter_materials(self):
        self.assertEqual(self.recipe('Fusion.AddProcMixN(c,true,true,aux.FilterBoolFunctionEx(Card.IsType,TYPE_NORMAL),2)'),
                         ('fusion', [cards.MAT(typ=T.core.TYPE_NORMAL)] * 2, 0))

    def test_unsupported_options_are_rejected(self):
        for expression in ('Fusion.AddProcMixN(c,false,true,1,5)',
                           'Fusion.AddProcMixN(c,false,true,1,0)',
                           'Fusion.AddProcMixN(c,false,true,1,2,2)',
                           'Fusion.AddProcMix(c,false,true,1)',
                           'Fusion.AddProcMix(c,false,true,nil,nil)',
                           'Fusion.AddProcMix(c,false,true,true,false)'):
            with self.subTest(expression=expression), self.assertRaises(T.Unsupported):
                self.recipe(expression)

    def test_polymerization_default_effect(self):
        effects, _, proc, _ = T.translate(24094653)
        self.assertEqual(effects, cards.CARDS[24094653])
        self.assertIsNone(proc)


    def test_stringid_matches_pinned_oracle_utility(self):
        # Même codage que data/CardScripts/utility.lua, y compris le masque de l'index.
        translator = T.Translator.__new__(T.Translator)
        for code, index, expected in ((11384280, 0, 11937282785280),
                                      (6983839, 1, 7323085963265),
                                      (12, 0x100003, 12582915)):
            node = ast.parse(f'local value=aux.Stringid({code},{index})').body.body[0].values[0]
            self.assertEqual(translator.ie_val(node, {}), expected)


    def test_named_filter_does_not_gain_numeric_substitution_permission(self):
        proc = self.recipe('Fusion.AddProcMix(c,true,true,aux.FilterBoolFunction(Card.IsCode,100),200)')
        self.assertEqual(proc[2], 2)
        self.assertEqual(self.recipe('Fusion.AddProcMixN(c,true,true,100,3)')[2], 7)

    def test_real_substitute_property(self):
        effects,_,_,_=T.translate(99426834)
        self.assertEqual(effects,[cards.cont(T.core.Z_HAND|T.core.Z_MZONE|T.core.Z_SZONE|T.core.Z_GRAVE,T.core.MC_FUS_SUB,0)])


if __name__ == '__main__':
    unittest.main()
