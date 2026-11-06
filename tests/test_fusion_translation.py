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
                         ('fusion', [cards.MAT(code=34460851), cards.MAT(code=44287299)]))

    def test_repeated_materials(self):
        self.assertEqual(self.recipe('Fusion.AddProcMixN(c,false,true,89631139,3)'),
                         ('fusion', [cards.MAT(code=89631139)] * 3))

    def test_filter_materials(self):
        self.assertEqual(self.recipe('Fusion.AddProcMixN(c,true,true,aux.FilterBoolFunctionEx(Card.IsType,TYPE_NORMAL),2)'),
                         ('fusion', [cards.MAT(typ=T.core.TYPE_NORMAL)] * 2))

    def test_unsupported_options_are_rejected(self):
        for expression in ('Fusion.AddProcMix(c,true,true,1,2)',
                           'Fusion.AddProcMixN(c,false,true,1,5)',
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


if __name__ == '__main__':
    unittest.main()
