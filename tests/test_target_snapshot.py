"""Le groupe cible survit à une opération qui réécrit son registre résultat."""
from pathlib import Path
import sys
import ast
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import core
import cards
import translate

SCRIPT = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chk==0 then return Duel.IsExistingTarget(Card.IsMonster,tp,LOCATION_MZONE,0,1,nil) end
  local g=Duel.SelectTarget(tp,Card.IsMonster,tp,LOCATION_MZONE,0,1,1,nil)
  Duel.Destroy(g,REASON_EFFECT)
  EXTRA
 end)
 e:SetOperation(function(e,tp)
  local tc=Duel.GetFirstTarget()
  if tc and tc:IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''

class TargetSnapshotTests(unittest.TestCase):
    def translate(self, extra=''):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930091.lua').write_text(SCRIPT.replace('EXTRA', extra))
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                return translate.translate(600930091)[0][0]
            finally:
                translate.OFFICIAL = old

    def test_copy_is_elided_when_group_is_dead(self):
        allocate = translate.Translator.allocate
        prefix = [cards.SELF(0)]
        copied = prefix + [translate.Translator.vcopy(1, 0), cards.DESTROY(1)]
        direct = prefix + [cards.DESTROY(0)]
        def lower(program):
            return allocate((0, 0, 0, 0, [], [], program))[6]
        self.assertEqual(lower(copied), lower(direct))

    def test_condition_precedes_destructive_action(self):
        module = ast.parse((ROOT / 'tests' / 'condition_before_action_oracle_reference.py').read_text())
        script = next(ast.literal_eval(node.value) for node in module.body
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'spell')
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930091.lua').write_text(script)
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                program = translate.translate(600930091)[0][0][6]
            finally:
                translate.OFFICIAL = old
        self.assertLess(next(i for i, op in enumerate(program) if op[0] == core.OP_FILTER),
                        next(i for i, op in enumerate(program) if op[0] == core.OP_MOVE))

    def test_target_snapshot_is_distinct_from_destroy_result(self):
        effect = self.translate()
        target, resolve = effect[5], effect[6]
        destroy = next(x for x in target if x[0] == core.OP_MOVE and x[3] & 1)
        requirement = next(x for x in resolve if x[0] == core.OP_REQ and x[3] & 2)
        self.assertNotEqual(destroy[1], requirement[1])
        self.assertTrue(any(x[0] == core.OP_FILTER and x[1] == requirement[1] for x in target))

    def test_later_use_preserves_original_lua_group(self):
        effect = self.translate('Duel.SendtoGrave(g,REASON_EFFECT)')
        target = effect[5]
        destroy = next(x for x in target if x[0] == core.OP_MOVE and x[3] & 1)
        send = next(x for x in target if x[0] == core.OP_MOVE and not x[3] & 1)
        self.assertNotEqual(destroy[1], send[1])

if __name__ == '__main__':
    unittest.main()
