"""Les groupes de cibles et d'événement potentiellement multiples sont refusés."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import translate


class UnsupportedGroups(unittest.TestCase):
    def translate_script(self, text):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c4206964.lua').write_text(text)
            old = translate.OFFICIAL
            try:
                translate.OFFICIAL = folder
                return translate.translate(4206964)
            finally:
                translate.OFFICIAL = old

    def test_event_group_can_contain_multiple_special_summons(self):
        script = Path(translate.OFFICIAL, 'c4206964.lua').read_text()
        script = script.replace('EVENT_SUMMON_SUCCESS', 'EVENT_SPSUMMON_SUCCESS')
        script = script.replace('EVENT_FLIP_SUMMON_SUCCESS', 'EVENT_SPSUMMON_SUCCESS')
        with self.assertRaisesRegex(translate.Unsupported, 'groupe potentiellement multiple'):
            self.translate_script(script)

    def test_select_target_multiple_is_refused(self):
        script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return Duel.IsExistingTarget(Card.IsMonster,tp,LOCATION_MZONE,0,1,nil) end
  Duel.SelectTarget(tp,Card.IsMonster,tp,LOCATION_MZONE,0,1,2,nil)
 end)
 c:RegisterEffect(e)
end'''
        with self.assertRaisesRegex(translate.Unsupported, 'cibles multiples'):
            self.translate_script(script)


if __name__ == '__main__':
    unittest.main()
