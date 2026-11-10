"""Une cible déplacée pendant le ciblage perd sa relation à l’effet."""
from pathlib import Path
import shutil
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'engine'), str(ROOT / 'translate'), str(ROOT / 'oracle')]
import compare
import core
import driver
import translate
from oracle_bank import build

SPELL, NORMAL, MST = 600930051, 44287299, 5318639
driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Target relation probe')
script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chkc then return chkc:IsControler(tp) and chkc:IsLocation(LOCATION_SZONE) and chkc~=e:GetHandler() end
  if chk==0 then return Duel.IsExistingTarget(Card.IsSpellTrap,tp,LOCATION_SZONE,0,1,e:GetHandler()) end
  local g=Duel.SelectTarget(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,1,1,e:GetHandler())
  Duel.SendtoGrave(g,REASON_EFFECT)
 end)
 e:SetOperation(function(e,tp)
  local tc=Duel.GetFirstTarget()
  if tc and tc:IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''
with tempfile.TemporaryDirectory() as folder:
    Path(folder, f'c{SPELL}.lua').write_text(script)
    shutil.copy(Path(translate.OFFICIAL, f'c{MST}.lua'), folder)
    translate.OFFICIAL = folder
    driver.SCRIPTS.insert(0, folder)
    db, bank, cmap = build([SPELL, NORMAL, MST])
    fired = replied = False

    def chooser(dec, indices):
        global fired, replied
        turn = int(compare.LAST_STATE.turn)
        def find(kind, card):
            return next((i for i in indices if dec.options[i][0] == kind
                         and dec.options[i][1] == card), None)
        choice = None
        if turn == 1 and dec.player == 0:
            choice = find('sset', MST)
        if turn >= 3 and dec.player == 0 and not fired:
            choice = find('activate', SPELL)
            if choice is not None:
                fired = True
        if False and fired and not replied and dec.player == 1:
            choice = find('chain', MST)
            if choice is not None:
                replied = True
        if choice is None and dec.options[0][0] == 'cards':
            choice = next((i for i in indices if any(c[0] == MST and c[1] == 0
                                                     for c in dec.options[i][1])), None)
        if choice is None:
            choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
        return choice

    result = compare.play(db, bank, cmap, 3400, jax.jit(core.step), jax.jit(core.legal), mode='tr',
                          chooser=chooser, given=([[SPELL, MST, NORMAL, NORMAL] * 10, [MST, NORMAL] * 20], [[], []]))
    assert result['ok'], result.get('err')
    assert fired and not replied, (fired, replied)
    for code in (SPELL,):
        assert any(e['code'] == code and e['stage'] == 'resolved' and e['count']
                   for e in result['coverage']), result['coverage']
    print('cible déplacée pendant le ciblage : partie identique', flush=True)
