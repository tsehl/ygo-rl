"""Deux destructions de la relation à la source, comparées à ygopro-core."""
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

SPELL, COST, NORMAL, MST = 600930031, 600930041, 44287299, 5318639
SPELL_SCRIPT = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetOperation(function(e,tp)
  if e:GetHandler():IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''
COST_SCRIPT = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return true end
  Duel.SendtoGrave(e:GetHandler(),REASON_COST)
 end)
 e:SetOperation(function(e,tp)
  if not e:GetHandler():IsRelateToEffect(e) then return end
  Duel.Recover(tp,1234,REASON_EFFECT)
 end)
 c:RegisterEffect(e)
end'''
for code in (SPELL, COST):
    driver.TEST_CARDS[code] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Source relation probe')

with tempfile.TemporaryDirectory() as folder:
    Path(folder, f'c{SPELL}.lua').write_text(SPELL_SCRIPT)
    Path(folder, f'c{COST}.lua').write_text(COST_SCRIPT)
    shutil.copy(Path(translate.OFFICIAL, f'c{MST}.lua'), folder)
    translate.OFFICIAL = folder
    driver.SCRIPTS.insert(0, folder)
    db, bank, cmap = build([SPELL, COST, NORMAL, MST])
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    for code, interrupt in ((SPELL, True), (COST, False)):
        fired = replied = False

        def chooser(dec, indices):
            global fired, replied
            turn = int(compare.LAST_STATE.turn)
            def find(kind, card):
                return next((i for i in indices if dec.options[i][0] == kind
                             and dec.options[i][1] == card), None)
            choice = None
            if interrupt and turn == 2 and dec.player == 1:
                choice = find('sset', MST)
            if (turn >= 3 or not interrupt) and dec.player == 0 and not fired:
                choice = find('activate', code)
                if choice is not None:
                    fired = True
            if interrupt and fired and not replied and dec.player == 1:
                choice = find('chain', MST)
                if choice is not None:
                    replied = True
            if choice is None and dec.options[0][0] == 'cards':
                choice = next((i for i in indices if any(c[0] == code for c in dec.options[i][1])), None)
            if choice is None:
                choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
            return choice

        result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                              given=([[code, NORMAL] * 20, [MST if interrupt else NORMAL, NORMAL] * 20], [[], []]))
        assert result['ok'], result.get('err')
        assert fired and replied == interrupt, (fired, replied)
        assert any(e['code'] == code and e['stage'] == 'resolved' and e['count']
                   for e in result['coverage']), result['coverage']
        print(code, 'source déplacée : partie identique', flush=True)
