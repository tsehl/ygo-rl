"""Référence native : une magie détruite en chaîne perd sa relation à son effet."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'oracle'))
import driver

SPELL, NORMAL, MST = 600930031, 600930032, 5318639
db = driver.CardDB()
db.add({SPELL: ((0, 0, 2, 0, 0, 0, 0, 0), 'Relation probe'),
        NORMAL: ((0, 0, 17, 1000, 1000, 4, 1, 1), 'Normal')})
script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetOperation(function(e,tp)
  if e:GetHandler():IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end
'''

with tempfile.TemporaryDirectory() as folder:
    Path(folder, f'c{SPELL}.lua').write_text(script)
    driver.SCRIPTS.insert(0, folder)
    for interrupt, expected in ((False, 9234), (True, 8000)):
        duel = driver.Duel(db, [SPELL, NORMAL] * 20, [MST, NORMAL] * 20, seed=3400)
        fired = replied = False
        try:
            for step in range(80):
                dec = duel.next_decision()
                assert dec is not None
                if any(c == SPELL and stage == 'resolved' and n
                       for (c, stage, _), n in duel.coverage.items()):
                    break
                def find(kind, code):
                    return next((i for i, o in enumerate(dec.options)
                                 if o[0] == kind and o[1] == code), None)
                choice = None
                if interrupt and duel.turn == 2 and dec.player == 1:
                    choice = find('sset', MST)
                if duel.turn >= 3 and dec.player == 0:
                    choice = find('activate', SPELL)
                    if choice is not None:
                        fired = True
                if interrupt and fired and not replied and dec.player == 1:
                    choice = find('chain', MST)
                    if choice is not None:
                        replied = True
                if choice is None and dec.options[0][0] == 'cards':
                    choice = next((i for i, o in enumerate(dec.options)
                                   if any(c[0] == SPELL for c in o[1])), None)
                if choice is None:
                    choice = next((i for i, o in enumerate(dec.options)
                                   if o[0] in ('pass', 'to_ep')), 0)
                duel.respond(dec, choice)
            assert fired and replied == interrupt, (fired, replied)
            assert not duel.log, duel.log
            assert duel.lp[0] == expected, (interrupt, duel.lp, step)
            assert any(c == SPELL and stage == 'resolved' and n
                       for (c, stage, _), n in duel.coverage.items())
            print('réponse MST' if interrupt else 'sans réponse', 'LP', expected, flush=True)
        finally:
            duel.close()
