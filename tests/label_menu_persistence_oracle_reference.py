"""Référence native : un contrôle de coût modifie le label même si l'action n'est pas choisie."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'oracle'))
import driver

SPELL, NORMAL = 600930011, 600930012
db = driver.CardDB()
db.add({SPELL: ((0, 0, 2, 0, 0, 0, 0, 0), 'Label probe'),
        NORMAL: ((0, 0, 17, 1000, 1000, 4, 1, 1), 'Normal')})
script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE)
 e:SetCode(EVENT_FREE_CHAIN)
 e:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then e:SetLabel(e:GetLabel()+1) return true end
 end)
 e:SetOperation(function(e,tp) Duel.Recover(tp,e:GetLabel(),REASON_EFFECT) end)
 c:RegisterEffect(e)
end
'''

with tempfile.TemporaryDirectory() as folder:
    Path(folder, f'c{SPELL}.lua').write_text(script)
    driver.SCRIPTS.insert(0, folder)
    # Même graine et même carte en main : on active immédiatement, ou après
    # avoir décliné la première proposition et invoqué un monstre normal.
    for delayed, expected in ((False, 1), (True, 2)):
        duel = driver.Duel(db, [SPELL, NORMAL] * 20, [NORMAL] * 40, seed=3400)
        summoned = False
        try:
            for step in range(30):
                dec = duel.next_decision()
                assert dec is not None
                if duel.lp[0] != 8000:
                    break
                activate = next((i for i, o in enumerate(dec.options)
                                 if o[0] == 'activate' and o[1] == SPELL), None)
                summon = next((i for i, o in enumerate(dec.options) if o[0] == 'summon'), None)
                if delayed and not summoned and summon is not None and activate is not None:
                    choice, summoned = summon, True
                elif activate is not None:
                    choice = activate
                else:
                    choice = next((i for i, o in enumerate(dec.options)
                                   if o[0] in ('pass', 'to_ep')), 0)
                duel.respond(dec, choice)
            assert not duel.log, duel.log
            assert duel.lp[0] == 8000 + expected, (delayed, duel.lp, step)
            assert summoned == delayed
            assert any(c == SPELL and stage == 'resolved' and n
                       for (c, stage, _), n in duel.coverage.items())
            print('activation retardée' if delayed else 'activation immédiate',
                  'label', expected, flush=True)
        finally:
            duel.close()
