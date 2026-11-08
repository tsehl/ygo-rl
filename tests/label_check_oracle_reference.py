"""Référence native : une valeur de label passe des contrôles coût/cible à l'activation."""
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'oracle'))
import driver
SPELL,NORMAL=600930001,600930002
db=driver.CardDB()
db.add({SPELL:((0,0,2,0,0,0,0,0),'Label probe'),NORMAL:((0,0,17,1000,1000,4,1,1),'Normal')})
script='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE)
 e:SetCode(EVENT_FREE_CHAIN)
 e:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then e:SetLabel(100) return true end
  if e:GetLabel()==0 then e:SetLabel(200) else e:SetLabel(999) end
 end)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then
   if e:GetLabel()~=100 then return false end
   e:SetLabel(0)
   return true
  end
  if e:GetLabel()==200 then e:SetLabel(300) else e:SetLabel(999) end
 end)
 e:SetOperation(function(e,tp)
  if e:GetLabel()==300 then Duel.Recover(tp,1234,REASON_EFFECT)
  else Duel.Damage(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end
'''
with tempfile.TemporaryDirectory() as folder:
    Path(folder,f'c{SPELL}.lua').write_text(script)
    driver.SCRIPTS.insert(0,folder)
    for seed in (3400,3401):
        duel=driver.Duel(db,[SPELL,NORMAL]*20,[NORMAL]*40,seed=seed)
        try:
            for step in range(100):
                dec=duel.next_decision()
                assert dec is not None
                if duel.lp[0]!=8000:break
                choice=next((i for i,o in enumerate(dec.options) if o[0]=='activate' and o[1]==SPELL),None)
                if choice is None:
                    choice=next((i for i,o in enumerate(dec.options) if o[0] in ('pass','to_ep')),0)
                duel.respond(dec,choice)
            assert not duel.log,duel.log
            assert duel.lp[0]==9234,(seed,duel.lp,step)
            assert any(c==SPELL and stage=='resolved' and n for (c,stage,_),n in duel.coverage.items())
            print(seed,'label transmis entre contrôles, paiement, ciblage et résolution',flush=True)
        finally:duel.close()
