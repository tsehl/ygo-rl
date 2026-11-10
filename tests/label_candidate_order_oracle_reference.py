"""L’ordre des candidats influence les labels et les actions proposées."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import driver
S,N=600930061,600930062
db=driver.CardDB();db.add({S:((0,0,2,0,0,0,0,0),'labels'),N:((0,0,17,1000,1000,4,1,1),'normal')})
base='''local s,id=GetID()
function s.initial_effect(c)
 local e1=Effect.CreateEffect(c)
 e1:SetType(EFFECT_TYPE_ACTIVATE) e1:SetCode(EVENT_FREE_CHAIN)
 e1:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then e:SetLabel(1) return false end
 end)
 local e2=Effect.CreateEffect(c)
 e2:SetType(EFFECT_TYPE_ACTIVATE) e2:SetCode(EVENT_FREE_CHAIN)
 e2:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return e1:GetLabel()==1 end
 end)
 e2:SetOperation(function(e,tp) Duel.Recover(tp,1234,REASON_EFFECT) end)
 REGISTRATION
end'''
with tempfile.TemporaryDirectory() as folder:
 driver.SCRIPTS.insert(0,folder)
 for reverse in (False,True):
  order='c:RegisterEffect(e2) c:RegisterEffect(e1)' if reverse else 'c:RegisterEffect(e1) c:RegisterEffect(e2)'
  Path(folder,f'c{S}.lua').write_text(base.replace('REGISTRATION',order))
  d=driver.Duel(db,[S,N]*20,[N]*40,seed=3400)
  try:
   dec=d.next_decision(); acts=[o for o in dec.options if o[0]=='activate' and o[1]==S]
   assert not d.log, d.log
   assert bool(acts) == (not reverse), (reverse, acts)
   if acts:
    pick=dec.options.index(acts[0]);d.respond(dec,pick)
    for _ in range(20):
     dec=d.next_decision()
     if d.lp[0]!=8000:break
     assert dec is not None
     choice=next((i for i,o in enumerate(dec.options) if o[0] in ('pass','to_ep')),0)
     d.respond(dec,choice)
    assert d.lp[0]==9234, d.lp
    assert any(c==S and stage=='resolved' and n for (c,stage,_),n in d.coverage.items())
   print('ordre inversé' if reverse else 'ordre initial', 'candidat disponible', bool(acts))
  finally:d.close()
