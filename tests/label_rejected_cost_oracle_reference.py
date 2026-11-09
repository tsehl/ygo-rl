"""Un coût impossible laisse ses écritures de label visibles aux autres effets."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import driver
S,N=600930021,600930022
db=driver.CardDB();db.add({S:((0,0,2,0,0,0,0,0),'Label'),N:((0,0,17,1000,1000,4,1,1),'Normal')})
script='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then e:SetLabel(e:GetLabel()+1) return false end
 end)
 c:RegisterEffect(e)
 local e2=Effect.CreateEffect(c)
 e2:SetType(EFFECT_TYPE_ACTIVATE) e2:SetCode(EVENT_FREE_CHAIN)
 e2:SetOperation(function(e2,tp) Duel.Recover(tp,e:GetLabel(),REASON_EFFECT) end)
 c:RegisterEffect(e2)
end'''
with tempfile.TemporaryDirectory() as folder:
 Path(folder,f'c{S}.lua').write_text(script);driver.SCRIPTS.insert(0,folder)
 d=driver.Duel(db,[S,N]*20,[N]*40,seed=3400)
 try:
  for step in range(30):
   dec=d.next_decision()
   if d.lp[0]!=8000:break
   act=next((i for i,o in enumerate(dec.options) if o[0]=='activate' and o[1]==S),None)
   choice=act if act is not None else next((i for i,o in enumerate(dec.options) if o[0] in ('pass','to_ep')),0)
   d.respond(dec,choice)
  assert not d.log, d.log
  assert d.lp[0] == 8001, (d.lp, step)
  assert any(c == S and stage == 'resolved' and n for (c, stage, _), n in d.coverage.items())
  print('coût refusé : écriture de label conservée et visible')
 finally:d.close()
