"""Un label d’effet reste accessible après le déplacement de sa carte."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import driver
S,N=600930071,600930072
db=driver.CardDB();db.add({S:((0,0,33,1000,1000,4,1,1),'Label move'),N:((0,0,17,1000,1000,4,1,1),'Normal')})
script='''local s,id=GetID()
function s.initial_effect(c)
 local e1=Effect.CreateEffect(c)
 e1:SetType(EFFECT_TYPE_IGNITION) e1:SetRange(LOCATION_MZONE)
 e1:SetOperation(function(e,tp)
  e:SetLabel(7) Duel.SendtoGrave(e:GetHandler(),REASON_EFFECT)
 end)
 c:RegisterEffect(e1)
 local e2=Effect.CreateEffect(c)
 e2:SetType(EFFECT_TYPE_SINGLE|EFFECT_TYPE_TRIGGER_F) e2:SetCode(EVENT_TO_GRAVE)
 e2:SetOperation(function(e,tp)
  if e1:GetLabel()==7 then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e2)
end'''
with tempfile.TemporaryDirectory() as folder:
 Path(folder,f'c{S}.lua').write_text(script);driver.SCRIPTS.insert(0,folder)
 d=driver.Duel(db,[S,N]*20,[N]*40,seed=3400)
 try:
  for step in range(50):
   dec=d.next_decision()
   if d.lp[0]!=8000:break
   choice=None
   for typ in ('activate','chain','summon','to_ep','pass'):
    choice=next((i for i,o in enumerate(dec.options) if o[0]==typ and (typ not in ('activate','chain','summon') or o[1]==S)),None)
    if choice is not None:break
   d.respond(dec,0 if choice is None else choice)
  assert not d.log, d.log
  assert d.lp[0] == 9234, (d.lp, step)
  assert sum(n for (c,stage,_),n in d.coverage.items() if c==S and stage=='resolved') >= 2
  print('label conservé après déplacement et lu par le déclencheur')
 finally:d.close()
