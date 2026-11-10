"""Une cible indestructible reste liée si la destruction échoue."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import driver
S,M,N=600930091,600930092,600930093
db=driver.CardDB();db.add({S:((0,0,2,0,0,0,0,0),'Target'),M:((0,0,33,1000,1000,4,1,1),'Protected'),N:((0,0,17,1000,1000,4,1,1),'Normal')})
spell='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chk==0 then return Duel.IsExistingTarget(Card.IsMonster,tp,LOCATION_MZONE,0,1,nil) end
  local g=Duel.SelectTarget(tp,Card.IsMonster,tp,LOCATION_MZONE,0,1,1,nil)
  Duel.Destroy(g,REASON_EFFECT)
 end)
 e:SetOperation(function(e,tp)
  local tc=Duel.GetFirstTarget()
  if tc and tc:IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''
monster='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_SINGLE) e:SetCode(EFFECT_INDESTRUCTABLE_EFFECT) e:SetValue(1)
 c:RegisterEffect(e)
end'''
with tempfile.TemporaryDirectory() as folder:
 Path(folder,f'c{S}.lua').write_text(spell);Path(folder,f'c{M}.lua').write_text(monster);driver.SCRIPTS.insert(0,folder)
 d=driver.Duel(db,[S,M]*20,[N]*40,seed=3400)
 try:
  for step in range(35):
   dec=d.next_decision()
   if d.lp[0]!=8000:break
   choice=None
   for typ in ('activate','summon','to_ep','pass'):
    choice=next((i for i,o in enumerate(dec.options) if o[0]==typ and (typ not in ('activate','summon') or o[1] in (S,M))),None)
    if choice is not None:break
   d.respond(dec,0 if choice is None else choice)
  assert not d.log, d.log
  assert d.lp[0] == 9234, (d.lp, step)
  assert any(c == S and stage == 'resolved' and n for (c,stage,_),n in d.coverage.items())
  print('cible indestructible : relation conservée, LP 9234')
 finally:d.close()
