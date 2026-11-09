"""Référence native : la première cible reste la première si elle quitte le terrain."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
import driver
S,N,M=600930051,44287299,5318639
db=driver.CardDB();db.add({S:((0,0,2,0,0,0,0,0),'Target relation')})
script='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chkc then return chkc:IsControler(tp) and chkc:IsLocation(LOCATION_SZONE) and chkc~=e:GetHandler() end
  if chk==0 then return Duel.IsExistingTarget(Card.IsSpellTrap,tp,LOCATION_SZONE,0,1,e:GetHandler()) end
  Duel.SelectTarget(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,2,2,e:GetHandler())
 end)
 e:SetOperation(function(e,tp)
  local tc=Duel.GetFirstTarget()
  if tc and tc:IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''
with tempfile.TemporaryDirectory() as folder:
 Path(folder,f'c{S}.lua').write_text(script);driver.SCRIPTS.insert(0,folder)
 for hit in (0,1):
  interrupt=True
  d=driver.Duel(db,[S,M,M,N]*10,[M,N]*20,seed=3400); fired=replied=False
  try:
   for step in range(100):
    dec=d.next_decision()
    if any(c==S and stage=='resolved' and n for (c,stage,_),n in d.coverage.items()):break
    def find(k,code):return next((i for i,o in enumerate(dec.options) if o[0]==k and o[1]==code),None)
    choice=None
    if (d.turn==1 and dec.player==0) or (interrupt and d.turn==2 and dec.player==1):choice=find('sset',M)
    if d.turn>=3 and dec.player==0 and not fired:
     choice=find('activate',S)
     if choice is not None:fired=True
    if fired and interrupt and not replied and dec.player==1:
     choice=find('chain',M)
     if choice is not None:replied=True
    if choice is None and dec.options[0][0]=='cards':
     candidates=[i for i,o in enumerate(dec.options) if any(c[0]==M and c[1]==0 for c in o[1])]
     choice=(candidates[min(hit,len(candidates)-1)] if candidates and dec.player==1 else (candidates[0] if candidates else None))
    if choice is None:choice=next((i for i,o in enumerate(dec.options) if o[0] in ('pass','to_ep')),0)
    d.respond(dec,choice)
   assert fired and replied, (fired, replied)
   assert not d.log, d.log
   assert d.lp[0] == (9234 if hit == 0 else 8000), (hit, d.lp, step)
   for code in (S, M):
    assert any(c == code and stage == 'resolved' and n for (c, stage, _), n in d.coverage.items())
   print('cible détruite', hit, 'LP', d.lp[0])
  finally:d.close()
