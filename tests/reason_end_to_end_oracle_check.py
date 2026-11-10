"""Raisons combinées : les effets sondes doivent correspondre à ygopro-core."""
from pathlib import Path
import sys,tempfile
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'oracle')]
import core,cards,driver,compare
PROBE,SPELL,NORMAL=600980101,600980102,600980103
rows={PROBE:((0,0,33,500,500,4,1,1),'Reason probe'),SPELL:((0,0,2,0,0,0,0,0),'Reason spell'),
      NORMAL:((0,0,17,1000,1000,4,1,1),'Normal')}
db=driver.CardDB();db.add(rows)
probe='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetDescription(aux.Stringid(id,0))
 e:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_F)
 e:SetCode(EVENT_TO_GRAVE)
 e:SetCondition(function(e) return e:GetHandler():IsReason(MASK) end)
 e:SetOperation(function(e,tp) Duel.Recover(tp,100,REASON_EFFECT) end)
 c:RegisterEffect(e)
 if COST_TEST then
  local e2=Effect.CreateEffect(c)
  e2:SetDescription(aux.Stringid(id,1))
  e2:SetType(EFFECT_TYPE_IGNITION)
  e2:SetRange(LOCATION_MZONE)
  e2:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
   if chk==0 then return e:GetHandler():IsReleasable() end
   Duel.Release(e:GetHandler(),REASON_COST)
  end)
  e2:SetOperation(function() end)
  c:RegisterEffect(e2)
 end
end
'''
spell='''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE)
 e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return Duel.IsExistingMatchingCard(aux.TRUE,tp,LOCATION_MZONE,0,1,nil) end
 end)
 e:SetOperation(function(e,tp)
  local g=Duel.GetMatchingGroup(aux.TRUE,tp,LOCATION_MZONE,0,nil)
  ACTION
 end)
 c:RegisterEffect(e)
end
'''
step,legal=jax.jit(core.step),jax.jit(core.legal)
cases=[('destroy','Duel.Destroy(g,REASON_EFFECT)',core.REASON_DESTROY,True),
       ('send','Duel.SendtoGrave(g,REASON_EFFECT)',core.REASON_DESTROY,False),
       ('release_cost','',core.REASON_COST,True),('battle','',core.REASON_BATTLE,True)]
with tempfile.TemporaryDirectory() as folder:
 driver.SCRIPTS.insert(0,folder);path=Path(folder)
 for name,action,mask,expected in cases:
  is_cost=name=='release_cost'
  p=probe.replace('MASK',str(mask)).replace('COST_TEST','true' if is_cost else 'false')
  (path/f'c{PROBE}.lua').write_text(p);(path/f'c{SPELL}.lua').write_text(spell.replace('ACTION',action or 'return'))
  probe_effects=[cards.trigger(cards.ev(core.EV_TO_GRAVE),cost=[cards.COND_REASON(mask)],resolve=[cards.RECOVER(cards.ME,100)])]
  if is_cost:probe_effects.append(cards.ignition(cost=[cards.SELF(0),cards.MOVE(0,reason=core.REASON_COST|core.REASON_RELEASE)]))
  op=cards.DESTROY(0,reason=core.REASON_EFFECT) if name=='destroy' else cards.MOVE(0,reason=core.REASON_EFFECT)
  spell_effect=[cards.act(target=[cards.FILTER(0,cards.F(me=core.Z_MZONE)),cards.REQ(0,1)],
                          resolve=[cards.FILTER(0,cards.F(me=core.Z_MZONE)),op])]
  codes=[PROBE,SPELL,NORMAL];effects={PROBE:probe_effects,SPELL:spell_effect,NORMAL:[]}
  bank=core.make_bank(np.array(codes),[500,0,1000],[500,0,1000],[4,0,4],ctype=[33,2,17],race=[1,0,1],attr=[1,0,1],
    **cards.assemble(codes,cards=effects,ritual={},fusion={},synchro={},xyz={},link={}))
  compare.EFFECT_KINDS.clear();compare.EFFECT_KINDS.update({code:[e[0] for e in eff] for code,eff in effects.items()})
  cmap={c:i for i,c in enumerate(codes)}
  for seed in (2700,2701):
   def chooser(dec,indices):
    for kind in ('chain','activate','summon','attack','to_bp','to_ep','pass'):
     for i in indices:
      option=dec.options[i]
      if option[0]!=kind:continue
      if kind=='activate' and option[1]!=(PROBE if is_cost else SPELL):continue
      if kind=='activate' and not action and not is_cost:continue
      if kind=='attack' and name!='battle':continue
      return i
    return indices[0]
   deck=[PROBE,SPELL]*20 if action else [PROBE]*40
   result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,given=([deck,[NORMAL]*40],[[],[]]))
   assert result['ok'],(name,seed,result.get('err'))
   resolved=sum(e['count'] for e in result['coverage'] if e['code']==PROBE and e['stage']=='resolved' and (int(e['description'])&0xfffff)==0)
   assert (resolved>0)==expected,(name,seed,resolved)
   print(name,seed,'identique, condition résolue',resolved,'fois',flush=True)
