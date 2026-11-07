"""Sonde native des raisons combinées, indépendante du moteur JAX.

Les cartes fictives ne servent qu'à établir la référence pour IsReason. La couverture
est relevée sur les effets réellement résolus ; aucune partie d'entraînement.
"""
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'oracle'))
import driver

PROBE,SPELL,NORMAL=600980001,600980002,600980003
monster=(0,0,33,500,500,4,1,1)
spell=(0,0,2,0,0,0,0,0)
normal=(0,0,17,1000,1000,4,1,1)
db=driver.CardDB()
db.add({PROBE:(monster,'Reason probe'),SPELL:(spell,'Reason action'),NORMAL:(normal,'Normal')})
probe='''local s,id=GetID()
function s.initial_effect(c)
  local masks={REASON_EFFECT,REASON_DESTROY,REASON_COST,REASON_RELEASE,REASON_BATTLE,REASON_REDIRECT}
  for i,mask in ipairs(masks) do
    local e=Effect.CreateEffect(c)
    e:SetDescription(aux.Stringid(id,i-1))
    e:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_F)
    e:SetCode(EVENT_TO_GRAVE)
    e:SetLabel(mask)
    e:SetCondition(function(e) return e:GetHandler():IsReason(e:GetLabel()) end)
    e:SetOperation(function(e,tp) Duel.Recover(tp,100,REASON_EFFECT) end)
    c:RegisterEffect(e)
  end
  if COST_TEST then
    local e=Effect.CreateEffect(c)
    e:SetDescription(aux.Stringid(id,6))
    e:SetType(EFFECT_TYPE_IGNITION)
    e:SetRange(LOCATION_MZONE)
    e:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
      if chk==0 then return e:GetHandler():IsReleasable() end
      Duel.Release(e:GetHandler(),REASON_COST)
    end)
    e:SetOperation(function() end)
    c:RegisterEffect(e)
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
CASES=[('destroy','Duel.Destroy(g,REASON_EFFECT)',False,{0,1}),
       ('send','Duel.SendtoGrave(g,REASON_EFFECT)',False,{0}),
       ('release_cost','',True,{2,3}),
       ('battle','',False,{1,4}),
       ('pendulum_redirect','Duel.Destroy(g,REASON_EFFECT)',False,{0,1})]
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder);driver.SCRIPTS.insert(0,folder)
    for name,action,cost,expected in CASES:
        is_pendulum=name=='pendulum_redirect'
        db.rows[PROBE]=(PROBE,)+(monster[:2]+(monster[2]|0x1000000,)+monster[3:] if is_pendulum else monster)
        source=probe.replace('COST_TEST','true' if cost else 'false')
        if is_pendulum:source=source.replace('EVENT_TO_GRAVE','EVENT_TO_DECK')
        (path/f'c{PROBE}.lua').write_text(source)
        (path/f'c{SPELL}.lua').write_text(spell.replace('ACTION',action or 'return'))
        for seed in (2600,2601):
            actual=set();actions=0
            deck=[PROBE,SPELL]*20 if action else [PROBE]*40
            duel=driver.Duel(db,deck,[NORMAL]*40,seed=seed)
            try:
                for step in range(1200):
                    dec=duel.next_decision()
                    if dec is None:break
                    for code,stage,desc in duel.coverage:
                        if code==PROBE and stage=='resolved' and desc is not None:
                            idx=int(desc)&0xfffff
                            if idx<6:actual.add(idx)
                    if duel.turn>=4 and actual.issuperset(expected):break
                    choice=None
                    for kind in ('chain','activate','summon','attack','to_bp','to_ep','pass'):
                        for i,option in enumerate(dec.options):
                            if option[0]!=kind:continue
                            if kind=='activate':
                                if cost and option[1]!=PROBE:continue
                                if action and option[1]!=SPELL:continue
                                if not cost and not action:continue
                                actions+=1
                            choice=i;break
                        if choice is not None:break
                    duel.respond(dec,0 if choice is None else choice)
                assert not duel.log,(name,duel.log)
            finally:duel.close()
            assert actual==expected,(name,seed,actual,expected,actions)
            if name!='battle':assert actions>0,(name,seed,actions)
        print(name,'2 parties, raisons observées',sorted(actual),flush=True)
