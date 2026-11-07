local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE)
 e:SetCode(EVENT_FREE_CHAIN)
 e:SetOperation(s.operation)
 c:RegisterEffect(e)
end
function s.operation(e,tp)
 local g=Duel.GetMatchingGroup(Card.IsCode,tp,LOCATION_HAND,0,nil,1)
 if #g>0 then Duel.SendtoGrave(g,REASON_EFFECT) end
 Duel.Draw(tp,1,REASON_EFFECT)
end
