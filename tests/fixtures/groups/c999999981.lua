local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE)
 e:SetCode(EVENT_FREE_CHAIN)
 e:SetOperation(s.operation)
 c:RegisterEffect(e)
end
function s.operation(e,tp)
 local g=Duel.GetMatchingGroup(nil,tp,LOCATION_HAND,0,nil)
 Duel.SendtoGrave(g,REASON_EFFECT)
 local moved=g:Filter(Card.IsLocation,nil,LOCATION_GRAVE)
 Duel.SendtoHand(moved,nil,REASON_EFFECT)
end
