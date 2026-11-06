-- Deux cartes ayant la propriété de substitut sont permises si une seule remplace un nom.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,true,true,aux.FilterBoolFunctionEx(Card.IsType,TYPE_EFFECT),34460851)
end
