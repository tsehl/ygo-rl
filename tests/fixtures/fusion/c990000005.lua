-- Test de règle uniquement : trois exemplaires d'un même matériau.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMixN(c,false,true,89631139,3)
end
