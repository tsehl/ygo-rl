-- Test de règle uniquement : deux matériaux nommés distincts, sans substitut.
local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,false,true,34460851,44287299)
end
