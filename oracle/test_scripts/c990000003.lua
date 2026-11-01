--Test Link L-T (carte fictive de test, voir TEST_CARDS dans driver.py)
local s,id=GetID()
function s.initial_effect(c)
	c:EnableReviveLimit()
	Link.AddProcedure(c,nil,2,2)
end
