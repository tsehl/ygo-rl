--Bibliothèque des cartes procédurales (jalon 6, étape 2) : une carte générée est une spécification (table Lua écrite
--par gen/tolua.py depuis gen/cardgen.py) ; PG.init la traduit en effets EDOPro. Chaque gabarit reprend un script
--officiel déjà vérifié contre le moteur JAX (cité en commentaire). Les constantes de la spécification sont celles du
--moteur (engine/core.py) : zones = LOCATION_*, positions = POS_*, types = TYPE_*, phases = PHASE_* (PH_BATTLE = toute
--la phase de combat), événements EV_* (bits du masque `on`).
--Cartes de test de règle du moteur : jamais dans l'entraînement (les cartes générées n'y entrent que comme banque).
PG={}
local BIG=1<<20
local ME,OPP=0,1
--EV_* du moteur → code d'événement ygopro
local EVENT={[0]=EVENT_SUMMON_SUCCESS,[1]=EVENT_FLIP_SUMMON_SUCCESS,[2]=EVENT_SPSUMMON_SUCCESS,
	[3]=EVENT_ATTACK_ANNOUNCE,[4]=EVENT_TO_GRAVE,[5]=EVENT_BATTLE_DESTROYED,[6]=EVENT_FLIP}
local EV_FLIP=6
local OPT_NAME,OPT_CARD=1,2
--statuts du moteur (ST_*)
local ST_SUMMON_TURN,ST_FLIP_TURN,ST_SPSUM,ST_LINKSUM=1,4096,8192,16384
--codes des effets continus (MC_*)
local MC_ATK,MC_DEF,MC_NO_ATTACK,MC_NO_BTARGET,MC_NO_ETARGET,MC_IMM_TRAP,MC_NO_SS,MC_INDES_BATTLE,MC_P_NO_TRAP=1,5,9,10,11,12,13,14,15
local MC_DISABLE,MC_NO_DIRECT,MC_INDES_EFF=16,17,18
local RIT_GREATER=1

local function who(w,tp) return w==ME and tp or 1-tp end
local function bounded(r) return r~=nil and (r[1]>-BIG or r[2]<BIG) end

function PG.status(c,st)
	if st<0 then
		local packed=-st
		if packed&(1<<30)~=0 then return c:IsReason(packed&0x1fffffff) end
		local mask=((packed&255)<<24)|((packed>>8)&65535)
		local matched=c:IsSummonType(mask)
		if packed&(1<<24)~=0 then return not matched end
		return matched
	end
	if st&ST_SUMMON_TURN~=0 and c:IsStatus(STATUS_SUMMON_TURN+STATUS_FLIP_SUMMON_TURN+STATUS_SPSUMMON_TURN) then return true end
	if st&ST_FLIP_TURN~=0 and c:IsStatus(STATUS_FLIP_SUMMON_TURN) then return true end
	if st&ST_SPSUM~=0 and c:IsSpecialSummoned() then return true end
	if st&ST_LINKSUM~=0 and c:IsLinkSummoned() then return true end
	return false
end

--phases du moteur (masque) : PH_BATTLE (8) = toute la phase de combat
function PG.phase(ph)
	if ph&PHASE_BATTLE_START~=0 and Duel.IsBattlePhase() then return true end
	return Duel.GetCurrentPhase()&ph&~PHASE_BATTLE_START~=0
end

--Filtre F du moteur (core.filter_mask) pour le joueur tp, effet e (carte source e:GetHandler()) ; noloc : sans les zones
function PG.match(c,f,e,tp,noloc)
	if not noloc then
		local loc=c:IsControler(tp) and (f.me or 0) or (f.op or 0)
		if not c:IsLocation(loc) then return false end
	end
	if f.pos and not c:IsPosition(f.pos) then return false end
	if f.typ and not c:IsType(f.typ) then return false end
	if f.not_typ and c:IsType(f.not_typ) then return false end
	if f.not_self and c==e:GetHandler() then return false end
	if f.reason then
		if f.reason>0 and not c:IsReason(f.reason) then return false end
		if f.reason<0 then
			local bits=-f.reason
			for i=0,28 do if bits&(1<<i)~=0 and not c:IsReason(1<<i) then return false end end
		end
	end
	if f.attr and not c:IsAttribute(f.attr) then return false end
	if f.race and not c:IsRace(f.race) then return false end
	if f.setc and not c:IsSetCard(f.setc) then return false end
	if f.code and not c:IsCode(f.code[1],f.code[2]) then return false end
	if f.st and not PG.status(c,f.st) then return false end
	if bounded(f.lv) then
		if c:IsType(TYPE_XYZ|TYPE_LINK) then return false end
		local lv=c:GetLevel()
		if lv<f.lv[1] or lv>f.lv[2] then return false end
	end
	if f.spsum==1 and not c:IsCanBeSpecialSummoned(e,0,tp,false,false) then return false end
	if f.spsum==2 and not c:IsCanBeSpecialSummoned(e,0,tp,false,false,POS_FACEUP_DEFENSE) then return false end
	if f.nodis and c:IsDisabled() then return false end
	if f.atk then
		local a=c:GetAttack()
		if a<f.atk[1] or a>f.atk[2] then return false end
	end
	if bounded(f.df) then
		if c:IsType(TYPE_LINK) then return false end
		local d=c:GetDefense()
		if d<f.df[1] or d>f.df[2] then return false end
	end
	return true
end

local function fn(f,e,tp,tg)
	return function(c) return PG.match(c,f,e,tp) and (not tg or c:IsCanBeEffectTarget(e)) end
end
local function group(f,e,tp,tg) return Duel.GetMatchingGroup(fn(f,e,tp,tg),tp,f.me or 0,f.op or 0,nil) end

--Filtre d'un matériau (recettes de fusion / synchro / Xyz / Link : attribut, type, archétype, nom, niveau)
function PG.mat(m)
	if m.code and m.code[1]==m.code[2] and not (m.attr or m.race or m.setc or m.lv or m.typ or m.not_typ) then return m.code[1] end
	if not (m.attr or m.race or m.setc or m.code or m.typ or m.not_typ or bounded(m.lv)) then return nil end
	return function(c,sc,st,tp)
		if tp==nil then sc,st,tp=nil,nil,nil end  --appel sans contexte d'invocation (Synchro.NonTuner…)
		if m.typ and not c:IsType(m.typ) then return false end
		if m.not_typ and c:IsType(m.not_typ) then return false end
		if m.attr and not (tp and c:IsAttribute(m.attr,sc,st,tp) or not tp and c:IsAttribute(m.attr)) then return false end
		if m.race and not (tp and c:IsRace(m.race,sc,st,tp) or not tp and c:IsRace(m.race)) then return false end
		if m.setc and not (tp and c:IsSetCard(m.setc,sc,st,tp) or not tp and c:IsSetCard(m.setc)) then return false end
		if m.code and not c:IsCode(m.code[1],m.code[2]) then return false end
		if bounded(m.lv) then
			if c:IsType(TYPE_XYZ|TYPE_LINK) then return false end
			local lv=c:GetLevel()
			if lv<m.lv[1] or lv>m.lv[2] then return false end
		end
		return true
	end
end

-------------------------------------------------------------------- conditions et coûts

function PG.cond(sp)
	local conds,dmg=sp.conds or {},sp.dmg or 0
	if #conds==0 and dmg==0 then return nil end
	return function(e,tp,eg,ep,ev,re,r,rp)
		local c=e:GetHandler()
		for _,k in ipairs(conds) do
			local t=k[1]
			if t=="turn" then
				if not Duel.IsTurnPlayer(who(k[2],tp)) then return false end
			elseif t=="phase" then
				if not PG.phase(k[2]) then return false end
			elseif t=="self" then  --Sangan (IsPreviousLocation), Mystic Tomato (IsLocation), Knightmare (IsLinkSummoned)
				if k[2]~=0 and not c:IsLocation(k[2]) then return false end
				if k[3]~=0 and not c:IsPreviousLocation(k[3]) then return false end
				if k[4]~=0 and not PG.status(c,k[4]) then return false end
			elseif t=="evp" then
				if ep~=who(k[3],tp) then return false end
			elseif t=="chain" then  --Magic Jammer
				if not (re:IsHasType(EFFECT_TYPE_ACTIVATE) and re:IsActiveType(k[2]) and Duel.IsChainNegatable(ev)) then return false end
			elseif t=="chain_field" then  --Ghost Ogre
				if not (re:GetHandler():IsOnField() and (re:IsMonsterEffect()
					or (re:IsSpellTrapEffect() and not re:IsHasType(EFFECT_TYPE_ACTIVATE)))) then return false end
			end
		end
		if dmg~=0 and not aux.StatChangeDamageStepCondition() then return false end
		return true
	end
end

local function discardable_other(e)
	return function(c) return c:IsDiscardable() and c~=e:GetHandler() end
end
local function gy_mon(c) return c:IsMonster() and c:IsAbleToRemoveAsCost() end

function PG.cost(cs)
	if not cs then return nil end
	local t=cs[1]
	if t=="lp" then
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			if chk==0 then return Duel.CheckLPCost(tp,cs[2]) end
			Duel.PayLPCost(tp,cs[2])
		end
	elseif t=="discard" then  --Magic Jammer
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			if chk==0 then return Duel.IsExistingMatchingCard(Card.IsDiscardable,tp,LOCATION_HAND,0,1,e:GetHandler()) end
			Duel.DiscardHand(tp,discardable_other(e),1,1,REASON_COST|REASON_DISCARD)
		end
	elseif t=="self" then  --Effect Veiler (main : Cost.SelfToGrave), sinon libération
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			local c=e:GetHandler()
			if c:IsLocation(LOCATION_HAND) then
				if chk==0 then return c:IsAbleToGraveAsCost() end
				Duel.SendtoGrave(c,REASON_COST)
			else
				if chk==0 then return c:IsReleasable() end
				Duel.Release(c,REASON_COST)
			end
		end
	elseif t=="tribute" then  --Cannon Soldier, sans la carte elle-même
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			local c=e:GetHandler()
			if chk==0 then return Duel.CheckReleaseGroupCost(tp,nil,1,false,nil,c) end
			local sg=Duel.SelectReleaseGroupCost(tp,nil,1,1,false,nil,c)
			Duel.Release(sg,REASON_COST)
		end
	elseif t=="banish_gy" then
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			if chk==0 then return Duel.IsExistingMatchingCard(gy_mon,tp,LOCATION_GRAVE,0,1,nil) end
			local g=Duel.SelectMatchingCard(tp,gy_mon,tp,LOCATION_GRAVE,0,1,1,nil)
			Duel.Remove(g,POS_FACEUP,REASON_COST)
		end
	elseif t=="detach" then
		return function(e,tp,eg,ep,ev,re,r,rp,chk)
			local c=e:GetHandler()
			if chk==0 then return c:CheckRemoveOverlayCard(tp,cs[2],REASON_COST) end
			c:RemoveOverlayCard(tp,cs[2],cs[2],REASON_COST)
		end
	end
	error("procgen : coût inconnu "..tostring(t))
end

-------------------------------------------------------------------- verbes

--Applique le verbe de la clause cl au groupe g ; rend vrai si la clause « a touché une carte » (condition du « puis »)
function PG.verb(cl,g,e,tp)
	local v=cl.verb
	if v=="destroy" then
		return Duel.Destroy(g,REASON_EFFECT)>0  --le registre ne garde que les cartes détruites
	end
	if #g==0 then return false end
	if v=="banish" then
		Duel.Remove(g,POS_FACEUP,REASON_EFFECT)
	elseif v=="bounce" or v=="tohand" then
		Duel.SendtoHand(g,nil,REASON_EFFECT)
		if cl.lock then  --Sangan : plus d'activation des cartes de ce nom ce tour
			local tc=g:GetFirst()
			if tc:IsLocation(LOCATION_HAND) then
				local e1=Effect.CreateEffect(e:GetHandler())
				e1:SetType(EFFECT_TYPE_FIELD)
				e1:SetProperty(EFFECT_FLAG_PLAYER_TARGET)
				e1:SetCode(EFFECT_CANNOT_ACTIVATE)
				e1:SetTargetRange(1,0)
				e1:SetValue(function(e,re,tp) return re:GetHandler():IsCode(e:GetLabel()) end)
				e1:SetLabel(tc:GetCode())
				e1:SetReset(RESET_PHASE|PHASE_END)
				Duel.RegisterEffect(e1,tp)
			end
		end
	elseif v=="tograve" then
		Duel.SendtoGrave(g,REASON_EFFECT)
	elseif v=="ss" then
		Duel.SpecialSummon(g,cl.tag or 0,tp,tp,false,false,cl.pos)
	elseif v=="atk" or v=="def" or v=="halve_base" or v=="halve" or v=="indes_battle" then
		--Rush Recklessly, Little Swordsman (rd), Number 17 (rd + perm), Castle Walls, Shrink, Reverse Glasses
		local c=e:GetHandler()
		local reset=RESETS_STANDARD_PHASE_END
		if cl.perm then reset=RESET_EVENT|RESETS_STANDARD_DISABLE
		elseif cl.rd then reset=RESETS_STANDARD_DISABLE_PHASE_END end
		for tc in g:Iter() do
			local e1=Effect.CreateEffect(c)
			e1:SetType(EFFECT_TYPE_SINGLE)
			if v=="atk" then e1:SetCode(EFFECT_UPDATE_ATTACK) e1:SetValue(cl.v)
			elseif v=="def" then e1:SetCode(EFFECT_UPDATE_DEFENSE) e1:SetValue(cl.v)
			elseif v=="indes_battle" then e1:SetCode(EFFECT_INDESTRUCTABLE_BATTLE) e1:SetValue(1)
			elseif v=="halve_base" then e1:SetCode(EFFECT_SET_BASE_ATTACK) e1:SetValue(tc:GetBaseAttack()/2)
			else e1:SetCode(EFFECT_SET_ATTACK_FINAL) e1:SetValue(tc:GetAttack()/2) end
			e1:SetReset(reset)
			tc:RegisterEffect(e1)
		end
	elseif v=="disable" then  --Effect Veiler
		local c=e:GetHandler()
		for tc in g:Iter() do
			Duel.NegateRelatedChain(tc,RESET_TURN_SET)
			local e1=Effect.CreateEffect(c)
			e1:SetType(EFFECT_TYPE_SINGLE)
			e1:SetCode(EFFECT_DISABLE)
			e1:SetReset(RESETS_STANDARD_PHASE_END)
			tc:RegisterEffect(e1)
			local e2=Effect.CreateEffect(c)
			e2:SetType(EFFECT_TYPE_SINGLE)
			e2:SetCode(EFFECT_DISABLE_EFFECT)
			e2:SetValue(RESET_TURN_SET)
			e2:SetReset(RESETS_STANDARD_PHASE_END)
			tc:RegisterEffect(e2)
		end
	else
		error("procgen : verbe inconnu "..tostring(v))
	end
	return true
end

-------------------------------------------------------------------- clauses
--Sémantique (gen/cardgen.py, clause_bricks) :
--  target : à l'activation, case libre si invocation, puis k cibles ; à la résolution, case libre (sinon l'effet
--           s'arrête), cibles liées qui vérifient encore le filtre ;
--  all    : au moins une carte à l'activation ; à la résolution, toutes les cartes du filtre ;
--  pick   : au moins k cartes à l'activation ; à la résolution, choix de min(k, nombre) cartes ;
--  event  : cartes de l'événement (ciblées, Trap Hole) ; à la résolution, celles qui vérifient encore le filtre ;
--  self   : la carte elle-même.

local function free(tp) return Duel.GetLocationCount(tp,LOCATION_MZONE)>0 end

--vérification à l'activation (chk==0) de la clause cl (first : première clause)
local function check(cl,first,e,tp,eg)
	local v,sel,f=cl.verb,cl.sel,cl.f
	if v=="draw" then return Duel.GetFieldGroupCount(tp,LOCATION_DECK,0)>=cl.n end
	if v=="damage" or v=="recover" or v=="negate" then return true end
	if v=="burn_per" then return #group(f,e,tp)>=1 end
	if v=="chain_destroy" then return true end  --vérifié à part (re)
	if not first then return true end
	if v=="ss" and sel~="event" and sel~="all" and not free(tp) then return false end
	if sel=="target" then return Duel.IsExistingTarget(fn(f,e,tp),tp,f.me or 0,f.op or 0,cl.k,nil) end
	if sel=="all" then return #group(f,e,tp)>=1 end
	if sel=="pick" then return #group(f,e,tp)>=cl.k end
	if sel=="event" then return eg~=nil and eg:IsExists(fn(f,e,tp,true),1,nil) end
	if sel=="self" then return PG.match(e:GetHandler(),f,e,tp) end
	if sel=="min" then return #group(f,e,tp)>=1 end
	error("procgen : sélecteur inconnu "..tostring(sel))
end

--à l'activation (chk==1) : choix des cibles
local function choose(cl,e,tp,eg)
	local f=cl.f
	if cl.sel=="target" then
		Duel.SelectTarget(tp,fn(f,e,tp),tp,f.me or 0,f.op or 0,cl.k,cl.k,nil)
	elseif cl.sel=="event" then
		Duel.SetTargetCard(eg:Filter(fn(f,e,tp,true),nil))
	end
end

--résolution : rend (continuer, touché)
local function resolve(cl,first,e,tp,eg,ep,ev,re)
	local v,sel,f=cl.verb,cl.sel,cl.f
	if v=="draw" then Duel.Draw(tp,cl.n,REASON_EFFECT) return true,true end
	if v=="damage" then Duel.Damage(1-tp,cl.v,REASON_EFFECT) return true,true end
	if v=="recover" then Duel.Recover(tp,cl.v,REASON_EFFECT) return true,true end
	if v=="burn_per" then  --Just Desserts
		Duel.Damage(1-tp,cl.v*#group(f,e,tp),REASON_EFFECT)
		return true,true
	end
	if v=="negate" then  --Magic Jammer
		if Duel.NegateActivation(ev) and re:GetHandler():IsRelateToEffect(re) then
			Duel.Destroy(eg,REASON_EFFECT)
		end
		return true,true
	end
	if v=="chain_destroy" then  --Ghost Ogre
		if re:GetHandler():IsRelateToEffect(re) then Duel.Destroy(eg,REASON_EFFECT) end
		return true,true
	end
	if v=="ss" and sel~="all" and not free(tp) then return false,false end
	local g
	if sel=="target" and cl.first_only then
		local tc=Duel.GetFirstTarget()
		if not tc or not tc:IsRelateToEffect(e) then return false,false end
		g=Group.FromCards(tc):Filter(fn(f,e,tp),nil)
	elseif sel=="target" or sel=="event" then
		g=(Duel.GetTargetCards(e) or Group.CreateGroup()):Filter(fn(f,e,tp),nil)
	elseif sel=="all" then
		g=group(f,e,tp)
	elseif sel=="pick" then
		local all=group(f,e,tp)
		local n=math.min(cl.k,#all)
		g=n>0 and all:Select(tp,n,n,nil) or Group.CreateGroup()
	elseif sel=="min" then  --Fissure
		local all=group(f,e,tp)
		if #all==0 then return true,false end
		g=all:GetMinGroup(Card.GetAttack)
		if #g>1 then g=g:Select(tp,1,1,nil) end
	elseif sel=="self" then
		local c=e:GetHandler()
		g=(c:IsRelateToEffect(e) and PG.match(c,f,e,tp)) and Group.FromCards(c) or Group.CreateGroup()
	end
	return true,PG.verb(cl,g,e,tp)
end

function PG.target(sp)
	local cls=sp.clauses
	local c1=cls[1]
	return function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
		if chkc then
			return c1.sel=="target" and PG.match(chkc,c1.f,e,tp)
		end
		if chk==0 then
			if c1.verb=="chain_destroy" and not re:GetHandler():IsDestructable() then return false end
			for i,cl in ipairs(cls) do
				if not check(cl,i==1,e,tp,eg) then return false end
			end
			return true
		end
		if c1.sel then choose(c1,e,tp,eg) end
	end
end

function PG.operation(sp)
	local cls=sp.clauses
	return function(e,tp,eg,ep,ev,re,r,rp)
		if sp.related and not e:GetHandler():IsRelateToEffect(e) then return end
		local go,hit=resolve(cls[1],true,e,tp,eg,ep,ev,re)
		if not go then return end
		if cls[2] and hit then resolve(cls[2],false,e,tp,eg,ep,ev,re) end
	end
end

-------------------------------------------------------------------- effets activables

local function base(c,sp,id,idx)
	local e=Effect.CreateEffect(c)
	e:SetDescription(aux.Stringid(id,idx))  --effet idx du moteur (engine/compare.py translate)
	local prop=0
	local c1=sp.clauses and sp.clauses[1]
	if c1 and (c1.sel=="target" or c1.sel=="event") then prop=prop|EFFECT_FLAG_CARD_TARGET end
	if (sp.dmg or 0)~=0 then prop=prop|EFFECT_FLAG_DAMAGE_STEP end
	if sp.kind=="trigger" and sp.optional then prop=prop|EFFECT_FLAG_DELAY end
	if prop~=0 then e:SetProperty(prop) end
	if sp.opt==OPT_NAME then
		e:SetCountLimit(1,id,sp.refund and EFFECT_COUNT_CODE_OATH or 0)
	elseif sp.opt==OPT_CARD then e:SetCountLimit(1) end
	local cond=PG.cond(sp)
	if cond then e:SetCondition(cond) end
	local cost=PG.cost(sp.cost)
	if cost then e:SetCost(cost) end
	if sp.clauses and #sp.clauses>0 then
		e:SetTarget(PG.target(sp))
		e:SetOperation(PG.operation(sp))
	end
	return e
end

local function has_chain(sp)
	for _,k in ipairs(sp.conds or {}) do
		if k[1]=="chain" or k[1]=="chain_field" then return true end
	end
	return false
end

local function events(on)
	local out={}
	for i=0,7 do if on&(1<<i)~=0 then out[#out+1]=i end end
	return out
end

function PG.effect(c,sp,id,idx)
	local k=sp.kind
	if k=="act" then
		local e=base(c,sp,id,idx)
		e:SetType(EFFECT_TYPE_ACTIVATE)
		if (sp.on or 0)==0 then
			e:SetCode(has_chain(sp) and EVENT_CHAINING or EVENT_FREE_CHAIN)
			c:RegisterEffect(e)
		else  --un clone par événement (Trap Hole : SUMMON_SUCCESS puis FLIP_SUMMON_SUCCESS)
			for i,ev in ipairs(events(sp.on)) do
				local ei=i==1 and e or e:Clone()
				ei:SetCode(EVENT[ev])
				c:RegisterEffect(ei)
			end
		end
	elseif k=="hand_proc" then
		local payment=PG.cost(sp.cost)
		local e=Effect.CreateEffect(c)
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(EFFECT_SPSUMMON_PROC)
		e:SetProperty(EFFECT_FLAG_UNCOPYABLE)
		e:SetRange(LOCATION_HAND)
		if sp.opt==OPT_NAME then e:SetCountLimit(1,id,EFFECT_COUNT_CODE_OATH) end
		e:SetCondition(function(e,c)
			if c==nil then return true end
			local tp=c:GetControler()
			if Duel.GetLocationCount(tp,LOCATION_MZONE)<=0 then return false end
			if payment and not payment(e,tp,nil,nil,nil,nil,nil,nil,0) then return false end
			for _,check in ipairs(sp.counts) do
				local n=group(check[1],e,tp):GetCount()
				if n<check[2] or n>check[3] then return false end
			end
			return true
		end)
		if payment then
			e:SetOperation(function(e,tp) payment(e,tp,nil,nil,nil,nil,nil,nil,1) end)
		end
		c:RegisterEffect(e)
	elseif k=="ignition" then
		local e=base(c,sp,id,idx)
		e:SetType(EFFECT_TYPE_IGNITION)
		e:SetRange(LOCATION_MZONE)
		c:RegisterEffect(e)
	elseif k=="trigger" then
		local e=base(c,sp,id,idx)
		local ev=events(sp.on)[1]
		if ev==EV_FLIP then  --Man-Eater Bug (obligatoire), Penguin Soldier (optionnel)
			e:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_FLIP+(sp.optional and EFFECT_TYPE_TRIGGER_O or 0))
		else
			e:SetType(EFFECT_TYPE_SINGLE+(sp.optional and EFFECT_TYPE_TRIGGER_O or EFFECT_TYPE_TRIGGER_F))
			e:SetCode(EVENT[ev])
		end
		c:RegisterEffect(e)
	elseif k=="quick" then
		local e=base(c,sp,id,idx)
		e:SetType(EFFECT_TYPE_QUICK_O)
		e:SetCode(has_chain(sp) and EVENT_CHAINING or EVENT_FREE_CHAIN)
		e:SetRange(sp.rng)
		c:RegisterEffect(e)
	elseif k=="cont" then
		PG.cont(c,sp)
	elseif k=="pend" then
		Pendulum.AddProcedure(c)
	elseif k=="ritual" then  --Black Luster Ritual (Greater), End of the World (Equal)
		if sp.mode==RIT_GREATER then Ritual.AddProcGreaterCode(c,sp.lv,nil,sp.code)
		else Ritual.AddProcEqualCode(c,nil,nil,sp.code) end
	elseif k=="fusion_spell" then  --Polymerization
		local sc=sp.f.setc
		Fusion.RegisterSummonEff(c,sc and function(fc) return fc:IsSetCard(sc) end or nil)
	else
		error("procgen : genre d'effet inconnu "..tostring(k))
	end
end

-------------------------------------------------------------------- effets continus

local function cont_cond(cs)
	if not (cs.cnt or (cs.tp or 0)~=0 or (cs.ph or 0)~=0) then return nil end
	return function(e)
		local tp=e:GetHandlerPlayer()
		if cs.cnt then  --Command Knight
			local f=cs.cnt[1]
			local n=Duel.GetMatchingGroupCount(fn(f,e,tp),tp,f.me or 0,f.op or 0,nil)
			if n<cs.cnt[2] or n>cs.cnt[3] then return false end
		end
		if (cs.tp or 0)~=0 and not Duel.IsTurnPlayer(who(cs.tp-1,tp)) then return false end  --Banner of Courage
		if (cs.ph or 0)~=0 and not PG.phase(cs.ph) then return false end
		return true
	end
end

local function tfilter(f) return function(e,c) return PG.match(c,f,e,e:GetHandlerPlayer()) end end
local function trange(e,f) e:SetTargetRange((f.me or 0)&LOCATION_MZONE,(f.op or 0)&LOCATION_MZONE) end

function PG.cont(c,sp)
	local cs=sp.cont
	local code=cs.code
	local e=Effect.CreateEffect(c)
	e:SetRange(sp.rng)
	local cond=cont_cond(cs)
	if cond then e:SetCondition(cond) end
	if code==MC_ATK or code==MC_DEF then
		e:SetCode(code==MC_ATK and EFFECT_UPDATE_ATTACK or EFFECT_UPDATE_DEFENSE)
		if cs.tgt then  --Banner of Courage, Command Knight
			e:SetType(EFFECT_TYPE_FIELD)
			trange(e,cs.tgt)
			e:SetTarget(tfilter(cs.tgt))
		else
			e:SetType(EFFECT_TYPE_SINGLE)
			e:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
		end
		if cs.per then
			local f=cs.per
			e:SetValue(function(ef,tc)
				local tp=ef:GetHandlerPlayer()
				return cs.val*Duel.GetMatchingGroupCount(fn(f,ef,tp),tp,f.me or 0,f.op or 0,nil)
			end)
		else
			e:SetValue(cs.val)
		end
	elseif code==MC_NO_ATTACK or code==MC_NO_DIRECT then  --Gravity Bind, Big-Tusked Mammoth
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(code==MC_NO_ATTACK and EFFECT_CANNOT_ATTACK or EFFECT_CANNOT_DIRECT_ATTACK)
		trange(e,cs.tgt)
		e:SetTarget(tfilter(cs.tgt))
	elseif code==MC_NO_BTARGET then  --Command Knight
		e:SetType(EFFECT_TYPE_SINGLE)
		e:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
		e:SetCode(EFFECT_CANNOT_BE_BATTLE_TARGET)
		e:SetValue(aux.imval1)
	elseif code==MC_INDES_BATTLE then  --Obnoxious Celtic Guard
		e:SetType(EFFECT_TYPE_SINGLE)
		e:SetCode(EFFECT_INDESTRUCTABLE_BATTLE)
		if cs.val==0 then e:SetValue(1)
		else e:SetValue(function(ef,bc) return bc:GetAttack()>=cs.val end) end
	elseif code==MC_NO_ETARGET then  --Lord of D.
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(EFFECT_CANNOT_BE_EFFECT_TARGET)
		e:SetProperty(EFFECT_FLAG_IGNORE_IMMUNE)
		trange(e,cs.tgt)
		e:SetTarget(tfilter(cs.tgt))
		e:SetValue(1)
	elseif code==MC_IMM_TRAP then  --Wildheart
		e:SetType(EFFECT_TYPE_SINGLE)
		e:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
		e:SetCode(EFFECT_IMMUNE_EFFECT)
		e:SetValue(function(ef,te) return te:IsTrapEffect() end)
	elseif code==MC_NO_SS then  --Quarantine
		local f=cs.tgt
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(EFFECT_CANNOT_SPECIAL_SUMMON)
		e:SetProperty(EFFECT_FLAG_PLAYER_TARGET)
		e:SetTargetRange((f.me or 0)~=0 and 1 or 0,(f.op or 0)~=0 and 1 or 0)
		e:SetTarget(function(ef,tc) return PG.match(tc,f,ef,ef:GetHandlerPlayer(),true) end)
	elseif code==MC_P_NO_TRAP then  --Mirage Dragon
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(EFFECT_CANNOT_ACTIVATE)
		e:SetProperty(EFFECT_FLAG_PLAYER_TARGET)
		e:SetTargetRange(cs.val&1,(cs.val>>1)&1)
		e:SetValue(function(ef,re,tp) return re:GetHandler():IsTrap() and re:IsHasType(EFFECT_TYPE_ACTIVATE) end)
	elseif code==MC_DISABLE then  --Skill Drain
		e:SetType(EFFECT_TYPE_FIELD)
		e:SetCode(EFFECT_DISABLE)
		trange(e,cs.tgt)
		e:SetTarget(function(ef,tc) return tc:IsType(TYPE_EFFECT) or (tc:GetOriginalType()&TYPE_EFFECT)==TYPE_EFFECT end)
	elseif code==MC_INDES_EFF then
		e:SetCode(EFFECT_INDESTRUCTABLE_EFFECT)
		e:SetValue(1)
		if cs.tgt then
			e:SetType(EFFECT_TYPE_FIELD)
			trange(e,cs.tgt)
			e:SetTarget(tfilter(cs.tgt))
		else
			e:SetType(EFFECT_TYPE_SINGLE)
			e:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
		end
	else
		error("procgen : effet continu inconnu "..tostring(code))
	end
	c:RegisterEffect(e)
end

-------------------------------------------------------------------- cadres

function PG.init(c,s,id)
	local r=s.recipe
	if r then
		local k=r[1]
		c:EnableReviveLimit()
		if k=="fusion" then  --Flame Swordsman (noms), Fusion.AddProcMix avec des fonctions pour les filtres
			local mats={}
			for i,m in ipairs(r[2]) do mats[i]=PG.mat(m) or aux.TRUE end  --pas de nil dans la liste variadique
			Fusion.AddProcMix(c,true,true,table.unpack(mats))
		elseif k=="synchro" then  --Naturia Leodrake
			Synchro.AddProcedure(c,PG.mat(r[2][1]),1,1,Synchro.NonTuner(PG.mat(r[2][2])),1,99)
		elseif k=="xyz" then  --Gem-Knight Pearl
			Xyz.AddProcedure(c,PG.mat(r[2][1]),s.lv,r[2][2])
		elseif k=="link" then  --LANphorhynchus
			Link.AddProcedure(c,PG.mat(r[2][1]),r[2][2],r[2][3])
		end
	end
	for i,sp in ipairs(s.effects or {}) do PG.effect(c,sp,id,i-1) end
end
