# Inventaire des fonctionnalités (généré par translate/inventory.py)

13541 scripts officiels ; 64 cartes implémentées dans le moteur (engine/cards.py) ; 136 items couverts (présents dans leurs scripts), 1348 items non ignorés au total.

Cartes sans aucun item manquant : **957 (7.1 %)**. Ignorés : plomberie et données génériques (groupes, filtres de base, affichage) : le traducteur doit les comprendre, mais ce ne sont pas des fonctionnalités du moteur.

Approximation : « couvert » = vu dans une carte implémentée ; la mesure exacte viendra du traducteur (cartes traduites et vérifiées contre l'oracle).

## Nombre d'items manquants par carte

| manquants | cartes | cumul |
|---|---|---|
| 0 | 957 | 7.1 % |
| 1 | 1107 | 15.2 % |
| 2 | 1360 | 25.3 % |
| 3 | 1318 | 35.0 % |
| 4 | 1301 | 44.6 % |
| 5 | 1163 | 53.2 % |
| 6 | 1111 | 61.4 % |
| 7 | 993 | 68.8 % |
| 8 | 842 | 75.0 % |
| 9 | 709 | 80.2 % |
| 10 | 609 | 84.7 % |
| 11 | 548 | 88.8 % |
| 12 | 353 | 91.4 % |
| 13 | 305 | 93.6 % |
| 14 | 251 | 95.5 % |
| 15 | 172 | 96.7 % |
| 16 | 115 | 97.6 % |
| 17 | 93 | 98.3 % |
| 18 | 56 | 98.7 % |
| 19 | 54 | 99.1 % |
| 20+ | 124 | 100.0 % |

## Courbe gloutonne (fonctionnalité qui débloque le plus de cartes à chaque pas)

| pas | item | débloquées | utilisé par (restantes) | jouables | % |
|---|---|---|---|---|---|
| 1 | `Fusion.AddProcMix` | 58 | 359 | 1015 | 7.5 % |
| 2 | `EFFECT_COUNT_CODE_OATH` | 52 | 1122 | 1067 | 7.9 % |
| 3 | `CHAININFO_TARGET_CARDS` | 34 | 441 | 1101 | 8.1 % |
| 4 | `aux.NecroValleyFilter` | 32 | 905 | 1133 | 8.4 % |
| 5 | `EVENT_BATTLE_DAMAGE` | 32 | 206 | 1165 | 8.6 % |
| 6 | `EFFECT_DIRECT_ATTACK` | 32 | 196 | 1197 | 8.8 % |
| 7 | `EFFECT_PIERCE` | 30 | 187 | 1227 | 9.1 % |
| 8 | `:IsDiscardable` | 25 | 265 | 1252 | 9.2 % |
| 9 | `Duel.SendtoGrave` | 24 | 1808 | 1276 | 9.4 % |
| 10 | `:IsAbleToGraveAsCost` | 62 | 611 | 1338 | 9.9 % |
| 11 | `:IsAbleToGrave` | 48 | 520 | 1386 | 10.2 % |
| 12 | `Duel.GetFieldGroup` | 40 | 461 | 1426 | 10.5 % |
| 13 | `Duel.SendtoDeck` | 27 | 907 | 1453 | 10.7 % |
| 14 | `:IsAbleToDeck` | 51 | 494 | 1504 | 11.1 % |
| 15 | `Duel.GetTargetCards` | 36 | 656 | 1540 | 11.4 % |
| 16 | `Duel.GetOperatedGroup` | 36 | 335 | 1576 | 11.6 % |
| 17 | `aux.SelectUnselectGroup` | 29 | 795 | 1605 | 11.9 % |
| 18 | `:IsLevel` | 26 | 422 | 1631 | 12.0 % |
| 19 | `Duel.GetMZoneCount` | 31 | 512 | 1662 | 12.3 % |
| 20 | `Duel.SelectEffect` | 25 | 371 | 1687 | 12.5 % |
| 21 | `:IsAbleToRemoveAsCost` | 24 | 654 | 1711 | 12.6 % |
| 22 | `aux.SpElimFilter` | 67 | 485 | 1778 | 13.1 % |
| 23 | `Duel.SelectOption` | 25 | 229 | 1803 | 13.3 % |
| 24 | `Duel.ChangePosition` | 25 | 505 | 1828 | 13.5 % |
| 25 | `:IsPublic` | 24 | 264 | 1852 | 13.7 % |
| 26 | `EFFECT_UPDATE_LEVEL` | 25 | 228 | 1877 | 13.9 % |
| 27 | `EFFECT_FLAG_CANNOT_DISABLE` | 31 | 2653 | 1908 | 14.1 % |
| 28 | `Duel.GetFirstMatchingCard` | 23 | 65 | 1931 | 14.3 % |
| 29 | `EVENT_BATTLE_DESTROYING` | 23 | 326 | 1954 | 14.4 % |
| 30 | `EFFECT_CHANGE_CODE` | 23 | 174 | 1977 | 14.6 % |
| 31 | `EFFECT_CHANGE_LEVEL` | 22 | 206 | 1999 | 14.8 % |
| 32 | `:HasLevel` | 30 | 306 | 2029 | 15.0 % |
| 33 | `:ListsCode` | 22 | 220 | 2051 | 15.1 % |
| 34 | `EFFECT_CANNOT_SELECT_BATTLE_TARGET` | 22 | 125 | 2073 | 15.3 % |
| 35 | `:IsPosition` | 22 | 214 | 2095 | 15.5 % |
| 36 | `:IsCanChangePosition` | 28 | 151 | 2123 | 15.7 % |
| 37 | `:IsDefensePos` | 24 | 145 | 2147 | 15.9 % |
| 38 | `:IsCanTurnSet` | 24 | 141 | 2171 | 16.0 % |
| 39 | `Duel.IsEnvironment` | 23 | 84 | 2194 | 16.2 % |
| 40 | `:RandomSelect` | 19 | 161 | 2213 | 16.3 % |
| 41 | `:IsSummonPlayer` | 18 | 167 | 2231 | 16.5 % |
| 42 | `Gemini.AddProcedure` | 18 | 45 | 2249 | 16.6 % |
| 43 | `EVENT_PHASE` | 17 | 1314 | 2266 | 16.7 % |
| 44 | `EFFECT_TYPE_CONTINUOUS` | 39 | 2163 | 2305 | 17.0 % |
| 45 | `PHASE_STANDBY` | 23 | 442 | 2328 | 17.2 % |
| 46 | `Duel.GetTurnPlayer` | 30 | 245 | 2358 | 17.4 % |
| 47 | `:IsSummonLocation` | 20 | 226 | 2378 | 17.6 % |
| 48 | `:IsPreviousControler` | 22 | 754 | 2400 | 17.7 % |
| 49 | `:IsReason` | 21 | 1579 | 2421 | 17.9 % |
| 50 | `REASON_BATTLE` | 90 | 949 | 2511 | 18.5 % |
| 51 | `EVENT_DESTROYED` | 50 | 657 | 2561 | 18.9 % |
| 52 | `REASON_DESTROY` | 46 | 456 | 2607 | 19.3 % |
| 53 | `:IsPreviousPosition` | 43 | 482 | 2650 | 19.6 % |
| 54 | `EVENT_REMOVE` | 25 | 200 | 2675 | 19.8 % |
| 55 | `Duel.GetLP` | 21 | 255 | 2696 | 19.9 % |
| 56 | `Duel.DiscardDeck` | 21 | 136 | 2717 | 20.1 % |
| 57 | `Duel.IsPlayerCanDiscardDeck` | 24 | 102 | 2741 | 20.2 % |
| 58 | `aux.ToHandOrElse` | 22 | 130 | 2763 | 20.4 % |
| 59 | `Duel.NegateAttack` | 17 | 124 | 2780 | 20.5 % |
| 60 | `EVENT_BE_BATTLE_TARGET` | 24 | 150 | 2804 | 20.7 % |
| 61 | `:IsRelateToBattle` | 19 | 600 | 2823 | 20.8 % |
| 62 | `:GetBattleTarget` | 38 | 478 | 2861 | 21.1 % |
| 63 | `PHASE_DAMAGE` | 27 | 282 | 2888 | 21.3 % |
| 64 | `EVENT_DAMAGE_STEP_END` | 28 | 146 | 2916 | 21.5 % |
| 65 | `EVENT_BATTLE_START` | 21 | 126 | 2937 | 21.7 % |
| 66 | `Duel.GetLocationCountFromEx` | 18 | 423 | 2955 | 21.8 % |
| 67 | `REASON_RETURN` | 19 | 55 | 2974 | 22.0 % |
| 68 | `PHASE_BATTLE` | 18 | 295 | 2992 | 22.1 % |
| 69 | `EFFECT_AVOID_BATTLE_DAMAGE` | 19 | 136 | 3011 | 22.2 % |
| 70 | `EVENT_BATTLED` | 19 | 160 | 3030 | 22.4 % |
| 71 | `EFFECT_INDESTRUCTABLE_COUNT` | 20 | 105 | 3050 | 22.5 % |
| 72 | `:IsTributeSummoned` | 18 | 109 | 3068 | 22.7 % |
| 73 | `:GetPreviousLocation` | 18 | 97 | 3086 | 22.8 % |
| 74 | `:GetPreviousControler` | 18 | 116 | 3104 | 22.9 % |
| 75 | `:GetReasonPlayer` | 18 | 152 | 3122 | 23.1 % |
| 76 | `EVENT_LEAVE_FIELD` | 19 | 297 | 3141 | 23.2 % |
| 77 | `:GetDescription` | 17 | 123 | 3158 | 23.3 % |
| 78 | `STATUS_BATTLE_DESTROYED` | 17 | 330 | 3175 | 23.4 % |
| 79 | `EFFECT_FLAG_DAMAGE_CAL` | 24 | 312 | 3199 | 23.6 % |
| 80 | `:IsHasProperty` | 21 | 188 | 3220 | 23.8 % |
| 81 | `Duel.GetOperationInfo` | 22 | 79 | 3242 | 23.9 % |
| 82 | `EVENT_DAMAGE` | 19 | 69 | 3261 | 24.1 % |
| 83 | `EFFECT_SELF_DESTROY` | 18 | 87 | 3279 | 24.2 % |
| 84 | `:SetUniqueOnField` | 20 | 157 | 3299 | 24.4 % |
| 85 | `EVENT_FLIP` | 18 | 132 | 3317 | 24.5 % |
| 86 | `:CreateEffectRelation` | 18 | 114 | 3335 | 24.6 % |
| 87 | `:CompleteProcedure` | 17 | 225 | 3352 | 24.8 % |
| 88 | `:IsAbleToDeckAsCost` | 18 | 52 | 3370 | 24.9 % |
| 89 | `Fusion.AddProcMixN` | 17 | 154 | 3387 | 25.0 % |
| 90 | `:IsFusionSummoned` | 17 | 148 | 3404 | 25.1 % |
| 91 | `Fusion.CreateSummonEff` | 17 | 98 | 3421 | 25.3 % |
| 92 | `:IsPreviousSetCard` | 18 | 84 | 3439 | 25.4 % |
| 93 | `Duel.RDComplete` | 17 | 40 | 3456 | 25.5 % |
| 94 | `:GetReasonCard` | 16 | 203 | 3472 | 25.6 % |
| 95 | `:IsLinkMonster` | 16 | 197 | 3488 | 25.8 % |
| 96 | `:UpdateAttack` | 17 | 125 | 3505 | 25.9 % |
| 97 | `:GetLinkedGroup` | 16 | 140 | 3521 | 26.0 % |
| 98 | `EFFECT_EXTRA_ATTACK` | 16 | 165 | 3537 | 26.1 % |
| 99 | `EFFECT_CANNOT_ATTACK_ANNOUNCE` | 16 | 111 | 3553 | 26.2 % |
| 100 | `:GetOriginalLevel` | 16 | 108 | 3569 | 26.4 % |
| 101 | `EFFECT_COUNT_CODE_DUEL` | 16 | 60 | 3585 | 26.5 % |
| 102 | `EFFECT_FLAG_UNCOPYABLE` | 15 | 1390 | 3600 | 26.6 % |
| 103 | `EFFECT_SPSUMMON_PROC` | 118 | 646 | 3718 | 27.5 % |
| 104 | `EFFECT_SPSUMMON_CONDITION` | 101 | 505 | 3819 | 28.2 % |
| 105 | `aux.ChkfMMZ` | 37 | 172 | 3856 | 28.5 % |
| 106 | `EFFECT_SUMMON_PROC` | 19 | 113 | 3875 | 28.6 % |
| 107 | `RESET_TOFIELD` | 22 | 226 | 3897 | 28.8 % |
| 108 | `EFFECT_FLAG_OATH` | 18 | 419 | 3915 | 28.9 % |
| 109 | `:GetAttackAnnouncedCount` | 19 | 58 | 3934 | 29.1 % |
| 110 | `EFFECT_CANNOT_BE_SYNCHRO_MATERIAL` | 18 | 72 | 3952 | 29.2 % |
| 111 | `EVENT_TO_HAND` | 19 | 87 | 3971 | 29.3 % |
| 112 | `EFFECT_SET_DEFENSE_FINAL` | 17 | 99 | 3988 | 29.5 % |
| 113 | `EFFECT_DOUBLE_TRIBUTE` | 16 | 29 | 4004 | 29.6 % |
| 114 | `:GetEquipGroup` | 15 | 146 | 4019 | 29.7 % |
| 115 | `:IsOriginalType` | 15 | 222 | 4034 | 29.8 % |
| 116 | `aux.addTempLizardCheck` | 17 | 184 | 4051 | 29.9 % |
| 117 | `:IsRitualMonster` | 17 | 120 | 4068 | 30.0 % |
| 118 | `EFFECT_EXTRA_SUMMON_COUNT` | 16 | 73 | 4084 | 30.2 % |
| 119 | `aux.DoubleSnareValidity` | 15 | 100 | 4099 | 30.3 % |
| 120 | `Duel.SetChainLimit` | 15 | 89 | 4114 | 30.4 % |
| 121 | `EFFECT_MUST_ATTACK` | 15 | 54 | 4129 | 30.5 % |
| 122 | `:GetPreviousRaceOnField` | 15 | 53 | 4144 | 30.6 % |
| 123 | `Spirit.AddProcedure` | 15 | 42 | 4159 | 30.7 % |
| 124 | `Duel.IsPhase` | 14 | 308 | 4173 | 30.8 % |
| 125 | `:GetBaseDefense` | 15 | 60 | 4188 | 30.9 % |
| 126 | `EFFECT_FLAG2_CHECK_SIMULTANEOUS` | 14 | 138 | 4202 | 31.0 % |
| 127 | `EFFECT_CANNOT_CHANGE_POSITION` | 15 | 75 | 4217 | 31.1 % |
| 128 | `EFFECT_FLAG_SET_AVAILABLE` | 14 | 246 | 4231 | 31.2 % |
| 129 | `:GetTextAttack` | 14 | 105 | 4245 | 31.3 % |
| 130 | `Duel.SetLP` | 16 | 83 | 4261 | 31.5 % |
| 131 | `Duel.TossDice` | 15 | 61 | 4276 | 31.6 % |
| 132 | `EFFECT_CANNOT_SUMMON` | 14 | 135 | 4290 | 31.7 % |
| 133 | `:IsOriginalRace` | 14 | 73 | 4304 | 31.8 % |
| 134 | `:IsAbleToHandAsCost` | 14 | 37 | 4318 | 31.9 % |
| 135 | `:IsImmuneToEffect` | 13 | 416 | 4331 | 32.0 % |
| 136 | `Duel.Overlay` | 35 | 258 | 4366 | 32.2 % |
| 137 | `:GetOverlayGroup` | 16 | 131 | 4382 | 32.4 % |
| 138 | `:CancelToGrave` | 15 | 115 | 4397 | 32.5 % |
| 139 | `EFFECT_ADD_TYPE` | 14 | 125 | 4411 | 32.6 % |
| 140 | `RESET_SELF_TURN` | 13 | 148 | 4424 | 32.7 % |
| 141 | `SUMMON_TYPE_FUSION` | 13 | 128 | 4437 | 32.8 % |
| 142 | `Fusion.IsMonsterFilter` | 15 | 89 | 4452 | 32.9 % |
| 143 | `:SetSPSummonOnce` | 14 | 76 | 4466 | 33.0 % |
| 144 | `:IsAttributeExcept` | 14 | 75 | 4480 | 33.1 % |
| 145 | `Duel.GetBattleMonster` | 14 | 73 | 4494 | 33.2 % |
| 146 | `:GetAttackedCount` | 14 | 27 | 4508 | 33.3 % |
| 147 | `:IsXyzSummoned` | 13 | 119 | 4521 | 33.4 % |
| 148 | `REASON_RULE` | 14 | 156 | 4535 | 33.5 % |
| 149 | `EVENT_DRAW` | 17 | 51 | 4552 | 33.6 % |
| 150 | `RESET_OPPO_TURN` | 13 | 113 | 4565 | 33.7 % |

## Items manquants les plus fréquents (toutes cartes)

| item | cartes |
|---|---|
| `EFFECT_FLAG_CANNOT_DISABLE` | 2653 |
| `EFFECT_TYPE_CONTINUOUS` | 2163 |
| `Duel.SendtoGrave` | 1808 |
| `:IsReason` | 1579 |
| `EFFECT_FLAG_UNCOPYABLE` | 1390 |
| `EVENT_PHASE` | 1314 |
| `EFFECT_COUNT_CODE_OATH` | 1122 |
| `:RegisterFlagEffect` | 1078 |
| `REASON_BATTLE` | 949 |
| `Duel.SendtoDeck` | 907 |
| `aux.NecroValleyFilter` | 905 |
| `aux.SelectUnselectGroup` | 795 |
| `Duel.SpecialSummonStep` | 784 |
| `Duel.SpecialSummonComplete` | 783 |
| `:IsPreviousControler` | 754 |
| `:GetFlagEffect` | 749 |
| `EVENT_DESTROYED` | 657 |
| `Duel.GetTargetCards` | 656 |
| `:IsAbleToRemoveAsCost` | 654 |
| `EFFECT_SPSUMMON_PROC` | 646 |
| `:IsAbleToGraveAsCost` | 611 |
| `:IsRelateToBattle` | 600 |
| `Synchro.AddProcedure` | 529 |
| `:IsAbleToGrave` | 520 |
| `Duel.GetMZoneCount` | 512 |
| `Duel.ChangePosition` | 505 |
| `EFFECT_SPSUMMON_CONDITION` | 505 |
| `:IsAbleToDeck` | 494 |
| `aux.SpElimFilter` | 485 |
| `:IsPreviousPosition` | 482 |
| `:GetBattleTarget` | 478 |
| `Duel.GetFieldGroup` | 461 |
| `REASON_DESTROY` | 456 |
| `PHASE_STANDBY` | 442 |
| `CHAININFO_TARGET_CARDS` | 441 |
| `EFFECT_TYPE_EQUIP` | 440 |
| `Duel.GetLocationCountFromEx` | 423 |
| `:IsLevel` | 422 |
| `EFFECT_FLAG_OATH` | 419 |
| `:IsImmuneToEffect` | 416 |
| `Duel.SelectEffect` | 371 |
| `Synchro.NonTuner` | 367 |
| `Duel.SSet` | 362 |
| `Duel.RegisterFlagEffect` | 362 |
| `:IsSSetable` | 361 |
| `Fusion.AddProcMix` | 359 |
| `Duel.GetOperatedGroup` | 335 |
| `STATUS_BATTLE_DESTROYED` | 330 |
| `EVENT_BATTLE_DESTROYING` | 326 |
| `Duel.IsPlayerCanSpecialSummonMonster` | 324 |
| `EFFECT_FLAG_DAMAGE_CAL` | 312 |
| `Duel.IsPhase` | 308 |
| `:HasLevel` | 306 |
| `Duel.Equip` | 306 |
| `EVENT_LEAVE_FIELD` | 297 |
| `PHASE_BATTLE` | 295 |
| `:IsForbidden` | 290 |
| `:GetEquipTarget` | 289 |
| `PHASE_DAMAGE` | 282 |
| `EFFECT_LEAVE_FIELD_REDIRECT` | 277 |
| `:AddCounter` | 271 |
| `:Reset` | 267 |
| `:IsDiscardable` | 265 |
| `Duel.NegateEffect` | 264 |
| `:IsPublic` | 264 |
| `Duel.Overlay` | 258 |
| `RESETS_REDIRECT` | 258 |
| `Duel.GetLP` | 255 |
| `RESET_CHAIN` | 253 |
| `Duel.SelectEffectYesNo` | 253 |
| `aux.AddEquipProcedure` | 250 |
| `EFFECT_DESTROY_REPLACE` | 247 |
| `EFFECT_FLAG_SET_AVAILABLE` | 246 |
| `EFFECT_EQUIP_LIMIT` | 245 |
| `Duel.GetTurnPlayer` | 245 |
| `Duel.CreateToken` | 243 |
| `REASON_REPLACE` | 242 |
| `Duel.GetTurnCount` | 240 |
| `EVENT_BE_MATERIAL` | 239 |
| `Duel.SelectOption` | 229 |
| `EFFECT_UPDATE_LEVEL` | 228 |
| `RESET_TOFIELD` | 226 |
| `:IsSummonLocation` | 226 |
| `:CompleteProcedure` | 225 |
| `:IsOriginalType` | 222 |
| `:GetMaterial` | 222 |
| `:ListsCode` | 220 |
| `Duel.MoveToField` | 216 |
| `:HasFlagEffect` | 214 |
| `:IsPosition` | 214 |
| `aux.GlobalCheck` | 207 |
| `EVENT_BATTLE_DAMAGE` | 206 |
| `EFFECT_CHANGE_LEVEL` | 206 |
| `:IsSynchroSummoned` | 203 |
| `:GetReasonCard` | 203 |
| `EVENT_REMOVE` | 200 |
| `Duel.GetCurrentChain` | 200 |
| `Duel.GetFlagEffect` | 199 |
| `:IsLinkMonster` | 197 |
| `EFFECT_DIRECT_ATTACK` | 196 |
| `SUMMON_TYPE_XYZ` | 193 |
| `:GetCounter` | 192 |
| `:IsHasProperty` | 188 |
| `EFFECT_PIERCE` | 187 |
| `:GetFieldID` | 186 |
| `aux.addTempLizardCheck` | 184 |
| `Duel.AddCustomActivityCounter` | 181 |
| `:EnableCounterPermit` | 180 |
| `:IsHasEffect` | 179 |
| `Duel.GetCustomActivityCount` | 176 |
| `ACTIVITY_SPSUMMON` | 176 |
| `Duel.GetDecktopGroup` | 175 |
| `EFFECT_MATERIAL_CHECK` | 174 |
| `EFFECT_CHANGE_CODE` | 174 |
| `aux.ChkfMMZ` | 172 |
| `:NegateEffects` | 172 |
| `Duel.IsChainDisablable` | 169 |
| `:IsSummonPlayer` | 167 |
| `EFFECT_EXTRA_ATTACK` | 165 |
| `REASON_MATERIAL` | 162 |
| `:RandomSelect` | 161 |
| `Duel.ConfirmDecktop` | 160 |
| `EVENT_BATTLED` | 160 |
| `Duel.GetControl` | 159 |
| `Duel.HasFlagEffect` | 157 |
| `:SetUniqueOnField` | 157 |
| `REASON_RULE` | 156 |
| `Fusion.AddProcMixN` | 154 |
| `CHAININFO_TRIGGERING_LOCATION` | 153 |
| `:GetReasonPlayer` | 152 |
| `:IsCanChangePosition` | 151 |
| `:GetFlagEffectLabel` | 150 |
| `EVENT_BE_BATTLE_TARGET` | 150 |
| `REASON_SYNCHRO` | 150 |
| `:IsFusionSummoned` | 148 |
| `RESET_SELF_TURN` | 148 |
| `Synchro.NonTunerEx` | 148 |
| `EVENT_DAMAGE_STEP_END` | 146 |
| `:GetEquipGroup` | 146 |
| `:IsDefensePos` | 145 |
| `:IsActivated` | 144 |
| `:GetOwnerPlayer` | 143 |
| `EVENT_CHAIN_SOLVING` | 142 |
| `:IsCanTurnSet` | 141 |
| `:GetLinkedGroup` | 140 |
| `EFFECT_FLAG2_CHECK_SIMULTANEOUS` | 138 |
| `EFFECT_AVOID_BATTLE_DAMAGE` | 136 |
| `Duel.DiscardDeck` | 136 |
| `:IsCanBeXyzMaterial` | 135 |
| `EFFECT_CANNOT_SUMMON` | 135 |
| `EVENT_CHAIN_SOLVED` | 133 |
| `:CheckUniqueOnField` | 132 |
| `EVENT_FLIP` | 132 |
| `Duel.DisableShuffleCheck` | 131 |
| `:GetOverlayGroup` | 131 |
| `EVENT_CUSTOM` | 131 |
| `aux.ToHandOrElse` | 130 |
| `SUMMON_TYPE_FUSION` | 128 |
| `:SetMaterial` | 127 |
| `SUMMON_TYPE_SPECIAL` | 126 |
| `EVENT_BATTLE_START` | 126 |
| `EFFECT_ADD_TYPE` | 125 |
| `Fusion.SummonEffOP` | 125 |
| `Fusion.SummonEffTG` | 125 |
| `EFFECT_CANNOT_SELECT_BATTLE_TARGET` | 125 |
| `:UpdateAttack` | 125 |
| `Duel.NegateAttack` | 124 |
| `:GetDescription` | 123 |
| `:IsRitualMonster` | 120 |
| `:IsXyzSummoned` | 119 |
| `:GetPreviousControler` | 116 |
| `:CancelToGrave` | 115 |
| `:CreateEffectRelation` | 114 |
| `:IsControlerCanBeChanged` | 114 |
| `Duel.SelectReleaseGroup` | 113 |
| `RESET_OPPO_TURN` | 113 |
| `EFFECT_SUMMON_PROC` | 113 |
| `EFFECT_CANNOT_TRIGGER` | 112 |
| `Duel.CheckReleaseGroup` | 111 |
| `EFFECT_CANNOT_ATTACK_ANNOUNCE` | 111 |
| `:IsTributeSummoned` | 109 |
| `REASON_XYZ` | 108 |
| `:GetOriginalLevel` | 108 |
| `:GetSummonType` | 106 |
| `:GetTextAttack` | 105 |
| `EFFECT_INDESTRUCTABLE_COUNT` | 105 |
| `:SetCardTarget` | 103 |
| `Duel.IsPlayerCanDiscardDeck` | 102 |
| `:IsAttack` | 102 |
| `aux.DoubleSnareValidity` | 100 |
| `SUMMON_TYPE_SYNCHRO` | 100 |
| `EFFECT_SET_DEFENSE_FINAL` | 99 |
| `Fusion.CreateSummonEff` | 98 |
| `Duel.IsAbleToEnterBP` | 98 |
| `PHASE_MAIN1` | 98 |
| `:GetPreviousLocation` | 97 |
| `:EquipByEffectAndLimitRegister` | 97 |
| `STATUS_DESTROY_CONFIRMED` | 97 |
| `:GetRank` | 95 |
| `:IsAbleToChangeControler` | 95 |
