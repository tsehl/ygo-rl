# Card script primitive statistics

- Source: `/Users/thomas/Documents/collab/ygo/data/CardScripts/official`
- Total card files matched: **13541**
- Files processed: **13541**; failed to read/parse: **0**
- Comments (`--` and `--[[ ]]`) stripped before matching.
- Full tables: `/Users/thomas/Documents/collab/ygo/analysis/primitives_stats.json`

## 1. Duel API calls (top 80 of 237 distinct)

| token | files | occurrences |
|---|---:|---:|
| SetOperationInfo | 10651 | 17216 |
| Hint | 9372 | 14957 |
| IsExistingMatchingCard | 7242 | 11109 |
| SelectMatchingCard | 5399 | 7146 |
| GetLocationCount | 5359 | 8940 |
| SelectTarget | 4788 | 5463 |
| IsExistingTarget | 4588 | 5316 |
| GetFirstTarget | 4490 | 4909 |
| SpecialSummon | 4419 | 4951 |
| GetMatchingGroup | 3176 | 5123 |
| SendtoHand | 2925 | 3204 |
| Destroy | 2752 | 3042 |
| ConfirmCards | 2577 | 2875 |
| RegisterEffect | 1986 | 2764 |
| SendtoGrave | 1808 | 2003 |
| BreakEffect | 1751 | 1975 |
| GetChainInfo | 1723 | 1855 |
| Remove | 1519 | 1703 |
| SelectYesNo | 1147 | 1245 |
| SetPossibleOperationInfo | 1048 | 1423 |
| IsTurnPlayer | 1022 | 1137 |
| GetFieldGroupCount | 999 | 1516 |
| SetTargetPlayer | 948 | 987 |
| SetTargetParam | 934 | 965 |
| SendtoDeck | 907 | 972 |
| Draw | 839 | 911 |
| Damage | 824 | 925 |
| GetAttacker | 817 | 1180 |
| HintSelection | 790 | 893 |
| SpecialSummonStep | 784 | 864 |
| SpecialSummonComplete | 783 | 793 |
| IsPlayerAffectedByEffect | 764 | 1185 |
| Release | 741 | 759 |
| SetTargetCard | 711 | 759 |
| GetTargetCards | 656 | 665 |
| IsPlayerCanDraw | 649 | 746 |
| GetAttackTarget | 627 | 888 |
| DiscardHand | 620 | 636 |
| GetMatchingGroupCount | 572 | 734 |
| ShuffleHand | 540 | 602 |
| GetMZoneCount | 512 | 559 |
| ChangePosition | 505 | 567 |
| CheckReleaseGroupCost | 485 | 497 |
| SelectReleaseGroupCost | 483 | 493 |
| GetFieldGroup | 461 | 599 |
| GetLocationCountFromEx | 423 | 493 |
| NegateActivation | 391 | 393 |
| IsChainNegatable | 380 | 386 |
| SelectEffect | 371 | 386 |
| IsMainPhase | 369 | 375 |
| RegisterFlagEffect | 362 | 476 |
| SSet | 362 | 367 |
| GetOperatedGroup | 335 | 349 |
| IsPlayerCanSpecialSummonMonster | 324 | 596 |
| IsPhase | 308 | 337 |
| Equip | 306 | 318 |
| ShuffleDeck | 277 | 294 |
| Recover | 269 | 291 |
| NegateEffect | 264 | 271 |
| Overlay | 258 | 282 |
| GetLP | 255 | 439 |
| SelectEffectYesNo | 253 | 254 |
| GetTurnPlayer | 245 | 276 |
| CreateToken | 243 | 262 |
| GetTurnCount | 240 | 431 |
| SelectOption | 229 | 434 |
| IsBattlePhase | 229 | 232 |
| MoveToField | 216 | 242 |
| GetCurrentChain | 200 | 223 |
| GetFlagEffect | 199 | 279 |
| GetCurrentPhase | 188 | 195 |
| AddCustomActivityCounter | 181 | 193 |
| GetCustomActivityCount | 176 | 205 |
| GetDecktopGroup | 175 | 234 |
| IsChainDisablable | 169 | 173 |
| ConfirmDecktop | 160 | 184 |
| GetControl | 159 | 160 |
| HasFlagEffect | 157 | 297 |
| PayLPCost | 156 | 162 |
| DiscardDeck | 136 | 159 |

## 2. aux helper references (top 80 of 113 distinct)

| token | files | occurrences |
|---|---:|---:|
| Stringid | 10385 | 20316 |
| FaceupFilter | 1393 | 2150 |
| NecroValleyFilter | 905 | 1041 |
| FilterBoolFunctionEx | 853 | 945 |
| SelectUnselectGroup | 795 | 1403 |
| Next | 500 | 557 |
| SpElimFilter | 485 | 502 |
| TRUE | 437 | 819 |
| TargetBoolFunction | 429 | 474 |
| FilterBoolFunction | 402 | 446 |
| RegisterClientHint | 266 | 273 |
| StatChangeDamageStepCondition | 256 | 259 |
| AddEquipProcedure | 250 | 250 |
| tgoval | 236 | 240 |
| GlobalCheck | 207 | 207 |
| addTempLizardCheck | 184 | 184 |
| ChkfMMZ | 172 | 278 |
| dncheck | 165 | 210 |
| AND | 161 | 293 |
| FALSE | 146 | 152 |
| NOT | 140 | 180 |
| ToHandOrElse | 130 | 131 |
| exccon | 120 | 120 |
| DoubleSnareValidity | 100 | 100 |
| GetMustBeMaterialGroup | 94 | 163 |
| AddEREquipLimit | 91 | 103 |
| indoval | 87 | 87 |
| fuslimit | 75 | 75 |
| bdocon | 74 | 74 |
| DelayedOperation | 71 | 71 |
| AddNormalSummonProcedure | 59 | 59 |
| ChangeBattleDamage | 53 | 53 |
| ReleaseCheckMMZ | 51 | 95 |
| ReleaseCheckTarget | 48 | 96 |
| dpcheck | 48 | 66 |
| RemainFieldCost | 44 | 44 |
| chainreg | 44 | 44 |
| bfgcost | 43 | 62 |
| AddUnionProcedure | 40 | 40 |
| AddValuesReset | 38 | 38 |
| RemoveUntil | 35 | 36 |
| AddNormalSetProcedure | 35 | 35 |
| imval1 | 32 | 32 |
| addContinuousLizardCheck | 30 | 30 |
| cannotmatfilter | 29 | 30 |
| bdogcon | 29 | 29 |
| DefaultFieldReturnOp | 28 | 28 |
| PersistentTargetFilter | 25 | 31 |
| LimitParams | 24 | 28 |
| GetMMZonesPointedTo | 23 | 44 |
| AddPersistentProcedure | 23 | 23 |
| FilterSummonCode | 21 | 22 |
| bdcon | 21 | 21 |
| IsUnionState | 20 | 29 |
| damcon1 | 20 | 22 |
| imval2 | 20 | 20 |
| FieldSummonProcTg | 19 | 19 |
| nvfilter | 19 | 19 |
| ritlimit | 17 | 17 |
| EquipAndLimitRegister | 16 | 18 |
| createContinuousLizardCheck | 16 | 16 |
| sumreg | 16 | 16 |
| AddZWEquipLimit | 14 | 14 |
| gbspcon | 14 | 14 |
| zptcon | 14 | 14 |
| EnableCheckReincarnation | 13 | 13 |
| synlimit | 12 | 12 |
| CheckSummonGate | 11 | 11 |
| EnableNeosReturn | 11 | 11 |
| bdgcon | 11 | 11 |
| CheckStealEquip | 10 | 23 |
| FilterEqualFunction | 9 | 13 |
| CheckUnionEquip | 8 | 11 |
| FossilLimit | 8 | 8 |
| PersistentTgCon | 8 | 8 |
| SetUnionState | 8 | 8 |
| qlifilter | 8 | 8 |
| OR | 7 | 12 |
| IceBarrierDiscardCost | 7 | 9 |
| IsZone | 7 | 8 |

## 3. Card method calls, uppercase names (top 80 of 445 distinct)

Excludes effect setters and `Clone`/`RegisterEffect` (see stat 4).

| token | files | occurrences |
|---|---:|---:|
| GetHandler | 10275 | 28480 |
| IsRelateToEffect | 7632 | 11159 |
| IsLocation | 5788 | 7550 |
| IsFaceup | 5521 | 8302 |
| IsSetCard | 5087 | 8173 |
| IsControler | 4885 | 6174 |
| IsCanBeSpecialSummoned | 4724 | 5560 |
| GetFirst | 3430 | 4570 |
| IsAbleToHand | 2789 | 3075 |
| EnableReviveLimit | 2641 | 2642 |
| IsCode | 2547 | 3070 |
| IsRace | 2373 | 3258 |
| IsExists | 2328 | 2932 |
| IsType | 2246 | 3029 |
| IsMonster | 2071 | 2366 |
| GetLabel | 1612 | 2277 |
| IsReason | 1579 | 2088 |
| GetLabelObject | 1387 | 2298 |
| IsPreviousLocation | 1379 | 1475 |
| IsAttribute | 1294 | 1682 |
| GetHandlerPlayer | 1179 | 1351 |
| GetControler | 1134 | 1561 |
| Select | 1127 | 1307 |
| RegisterFlagEffect | 1078 | 1258 |
| IsOnField | 1006 | 1052 |
| IsFacedown | 815 | 964 |
| Filter | 810 | 1105 |
| IsHasType | 770 | 856 |
| IsLevelBelow | 764 | 806 |
| GetLevel | 755 | 1253 |
| IsPreviousControler | 754 | 791 |
| GetFlagEffect | 749 | 878 |
| IsContains | 747 | 896 |
| IsSpellTrap | 714 | 771 |
| GetAttack | 712 | 1196 |
| IsStatus | 705 | 784 |
| IsMonsterEffect | 690 | 738 |
| IsAbleToRemove | 678 | 758 |
| IsAbleToRemoveAsCost | 654 | 720 |
| IsAbleToGraveAsCost | 611 | 659 |
| IsRelateToBattle | 600 | 821 |
| FilterCount | 581 | 740 |
| IsAbleToGrave | 520 | 556 |
| GetCode | 510 | 703 |
| IsAbleToDeck | 494 | 533 |
| IsPreviousPosition | 482 | 486 |
| GetBattleTarget | 478 | 707 |
| GetSequence | 471 | 565 |
| IsLevelAbove | 454 | 513 |
| Iter | 437 | 474 |
| IsLevel | 422 | 496 |
| IsImmuneToEffect | 416 | 468 |
| IsCanBeEffectTarget | 405 | 428 |
| KeepAlive | 405 | 415 |
| IsDestructable | 372 | 379 |
| GetClassCount | 366 | 492 |
| IsSSetable | 361 | 458 |
| Merge | 332 | 429 |
| DeleteGroup | 319 | 335 |
| IsSpell | 313 | 353 |
| GetOwner | 311 | 452 |
| GetBaseAttack | 310 | 471 |
| HasLevel | 306 | 388 |
| IsForbidden | 290 | 313 |
| GetEquipTarget | 289 | 502 |
| AddCounter | 271 | 318 |
| Reset | 267 | 337 |
| IsDiscardable | 265 | 279 |
| AddCard | 264 | 300 |
| IsPublic | 264 | 290 |
| FilterSelect | 247 | 268 |
| IsSummonLocation | 226 | 239 |
| CompleteProcedure | 225 | 228 |
| GetMaterial | 222 | 244 |
| IsOriginalType | 222 | 237 |
| ListsCode | 220 | 254 |
| IsPosition | 214 | 277 |
| HasFlagEffect | 214 | 250 |
| IsAttackPos | 204 | 252 |
| GetReasonCard | 203 | 257 |

## 4. Constants passed to effect setters

### SetCategory (top 50 of 32 distinct)

| token | files | occurrences |
|---|---:|---:|
| CATEGORY_SPECIAL_SUMMON | 5209 | 5831 |
| CATEGORY_TOHAND | 2890 | 3117 |
| CATEGORY_DESTROY | 2308 | 2436 |
| CATEGORY_SEARCH | 1794 | 1847 |
| CATEGORY_ATKCHANGE | 1078 | 1119 |
| CATEGORY_DRAW | 801 | 826 |
| CATEGORY_REMOVE | 764 | 801 |
| CATEGORY_DAMAGE | 730 | 755 |
| CATEGORY_TOGRAVE | 725 | 753 |
| CATEGORY_TODECK | 699 | 723 |
| CATEGORY_SET | 653 | 695 |
| CATEGORY_POSITION | 427 | 461 |
| CATEGORY_DISABLE | 415 | 426 |
| CATEGORY_NEGATE | 388 | 390 |
| CATEGORY_EQUIP | 378 | 390 |
| CATEGORY_HANDES | 266 | 272 |
| CATEGORY_RECOVER | 239 | 244 |
| CATEGORY_TOKEN | 230 | 239 |
| CATEGORY_CONTROL | 186 | 188 |
| CATEGORY_LVCHANGE | 163 | 170 |
| CATEGORY_DECKDES | 157 | 163 |
| CATEGORY_DEFCHANGE | 156 | 158 |
| CATEGORY_COUNTER | 150 | 167 |
| CATEGORY_FUSION_SUMMON | 145 | 146 |
| CATEGORY_LEAVE_GRAVE | 100 | 101 |
| CATEGORY_SUMMON | 99 | 102 |
| CATEGORY_TOEXTRA | 75 | 76 |
| CATEGORY_RELEASE | 60 | 61 |
| CATEGORY_COIN | 59 | 60 |
| CATEGORY_DISABLE_SUMMON | 50 | 50 |
| CATEGORY_DICE | 39 | 42 |
| EFFECT_FLAG_CANNOT_DISABLE | 1 | 1 |

### SetType (top 50 of 13 distinct)

| token | files | occurrences |
|---|---:|---:|
| EFFECT_TYPE_SINGLE | 8154 | 13879 |
| EFFECT_TYPE_FIELD | 6326 | 8926 |
| EFFECT_TYPE_TRIGGER_O | 4934 | 5939 |
| EFFECT_TYPE_ACTIVATE | 4455 | 4493 |
| EFFECT_TYPE_IGNITION | 4039 | 4585 |
| EFFECT_TYPE_CONTINUOUS | 2160 | 2652 |
| EFFECT_TYPE_QUICK_O | 1948 | 2085 |
| EFFECT_TYPE_TRIGGER_F | 1706 | 1870 |
| EFFECT_TYPE_EQUIP | 440 | 589 |
| EFFECT_TYPE_FLIP | 193 | 193 |
| EFFECT_TYPE_GRANT | 30 | 30 |
| EFFECT_TYPE_XMATERIAL | 28 | 30 |
| EFFECT_TYPE_QUICK_F | 17 | 17 |

### SetCode (top 50 of 319 distinct)

| token | files | occurrences |
|---|---:|---:|
| EVENT_FREE_CHAIN | 4867 | 5315 |
| EVENT_SPSUMMON_SUCCESS | 2306 | 2435 |
| EFFECT_UPDATE_ATTACK | 1915 | 1984 |
| EVENT_SUMMON_SUCCESS | 1345 | 1373 |
| EVENT_PHASE | 1314 | 1388 |
| EVENT_TO_GRAVE | 1237 | 1257 |
| EVENT_CHAINING | 965 | 986 |
| EFFECT_CANNOT_SPECIAL_SUMMON | 918 | 924 |
| PHASE_END | 772 | 788 |
| EVENT_DESTROYED | 657 | 678 |
| EFFECT_SPSUMMON_PROC | 645 | 646 |
| EFFECT_DISABLE | 532 | 544 |
| EFFECT_INDESTRUCTABLE_BATTLE | 515 | 519 |
| EFFECT_SPSUMMON_CONDITION | 504 | 504 |
| EVENT_ATTACK_ANNOUNCE | 487 | 496 |
| EFFECT_UPDATE_DEFENSE | 475 | 489 |
| EFFECT_INDESTRUCTABLE_EFFECT | 440 | 444 |
| EFFECT_DISABLE_EFFECT | 416 | 423 |
| PHASE_STANDBY | 406 | 414 |
| EFFECT_SET_ATTACK_FINAL | 358 | 369 |
| EVENT_BATTLE_DESTROYED | 344 | 344 |
| EVENT_BATTLE_DESTROYING | 326 | 330 |
| EFFECT_CANNOT_ACTIVATE | 319 | 327 |
| EFFECT_CANNOT_ATTACK | 318 | 323 |
| EFFECT_CANNOT_BE_EFFECT_TARGET | 316 | 319 |
| EVENT_LEAVE_FIELD | 297 | 326 |
| EFFECT_LEAVE_FIELD_REDIRECT | 277 | 278 |
| EFFECT_IMMUNE_EFFECT | 247 | 250 |
| EFFECT_DESTROY_REPLACE | 247 | 248 |
| EFFECT_EQUIP_LIMIT | 245 | 251 |
| EVENT_BE_MATERIAL | 237 | 239 |
| EFFECT_UPDATE_LEVEL | 227 | 239 |
| EFFECT_CHANGE_LEVEL | 205 | 214 |
| EVENT_BATTLE_DAMAGE | 204 | 207 |
| EVENT_REMOVE | 200 | 202 |
| EFFECT_DIRECT_ATTACK | 195 | 195 |
| EFFECT_PIERCE | 187 | 188 |
| EFFECT_CHANGE_CODE | 174 | 175 |
| EFFECT_MATERIAL_CHECK | 174 | 174 |
| EVENT_FLIP_SUMMON_SUCCESS | 171 | 172 |
| EVENT_BATTLED | 160 | 163 |
| EFFECT_EXTRA_ATTACK | 160 | 160 |
| EVENT_BE_BATTLE_TARGET | 150 | 151 |
| EVENT_DAMAGE_STEP_END | 146 | 151 |
| PHASE_BATTLE | 143 | 144 |
| EVENT_CHAIN_SOLVING | 142 | 144 |
| EFFECT_AVOID_BATTLE_DAMAGE | 136 | 136 |
| EFFECT_CANNOT_SUMMON | 135 | 136 |
| EVENT_PRE_DAMAGE_CALCULATE | 134 | 134 |
| EVENT_CHAIN_SOLVED | 133 | 159 |

### SetProperty (top 50 of 28 distinct)

| token | files | occurrences |
|---|---:|---:|
| EFFECT_FLAG_CARD_TARGET | 5129 | 5690 |
| EFFECT_FLAG_DELAY | 3143 | 3667 |
| EFFECT_FLAG_CANNOT_DISABLE | 2647 | 3203 |
| EFFECT_FLAG_PLAYER_TARGET | 2311 | 2633 |
| EFFECT_FLAG_CLIENT_HINT | 1616 | 1733 |
| EFFECT_FLAG_DAMAGE_STEP | 1517 | 1618 |
| EFFECT_FLAG_SINGLE_RANGE | 1439 | 1653 |
| EFFECT_FLAG_UNCOPYABLE | 1389 | 1613 |
| EFFECT_FLAG_IGNORE_IMMUNE | 631 | 663 |
| EFFECT_FLAG_OATH | 363 | 395 |
| EFFECT_FLAG_DAMAGE_CAL | 312 | 317 |
| EFFECT_FLAG_SET_AVAILABLE | 245 | 265 |
| EFFECT_FLAG2_CHECK_SIMULTANEOUS | 138 | 138 |
| EFFECT_FLAG_COPY_INHERIT | 85 | 87 |
| EFFECT_FLAG_OWNER_RELATE | 60 | 66 |
| EFFECT_FLAG_NO_TURN_RESET | 58 | 58 |
| EFFECT_FLAG_CANNOT_NEGATE | 56 | 57 |
| EFFECT_FLAG_IGNORE_RANGE | 46 | 46 |
| EFFECT_FLAG_SPSUM_PARAM | 33 | 33 |
| EFFECT_FLAG_EVENT_PLAYER | 24 | 24 |
| EFFECT_FLAG_BOTH_SIDE | 13 | 18 |
| EFFECT_FLAG_CANNOT_INACTIVATE | 13 | 14 |
| EFFECT_FLAG_REPEAT | 7 | 7 |
| EFFECT_FLAG_LIMIT_ZONE | 5 | 5 |
| EFFECT_CANNOT_DISABLE | 3 | 3 |
| EFFECT_FLAG_CANNOT_NEGATE_ACTIV_EFF | 2 | 4 |
| EFFECT | 1 | 1 |
| EFFECT_FLAG_ABSOLUTE_TARGET | 1 | 1 |

### SetRange (top 50 of 14 distinct)

| token | files | occurrences |
|---|---:|---:|
| LOCATION_MZONE | 5596 | 7959 |
| LOCATION_HAND | 2019 | 2082 |
| LOCATION_GRAVE | 1802 | 1832 |
| LOCATION_SZONE | 1305 | 2102 |
| LOCATION_PZONE | 375 | 479 |
| LOCATION_FZONE | 334 | 705 |
| LOCATION_EXTRA | 75 | 75 |
| LOCATION_REMOVED | 60 | 61 |
| LOCATION_DECK | 33 | 33 |
| LOCATION_EMZONE | 11 | 13 |
| LOCATION_MMZONE | 6 | 6 |
| LOCATION_STZONE | 3 | 8 |
| LOCATION_ONFIELD | 3 | 4 |
| LOCATION_ALL | 1 | 1 |

### SetCountLimit (top 50 of 4 distinct)

| token | files | occurrences |
|---|---:|---:|
| EFFECT_COUNT_CODE_OATH | 1122 | 1130 |
| EFFECT_COUNT_CODE_DUEL | 60 | 64 |
| EFFECT_COUNT_CODE_SINGLE | 57 | 75 |
| EFFECT_COUNT_CODE_CHAIN | 46 | 47 |

## 5. Effects per card (`Effect.CreateEffect`)

- Files: 13541
- Total effects: 34563
- **Mean effects per card: 2.5525**
- Max: 14

| effects | files |
|---|---:|
| 0 | 185 |
| 1 | 3252 |
| 2 | 3897 |
| 3 | 3191 |
| 4 | 1815 |
| 5 | 775 |
| 6 | 287 |
| 7 | 77 |
| 8 | 35 |
| 9 | 16 |
| 10+ | 11 |

## 6. Duel API coverage curve

Duel functions ranked by file count; for each N, the share of card files whose entire set of `Duel.*` calls lies within the top-N set.

- Files total: 13541 (with zero Duel calls: 597)
- Distinct Duel functions: 237

| N | covered files | % of all files | % of files with Duel calls |
|---:|---:|---:|---:|
| 10 | 1453 | 10.73% | 6.61% |
| 25 | 3756 | 27.74% | 24.41% |
| 50 | 7476 | 55.21% | 53.14% |
| 100 | 11658 | 86.09% | 85.45% |
| 150 | 13067 | 96.50% | 96.34% |
| 200 | 13499 | 99.69% | 99.68% |
| 300 | 13541 | 100.00% | 100.00% |

*Files with zero Duel calls count as covered in the first column (vacuously).*
