"""Banque des seules cartes d'un scénario oracle : mêmes traductions que validate.py."""
from pathlib import Path
import sys
import numpy as np
from contextlib import nullcontext
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'translate'),str(ROOT/'oracle')]
import core,cards,translate,compare,driver


def build(codes, fixtures=()):
    directory=Path(__file__).parent/'fixtures/fusion'
    for code in fixtures:
        driver.TEST_CARDS[code]=((0,0,65,3000,2500,8,1,1),f'Test Fusion {code}')
    if fixtures: driver.SCRIPTS.insert(0,str(directory))
    db=driver.CardDB()
    codes=sorted(set(codes))
    recipes=dict(fusion={},synchro={},xyz={},link={})
    effects={}; descriptions={}
    for code in codes:
        if db.rows[code][3] in (17,4113):
            effects[code]=[];descriptions[code]=[];continue
        with patch.object(translate,'OFFICIAL',str(directory)) if code in fixtures else nullcontext():
            eff,_,proc,desc=translate.translate(code)
        effects[code]=eff;descriptions[code]=desc
        if proc:
            kind,f,*rest=proc
            recipes[kind][code]=cards.FusionRecipe(f,rest[0]) if kind=='fusion' else ((f,rest[0],rest[1],0) if kind=='link' else (f,rest[0]))
    rows=[db.rows[c] for c in codes]
    bank=core.make_bank(np.array(codes),[r[4] for r in rows],[r[5] for r in rows],[r[6]&255 for r in rows],
                        ctype=[r[3] for r in rows],race=[r[7] for r in rows],attr=[r[8] for r in rows],
                        setcode=[cards.setcodes(r[2]) for r in rows],lsc=[r[6]>>24&255 for r in rows],rsc=[r[6]>>16&255 for r in rows],
                        **cards.assemble(codes,cards=effects,ritual={},**recipes))
    compare.DESC.clear();compare.DESC.update(descriptions)
    compare.EFFECT_RANGES.clear();compare.EFFECT_RANGES.update({c:[e[3] for e in eff] for c,eff in effects.items()})
    compare.EFFECT_KINDS.clear();compare.EFFECT_KINDS.update({c:[e[0] for e in eff] for c,eff in effects.items()})
    compare.EFFECT_EVENTS.clear();compare.EFFECT_EVENTS.update({c:[e[1] for e in eff] for c,eff in effects.items()})
    return db,bank,{c:i for i,c in enumerate(codes)}
