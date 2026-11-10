"""Seuls les matériaux nommés sans filtre supplémentaire autorisent un substitut."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen')]
import core,cards,cardgen
base=dict(ctype=core.TYPE_MONSTER|core.TYPE_FUSION,atk=2000,df=1000,lv=6,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[])
recipe=[{'code':101},{'code':102,'not_typ':cards.TYPE_EFFECT},{'code':(103,104)},{'typ':core.TYPE_MONSTER}]
spec=dict(base,code=100,recipe=('fusion',recipe))
bank=cardgen.to_bank([spec])
assert int(np.asarray(bank.fsub)[0])==1,int(np.asarray(bank.fsub)[0])
print('Masque de substitution : seul le nom brut est autorisé')
