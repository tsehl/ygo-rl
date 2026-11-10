"""La disjonction de raisons se compile en un masque existant, sans accepter les mélanges."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T,core
tr=T.Translator.__new__(T.Translator)
def term(source):
    fl=T.Filt();node=T.LA.parse('return '+source).body.body[0].values[0]
    tr.filter_term(fl,node,{},'c');return fl
f=term('c:IsReason(REASON_DESTROY) or c:IsReason(REASON_COST)')
assert f.build(core.Z_GRAVE,0)[core.F_REASON]==core.REASON_DESTROY|core.REASON_COST
f.set('reason',core.REASON_DESTROY)
assert f.d['reason']==core.REASON_DESTROY
for text in ('c:IsReason(REASON_COST) or c:IsType(TYPE_MONSTER)',
             'c:IsReason(REASON_COST) or not c:IsType(TYPE_EFFECT)'):
    try:term(text)
    except T.Unsupported:pass
    else:raise AssertionError(text)
print('Union et simplification des raisons correctes ; mélanges non représentables refusés')
