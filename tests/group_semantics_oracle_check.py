"""Références de groupe après mouvement et conditions sur un groupe vide."""
from pathlib import Path
from unittest.mock import patch
import jax
from oracle_bank import build
import core,compare,driver,translate
folder=Path(__file__).parent/'fixtures/groups'
codes=(999999981,999999982,999999983)
for code in codes:
    driver.TEST_CARDS[code]=((0,0,2,0,0,0,0,0),f'Group semantics {code}')
driver.SCRIPTS.insert(0,str(folder))
with patch.object(translate,'OFFICIAL',str(folder)):
    db,bank,cmap=build([*codes,44287299])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for code in codes:
    count=0
    for seed in (600,601):
        def chooser(dec,indices):
            for kind in ('activate','summon','attack','to_bp'):
                for i in indices:
                    if dec.options[i][0]==kind:return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',given=([[code,44287299]*20,[44287299]*40],[[],[]]),chooser=chooser)
        assert result['ok'],(code,seed,result.get('err'))
        count+=sum(e['count'] for e in result['coverage'] if e['code']==code and e['stage']=='resolved')
    assert count>0,(code,count)
    print(f'{code}: 2/2 identiques, {count} résolutions',flush=True)
