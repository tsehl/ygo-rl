"""Arnis : raison de combat et contrôleur de l'attaquant, script officiel."""
from pathlib import Path
import sys
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from oracle_bank import build
import core,compare
ARNIS,TARGET,STRONG,EQUAL=29687169,2468169,4148264,5053103
db,bank,cmap=build([ARNIS,TARGET,STRONG,EQUAL])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for attacker,expected,opponent in ((1,True,STRONG),(0,False,STRONG),(1,True,EQUAL)):
    total=0
    for seed in (2800,2801):
        def chooser(dec,indices):
            for kind in ('chain','summon','attack','to_bp','to_ep','pass'):
                for i in indices:
                    option=dec.options[i]
                    if option[0]!=kind:continue
                    if kind=='summon' and dec.player==0 and option[1]!=ARNIS:continue
                    if kind in ('attack','to_bp') and dec.player!=attacker:continue
                    return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',chooser=chooser,
            given=([[ARNIS,TARGET]*20,[opponent]*40],[[],[]]))
        assert result['ok'],(attacker,seed,result.get('err'))
        resolved=sum(e['count'] for e in result['coverage'] if e['code']==ARNIS and e['stage']=='resolved')
        total+=resolved
    assert (total>0)==expected,(attacker,total)
    print('attaquant',attacker,'adversaire',opponent,'2/2 identiques,',total,'effets résolus',flush=True)
