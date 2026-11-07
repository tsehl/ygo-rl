"""La protection temporaire doit être réellement confrontée à des attaques."""
import jax
from oracle_bank import build
import core,compare
from protection_policy import chooser_for
DRAW,CANDLE,ENEMY=41367003,47695416,85309439
db,bank,cmap=build([DRAW,CANDLE,ENEMY])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for seed in (1000,1001):
    protected_attacks=[0]
    result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',
                        given=([[DRAW,CANDLE]*20,[ENEMY]*40],[[],[]]),chooser=chooser_for(CANDLE,protected_attacks),extra_capacity=15)
    assert result['ok'],(seed,result.get('err'))
    resolved=sum(e['count'] for e in result['coverage'] if e['code']==DRAW and e['stage']=='resolved')
    assert resolved>0 and protected_attacks[0]>0,(seed,resolved,protected_attacks)
    print(f'{seed}: identique, {resolved} résolutions, {protected_attacks[0]} attaques vers un monstre protégé',flush=True)
