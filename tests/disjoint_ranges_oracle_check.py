"""Deux effets sans description, identifiés par leur portée déclarée."""
import jax
from oracle_bank import build
import core,compare
PURPLE,ELF=48427163,44287299
db,bank,cmap=build([PURPLE,ELF])
step,legal=jax.jit(core.step),jax.jit(core.legal)
uses={'hand':0,'mzone':0,'grave':0}
for seed in (1200,1201):
    def chooser(dec,indices):
        for kind in ('activate','summon','attack','to_bp'):
            for i in indices:
                if dec.options[i][0]==kind:return i
        return indices[0]
    result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',
                        given=([[PURPLE]*40,[ELF]*40],[[],[]]),chooser=chooser,extra_capacity=15)
    assert result['ok'],(seed,result.get('err'))
    for item in result['trace']:
        option=item['option']
        if option[0]=='activate' and option[1]==PURPLE:uses[option[3]]+=1
assert uses['hand']>0 and uses['grave']>0,uses
print(f'2/2 identiques, activations par zone : {uses}',flush=True)
