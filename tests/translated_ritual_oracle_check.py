"""Procédures et monstres rituels officiels, sans les implémentations manuelles."""
import jax
from oracle_bank import build
import core,compare
BLACK,DEMISE,GREATER,EQUAL,MATERIAL,ELF=5405694,72426662,55761792,8198712,49003308,44287299
db,bank,cmap=build([BLACK,DEMISE,GREATER,EQUAL,MATERIAL,ELF])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for monster,spell in ((BLACK,GREATER),(DEMISE,EQUAL)):
    summons=effects=0
    for seed in (900,901):
        def chooser(dec,indices):
            for kind in ('activate','spsummon','summon','attack','to_bp'):
                for i in indices:
                    if dec.options[i][0]==kind:return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',
                            given=([[monster,spell,MATERIAL,MATERIAL]*10,[ELF]*40],[[],[]]),chooser=chooser)
        assert result['ok'],(monster,seed,result.get('err'))
        summons+=sum(e['count'] for e in result['coverage'] if e['code']==monster and e['stage']=='summoned')
        effects+=sum(e['count'] for e in result['coverage'] if e['code']==monster and e['stage']=='resolved')
    assert summons>0,(monster,summons)
    if monster==DEMISE:assert effects>0,(monster,effects)
    print(f'{monster}: 2/2 identiques, {summons} invocations rituelles, {effects} effets résolus',flush=True)
