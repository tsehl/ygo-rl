"""Vampire Dragon garde son invocation sacrifice au cimetière, puis la perd à la réanimation."""
import jax,numpy as np
from oracle_bank import build
import core,compare
DRAGON,REBORN,HOLE,ELF=36352429,83764718,53129443,44287299
db,bank,cmap=build([DRAGON,REBORN,HOLE,ELF])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for seed in (2200,2201):
    seen=set()
    def chooser(dec,indices):
        state=compare.LAST_STATE
        dragon=np.asarray(state.cid)==cmap[DRAGON]
        field=dragon&(np.asarray(state.zone)==core.Z_MZONE)
        grave=dragon&(np.asarray(state.zone)==core.Z_GRAVE)
        tribute=np.asarray(core.summon_matches(state.suminfo,core.summon_mask(core.SUMMON_TRIBUTE)))
        if (field&tribute).any():seen.add('tribute')
        if (grave&tribute).any():seen.add('tribute grave')
        if (field&~tribute).any():seen.add('revived')
        if 'revived' in seen and (grave&~tribute).any():seen.add('revived grave')
        if dec.options[0][0]=='cards':
            for i in indices:
                cards=dec.options[i][1]
                if len(cards)==1 and cards[0][0]==DRAGON:return i
        for kind in ('chain','yes','activate','summon','to_ep','pass'):
            for i in indices:
                option=dec.options[i]
                if option[0]!=kind:continue
                if kind=='activate':
                    if option[1]==HOLE and not field.any():continue
                    if option[1]==REBORN and not grave.any():continue
                return i
        return indices[0]
    result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',chooser=chooser,
        given=([[DRAGON,REBORN,HOLE,ELF]*10,[ELF]*40],[[],[]]))
    assert result['ok'],(seed,result.get('err'))
    assert seen=={'tribute','tribute grave','revived','revived grave'},(seed,seen)
    resolved=sum(e['count'] for e in result['coverage'] if e['code']==DRAGON and e['stage']=='resolved')
    assert resolved>0,(seed,resolved)
    print(seed,'identique, sacrifice → cimetière → réanimation → cimetière,',resolved,'effets résolus',flush=True)
