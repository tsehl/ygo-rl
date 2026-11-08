"""Rechercher cinq noms de magie différents doit garder les cinq interdictions actives."""
from pathlib import Path
import sys,tempfile
import numpy as np
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare

BASE=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
EFFECT=dict(kind='act',on=0,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=None,clauses=[dict(verb='recover',v=100)])
NORMAL,SEARCH=600940000,600940001
spells=list(range(600940002,600940007))
flush=dict(EFFECT,kind='ignition',rng=core.Z_MZONE,clauses=[dict(verb='tograve',sel='all',f=dict(me=core.Z_HAND),k=1,v=0)])
search=dict(EFFECT,kind='ignition',rng=core.Z_MZONE,
    clauses=[dict(verb='tohand',sel='pick',f=dict(me=core.Z_DECK,typ=core.TYPE_SPELL),k=1,v=0,lock=True)])
specs=[dict(BASE,code=NORMAL),dict(BASE,code=SEARCH,ctype=33,effects=[search,flush])]
specs += [dict(BASE,code=c,ctype=core.TYPE_SPELL,atk=0,df=0,lv=0,effects=[EFFECT]) for c in spells]
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
minimum=5
max_candidates=0
ctypes=np.asarray(bank.ctype)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (1900,1901):
        found=set()
        def chooser(dec,indices):
            global max_candidates
            state=compare.LAST_STATE
            count=0
            if int(state.dec)==core.D_IDLE:
                hand=(np.asarray(state.zone)==core.Z_HAND)&(np.asarray(state.ctrl)==dec.player)
                count=int((hand&((ctypes[np.asarray(state.cid)]&core.TYPE_SPELL)!=0)).sum())
                max_candidates=max(max_candidates,count)
            if dec.options[0][0]=='cards':
                choices=[i for i in indices if len(dec.options[i][1])==1 and dec.options[i][1][0][0] in spells]
                for i in choices:
                    code=dec.options[i][1][0][0]
                    if code not in found:found.add(code);return i
                if choices:return choices[0]
            for kind in ('activate','summon','to_ep','pass'):
                for i in indices:
                    option=dec.options[i]
                    if option[0]!=kind:continue
                    if kind=='activate':
                        if option[1]!=SEARCH:continue
                        # À 25 magies, vider la main évite d'énumérer des centaines
                        # de milliers de choix de défausse en End Phase dans le pilote.
                        effect=int(option[-1])&0xfffff
                        if effect!=(1 if count>core.KC else 0):continue
                    return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
            given=([spells*6+[SEARCH]*10,[NORMAL]*40],[[],[]]))
        assert result['ok'],(seed,result.get('err'))
        minimum=min(minimum,len(found))
assert minimum==5,minimum
assert max_candidates>core.KC,max_candidates
print(f'2/2 identiques, cinq noms interdits persistants ; jusqu’à {max_candidates} magies en main candidates',flush=True)
