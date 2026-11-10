"""Un filtre de type exclu sélectionne le monstre normal, pas le monstre à effet."""
from pathlib import Path
import sys, tempfile
import jax, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cards,cardgen,tolua,driver,compare
B=600970000
base=dict(ctype=core.TYPE_MONSTER|core.TYPE_NORMAL,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
def act(clause):return dict(kind='act',on=0,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=None,clauses=[clause])
specs=[dict(base,code=B),dict(base,code=B+1,ctype=core.TYPE_MONSTER|cards.TYPE_EFFECT),
       dict(base,code=B+2,ctype=core.TYPE_SPELL,atk=0,df=0,lv=0,effects=[act(dict(verb='tograve',sel='all',f=dict(me=core.Z_MZONE),k=1,v=0))]),
       dict(base,code=B+3,ctype=core.TYPE_SPELL,atk=0,df=0,lv=0,effects=[act(dict(verb='tohand',sel='pick',f=dict(me=core.Z_GRAVE,typ=core.TYPE_MONSTER,not_typ=cards.TYPE_EFFECT),k=1,v=0))])]
bank=cardgen.to_bank(specs); cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (3000,3001):
        seen=[False]
        def chooser(dec,indices):
            state=compare.LAST_STATE
            zone=np.asarray(state.zone); ids=np.asarray(state.cid)
            field={int(np.asarray(bank.code)[ids[i]]) for i in np.flatnonzero(zone==core.Z_MZONE)}
            grave={int(np.asarray(bank.code)[ids[i]]) for i in np.flatnonzero(zone==core.Z_GRAVE)}
            if B in grave and B+1 in grave:seen[0]=True
            for kind in ('activate','summon','to_ep','pass'):
                for i in indices:
                    option=dec.options[i]
                    if option[0]!=kind:continue
                    if kind=='activate' and option[1]==B+2 and not {B,B+1} <= field:continue
                    if kind=='activate' and option[1]==B+3 and not seen[0]:continue
                    return i
            return indices[0]
        deck=[B,B+1,B+2,B+3]*10
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
                            given=([deck,[B]*40],[[],[]]))
        assert result['ok'],(seed,result.get('err'))
        resolved=sum(e['count'] for e in result['coverage'] if e['code']==B+3 and e['stage']=='resolved')
        assert seen[0] and resolved>0,(seed,seen,resolved)
        print(seed,'identique : types normal et effet au cimetière,',resolved,'recherches résolues',flush=True)
