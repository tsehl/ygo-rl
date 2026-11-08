"""Deux marqueurs génériques distincts : seule la condition correspondante doit se résoudre."""
from pathlib import Path
import sys,tempfile
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare
B=600970000
base=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
trigger=dict(kind='trigger',on=1<<core.EV_SPSUMMON,opt=0,optional=False,rng=0,dmg=0,
    conds=[('self',0,0,core.summon_mask(151))],cost=None,clauses=[dict(verb='recover',v=100)])
specs=[dict(base,code=B),dict(base,code=B+1,ctype=33,effects=[trigger])]
for i,tag in enumerate((151,152)):
    effect=dict(kind='act',on=0,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=None,
        clauses=[dict(verb='ss',sel='pick',f=dict(me=core.Z_HAND,code=B+1,spsum=1),k=1,v=0,pos=core.P_FU_ATK,tag=tag)])
    specs.append(dict(base,code=B+2+i,ctype=core.TYPE_SPELL,lv=0,atk=0,df=0,effects=[effect]))
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for i,tag in enumerate((151,152)):
        resolved=summoned=0
        for seed in (2500,2501):
            def chooser(dec,indices):
                state=compare.LAST_STATE
                target=(np.asarray(state.cid)==cmap[B+1])&(np.asarray(state.zone)==core.Z_MZONE)
                if target.any():
                    assert not np.asarray(state.proc)[target].any()
                    assert (np.asarray(state.suminfo)[target]&65535==tag).all()
                for kind in ('chain','activate','summon','to_ep','pass'):
                    for index in indices:
                        option=dec.options[index]
                        if option[0]!=kind:continue
                        if kind=='summon' and option[1]==B+1:continue
                        return index
                return indices[0]
            result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
                given=([[B+1,B+2+i]*20,[B]*40],[[],[]]))
            assert result['ok'],(tag,seed,result.get('err'))
            resolved+=sum(e['count'] for e in result['coverage'] if e['code']==B+1 and e['stage']=='resolved')
            summoned+=sum(e['count'] for e in result['coverage'] if e['code']==B+1 and e['stage']=='summoned')
        assert summoned>0 and (resolved>0)==(tag==151),(tag,summoned,resolved)
        print(tag,'2/2 identiques,',summoned,'invocations,',resolved,'conditions résolues',flush=True)
