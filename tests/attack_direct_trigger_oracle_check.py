"""Un déclencheur à la déclaration d'attaque doit se résoudre aussi en attaque directe."""
from pathlib import Path
import sys,tempfile
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare
B=600965000
base=dict(ctype=33,atk=1000,df=1000,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,recipe=None)
effect=dict(kind='trigger',on=1<<core.EV_ATTACK,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=None,clauses=[dict(verb='recover',v=100)])
specs=[dict(base,code=B,effects=[effect]),dict(base,code=B+1,ctype=17,effects=[])]
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (3300,3301):
        def chooser(dec,indices):
            order=('summon','to_bp','attack','to_ep','pass') if dec.player==0 else ('to_ep','pass')
            for kind in order:
                for i in indices:
                    if dec.options[i][0]==kind:return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
                            given=([[B]*40,[B+1]*40],[[],[]]))
        assert result['ok'],(seed,result.get('err'))
        resolved=sum(e['count'] for e in result['coverage'] if e['code']==B and e['stage']=='resolved')
        assert resolved>0,(seed,resolved)
        print(seed,'identique, déclencheur d’attaque directe résolu',resolved,'fois',flush=True)
