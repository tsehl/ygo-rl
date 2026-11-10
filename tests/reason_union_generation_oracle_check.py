"""Une union de raisons reconnaît destruction ou coût, mais pas un envoi simple."""
from pathlib import Path
import sys,tempfile
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare
B=600990000
base=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
def act(clause):return dict(kind='act',on=0,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=None,clauses=[clause])
specs=[dict(base,code=B)]
for i,verb in enumerate(('destroy','tograve')):
    specs.append(dict(base,code=B+1+i,ctype=core.TYPE_SPELL,atk=0,df=0,lv=0,
        effects=[act(dict(verb=verb,sel='all',f=dict(me=core.Z_MZONE),k=1,v=0))]))
specs.append(dict(base,code=B+3,ctype=core.TYPE_SPELL,atk=0,df=0,lv=0,
    effects=[act(dict(verb='tohand',sel='pick',f=dict(me=core.Z_GRAVE,typ=core.TYPE_MONSTER,
        reason=(core.REASON_COST|core.REASON_DESTROY)),k=1,v=0))]))
specs.append(dict(base,code=B+4,ctype=core.TYPE_MONSTER|cardgen.TYPE_EFFECT,
    effects=[dict(kind='ignition',on=0,opt=0,optional=False,rng=0,dmg=0,conds=[],cost=('self',),
                  clauses=[dict(verb='recover',v=100)])]))
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for cause,expected in ((B+1,True),(B+2,False),(B+4,True)):
        for seed in (2900,2901):
            def chooser(dec,indices):
                for kind in ('activate','summon','to_ep','pass'):
                    candidates=[i for i in indices if dec.options[i][0]==kind]
                    if kind=='activate':
                        candidates.sort(key=lambda i: dec.options[i][1]!=B+3)
                    if kind=='summon':
                        candidates.sort(key=lambda i: dec.options[i][1]!=cause)
                    if candidates:return candidates[0]
                return indices[0]
            deck=[B,cause,B+3,B]*10
            result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
                given=([deck,[B]*40],[[],[]]))
            assert result['ok'],(cause,seed,result.get('err'))
            caused=sum(e['count'] for e in result['coverage'] if e['code']==cause and e['stage']=='resolved')
            searched=sum(e['count'] for e in result['coverage'] if e['code']==B+3 and e['stage']=='resolved')
            assert caused>0 and (searched>0)==expected,(cause,seed,caused,searched)
            print(cause,seed,'identique :',caused,'déplacements,',searched,'effets filtrés résolus',flush=True)
