"""Protection temporaire procédurale, avec Lua indépendant et attaques réellement exercées."""
import sys,tempfile
from pathlib import Path
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'oracle'),str(ROOT/'gen')]
import core,cardgen,tolua,compare,driver
from protection_policy import chooser_for

base=dict(ctype=17,atk=600,df=600,lv=2,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
clause=dict(verb='indes_battle',sel='target',f=dict(me=core.Z_MZONE,pos=core.P_FACEUP),k=1,v=0)
effect=dict(kind='act',on=0,opt=core.OPT_NAME,optional=False,rng=0,dmg=0,conds=[],cost=None,clauses=[clause])
SPELL,OWN,ENEMY=600910000,600910001,600910002
specs=[dict(base,code=SPELL,ctype=core.TYPE_SPELL|core.TYPE_QUICKPLAY,atk=0,df=0,lv=0,effects=[effect]),
       dict(base,code=OWN),dict(base,code=ENEMY,atk=1000,df=1000,lv=3)]
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (1100,1101):
        protected=[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',
                            given=([[SPELL,OWN]*20,[ENEMY]*40],[[],[]]),chooser=chooser_for(OWN,protected),extra_capacity=15)
        assert result['ok'],(seed,result.get('err'))
        resolved=sum(e['count'] for e in result['coverage'] if e['code']==SPELL and e['stage']=='resolved')
        assert resolved>0 and protected[0]>0,(seed,resolved,protected)
        print(f'{seed}: identique, {resolved} résolutions, {protected[0]} cibles protégées attaquées',flush=True)
