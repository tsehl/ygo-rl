"""Une interdiction apparue en réponse à Monster Reborn doit empêcher son invocation."""
from pathlib import Path
import sys,tempfile
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cards,cardgen,tolua,driver,compare
B,REBORN=600960000,83764718
base=dict(ctype=33,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
effect=dict(kind='ignition',on=0,opt=core.OPT_NAME,optional=False,rng=core.Z_MZONE,dmg=0,
    conds=[],cost=('self',),clauses=[dict(verb='recover',v=100)])
quick=dict(effect,kind='quick',opt=core.OPT_CARD,cost=None,conds=[('chain',core.TYPE_SPELL)],
    clauses=[dict(verb='ss',sel='pick',f=dict(me=core.Z_HAND,code=B+2,spsum=1),k=1,v=0,pos=core.P_FU_ATK)])
allz=core.Z_HAND|core.Z_DECK|core.Z_MZONE|core.Z_SZONE|core.Z_GRAVE|core.Z_REMOVED|core.Z_EXTRA
block=dict(kind='cont',rng=core.Z_MZONE,cont=dict(code=core.MC_NO_SS,val=0,
    tgt=dict(me=allz,op=allz),per=None,cnt=None,tp=0,ph=0))
specs=[dict(base,code=B,effects=[effect]),dict(base,code=B+1,effects=[quick]),dict(base,code=B+2,effects=[block])]
codes=[B,B+1,B+2,REBORN];effects={c['code']:[cardgen.bricks(e) for e in c['effects']] for c in specs}
effects[REBORN]=cards.CARDS[REBORN]
bank=core.make_bank(np.array(codes),[500,500,500,0],[500,500,500,0],[4,4,4,0],ctype=[33,33,33,core.TYPE_SPELL],
    **cards.assemble(codes,cards=effects,ritual={},fusion={},synchro={},xyz={},link={}))
cmap={c:i for i,c in enumerate(codes)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({code:[e[0] for e in eff] for code,eff in effects.items()})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (2400,2401):
        def chooser(dec,indices):
            if dec.options[0][0]=='cards':
                for i in indices:
                    selected=dec.options[i][1]
                    if len(selected)==1 and selected[0][0]==B:return i
            for kind in ('chain','activate','summon','to_ep','pass'):
                for i in indices:
                    option=dec.options[i]
                    if option[0]!=kind:continue
                    if kind=='summon' and option[1]!=(B if dec.player==0 else B+1):continue
                    return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
            given=([[B,REBORN]*20,[B+1,B+2]*20],[[],[]]))
        assert result['ok'],(seed,result.get('err'))
        summoned=sum(e['count'] for e in result['coverage'] if e['code']==B+2 and e['stage']=='summoned')
        replied=sum(e['count'] for e in result['coverage'] if e['code']==B+1 and e['stage']=='resolved')
        assert summoned>0 and replied>0,(seed,summoned,replied)
        print(seed,'identique,',replied,'réponses invoquant une interdiction avant Reborn',flush=True)
