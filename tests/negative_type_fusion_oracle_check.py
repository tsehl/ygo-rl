"""Une recette combinant inclusion et exclusion de type doit filtrer ses matériaux."""
from pathlib import Path
import sys,tempfile
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cards,cardgen,tolua,driver,compare
B=600960000
base=dict(ctype=core.TYPE_MONSTER|core.TYPE_NORMAL,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
mat=dict(typ=core.TYPE_MONSTER,not_typ=cards.TYPE_EFFECT)
specs=[dict(base,code=B),dict(base,code=B+1,ctype=core.TYPE_MONSTER|cards.TYPE_EFFECT),
       dict(base,code=B+2,ctype=core.TYPE_MONSTER|core.TYPE_FUSION|cards.TYPE_EFFECT,lv=6,recipe=('fusion',[mat,mat])),
       dict(base,code=B+3,ctype=core.TYPE_SPELL,lv=0,atk=0,df=0,effects=[dict(kind='fusion_spell',f=dict(me=core.Z_EXTRA,fus=1))])]
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for material,expected in ((B,True),(B+1,False)):
        for seed in (3100,3101):
            def chooser(dec,indices):
                for kind in ('activate','summon','to_ep','pass'):
                    for i in indices:
                        if dec.options[i][0]==kind:return i
                return indices[0]
            deck=[material,B+3]*20
            result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,extra_capacity=15,
                                given=([deck,[B]*40],[[B+2],[]]))
            assert result['ok'],(material,seed,result.get('err'))
            resolved=sum(e['count'] for e in result['coverage'] if e['code']==B+3 and e['stage']=='resolved')
            summoned=sum(e['count'] for e in result['coverage'] if e['code']==B+2 and e['stage']=='summoned')
            assert (resolved>0)==expected and (summoned>0)==expected,(material,seed,resolved,summoned)
            print(material,seed,'identique :',resolved,'polymérisations,',summoned,'fusions',flush=True)
