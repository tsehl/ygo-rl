"""Chaque famille de paiement doit être effectivement jouée, pas seulement générée."""
import sys,tempfile
from pathlib import Path
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'oracle'),str(ROOT/'gen')]
import core,cardgen,tolua,driver,compare

base=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
specs=[dict(base,code=600900000),dict(base,code=600900001,atk=2500)]
for i,cost in enumerate((('lp',500),('discard',),('banish_gy',))):
    effect=dict(kind='hand_proc',on=0,opt=core.OPT_NAME,optional=False,rng=core.Z_HAND,dmg=0,conds=[],
                cost=cost,clauses=[],counts=[(dict(me=core.Z_MZONE),0,0)])
    specs.append(dict(base,ctype=33,code=600900002+i,atk=1800,effects=[effect]))
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c["code"]:[cardgen.bricks(e)[0] for e in c["effects"]] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for spec in specs[2:]:
        code=spec['code'];count=0
        for seed in (700,701):
            def chooser(dec,indices):
                for kind in ('spsummon','summon','attack','to_bp'):
                    for index in indices:
                        if dec.options[index][0]==kind:return index
                return indices[0]
            result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',
                                given=([[code,600900000]*20,[600900001]*40],[[],[]]),chooser=chooser)
            assert result['ok'],(code,seed,result.get('err'))
            count+=sum(t['option'][0]=='spsummon' and t['option'][1]==code and t['option'][3]=='hand' for t in result['trace'])
        assert count>0,(code,spec['effects'][0]['cost'])
        print(f"{spec['effects'][0]['cost']}: 2/2 identiques, {count} procédures avec paiement",flush=True)
