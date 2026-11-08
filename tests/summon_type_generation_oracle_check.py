"""Les cinq invocations de procédure doivent réellement résoudre leur condition de type."""
from pathlib import Path
import sys,tempfile
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare
B=600950000
base=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
specs=[dict(base,code=B),dict(base,code=B+1,ctype=17|core.TYPE_TUNER,lv=2)]
variants=[('fusion',core.TYPE_FUSION,core.SUMMON_FUSION,6,[{},{}]),
          ('synchro',core.TYPE_SYNCHRO,core.SUMMON_SYNCHRO,6,({},{})),
          ('xyz',core.TYPE_XYZ,core.SUMMON_XYZ,4,({},2)),
          ('link',core.TYPE_LINK,core.SUMMON_LINK,2,({},2,2,0)),
          ('ritual',core.TYPE_RITUAL,core.SUMMON_RITUAL,6,None)]
for i,(kind,flag,typ,lv,recipe) in enumerate(variants):
    effect=dict(kind='trigger',on=1<<core.EV_SPSUMMON,opt=0,optional=False,rng=0,dmg=0,
        conds=[('self',0,0,core.summon_mask(typ))],cost=None,clauses=[dict(verb='recover',v=100)])
    specs.append(dict(base,code=B+2+i,ctype=core.TYPE_MONSTER|cardgen.TYPE_EFFECT|flag,lv=lv,
        df=5 if kind=='link' else 500,recipe=(kind,recipe),effects=[effect]))
specs.append(dict(base,code=B+7,ctype=core.TYPE_SPELL,lv=0,atk=0,df=0,
    effects=[dict(kind='fusion_spell',f=dict(me=core.Z_EXTRA,fus=1))]))
specs.append(dict(base,code=B+8,ctype=core.TYPE_SPELL|core.TYPE_RITUAL,lv=0,atk=0,df=0,
    effects=[dict(kind='ritual',code=B+6,mode=core.RIT_GREATER)]))
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for i,(kind,flag,typ,lv,recipe) in enumerate(variants):
        monster=B+2+i;resolved=0;summons=0
        main=([B,B,B+7,B]*10 if kind=='fusion' else [B,B+1]*20 if kind=='synchro'
              else [B,B,B+6,B+8]*10 if kind=='ritual' else [B]*40)
        extras=[] if kind=='ritual' else [monster]
        for seed in (2300,2301):
            def chooser(dec,indices):
                state=compare.LAST_STATE
                selected=(np.asarray(state.cid)==cmap[monster])&(np.asarray(state.zone)==core.Z_MZONE)
                if selected.any():
                    assert np.asarray(core.summon_matches(state.suminfo,core.summon_mask(typ)))[selected].all(),kind
                for label in ('spsummon','activate','summon','to_ep','pass'):
                    for index in indices:
                        if dec.options[index][0]==label:return index
                return indices[0]
            result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,extra_capacity=15,
                given=([main,[B]*40],[extras,[]]))
            assert result['ok'],(kind,seed,result.get('err'))
            resolved+=sum(e['count'] for e in result['coverage'] if e['code']==monster and e['stage']=='resolved')
            summons+=sum(e['count'] for e in result['coverage'] if e['code']==monster and e['stage']=='summoned')
        assert resolved>0 and summons>0,(kind,resolved,summons)
        print(kind,'2/2 identiques,',summons,'invocations et',resolved,'conditions de type résolues',flush=True)
