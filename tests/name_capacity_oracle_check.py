"""Cinq noms distincts dans un tour : compteurs et interdictions doivent tous persister."""
from pathlib import Path
import sys,tempfile,collections
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'oracle')]
import core,cardgen,tolua,driver,compare

BASE=dict(ctype=17,atk=500,df=500,lv=4,attr=1,race=1,setc=[],lsc=0,rsc=0,effects=[],recipe=None)
EFFECT=dict(kind='ignition',on=0,opt=core.OPT_NAME,optional=False,rng=core.Z_MZONE,dmg=0,conds=[],cost=None,
            clauses=[dict(verb='recover',v=100)])
NORMAL=600930000
codes=list(range(600930001,600930006))
specs=[dict(BASE,code=NORMAL)]+[dict(BASE,code=c,ctype=33,effects=[EFFECT]) for c in codes]
bank=cardgen.to_bank(specs);cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']:[cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step,legal=jax.jit(core.step),jax.jit(core.legal)
maximum=0
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs,folder);driver.SCRIPTS.insert(0,folder)
    for seed in (1800,1801):
        board={0:set(),1:set()};used=collections.defaultdict(set)
        def chooser(dec,indices):
            for kind in ('activate','summon','to_ep','pass'):
                for i in indices:
                    o=dec.options[i]
                    if o[0]!=kind:continue
                    if kind=='summon':
                        if o[1] in board[dec.player]:continue
                        board[dec.player].add(o[1])
                    elif kind=='activate':used[int(compare.LAST_STATE.turn)].add(o[1])
                    return i
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='gen',chooser=chooser,
            given=([codes*8,[NORMAL]*40],[[],[]]))
        assert result['ok'],(seed,result.get('err'))
        maximum=max(maximum,max(map(len,used.values()),default=0))
assert maximum==5,maximum
print(f'2/2 identiques, {maximum} effets de noms distincts consommés dans le même tour',flush=True)
