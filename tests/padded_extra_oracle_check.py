"""Capacité fixe : mêmes décisions natives avec des Extra Decks asymétriques ou vides."""
import jax
from oracle_bank import build
import core,compare
POLY,FLAME,A,B=24094653,45231177,34460851,44287299
db,bank,cmap=build([POLY,FLAME,A,B])
step,legal=jax.jit(core.step),jax.jit(core.legal)
uses=0
for extras in (([],[]),([FLAME],[]),([FLAME]*3,[FLAME])):
    seed=800
    def chooser(dec,indices):
        for kind in ('activate','spsummon','summon','attack','to_bp'):
            for i in indices:
                if dec.options[i][0]==kind:return i
        return indices[0]
    r=compare.play(db,bank,cmap,seed,step,legal,mode='tr',
                   given=([[POLY,A,B,A,B]*8,[B]*40],extras),chooser=chooser,extra_capacity=15)
    assert r['ok'],(extras,r.get('err'))
    uses+=sum(e['count'] for e in r['coverage'] if e['code']==FLAME and e['stage']=='summoned')
    print(f'Extra Decks {list(map(len,extras))}: identique, N fixe=110',flush=True)
assert uses>0,uses
print(f'{uses} fusions effectivement invoquées, capacité inactive vérifiée',flush=True)
