"""Régression : prédicat sur SELF après destruction au combat, avant l'envoi au cimetière."""
import json
import sys
import tempfile
from pathlib import Path
import jax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'oracle'),str(ROOT/'gen')]
import core,cardgen,tolua,compare,driver

specs=json.loads((Path(__file__).parent/'fixtures/damage_flip_self.json').read_text())
bank=cardgen.to_bank(specs)
cmap={c['code']:i for i,c in enumerate(specs)}
db=driver.CardDB();db.add(tolua.cdb_rows(specs))
step,legal=jax.jit(core.step),jax.jit(core.legal)
with tempfile.TemporaryDirectory() as tmp:
    tolua.write(specs,tmp);driver.SCRIPTS.insert(0,tmp)
    total=0
    for seed in (400,401):
        def chooser(dec,indices):
            for kind in ('mset' if dec.player==0 else 'summon','chain','attack','to_bp'):
                for i in indices:
                    if dec.options[i][0]==kind:return i
            return indices[0]
        r=compare.play(db,bank,cmap,seed,step,legal,mode='gen',
                       given=([[specs[0]['code']]*40,[specs[1]['code']]*40],[[],[]]),chooser=chooser)
        assert r['ok'],(seed,r.get('err'))
        count=sum(e['count'] for e in r['coverage'] if e['code']==specs[0]['code'] and e['stage']=='activated')
        assert count>0,'effet FLIP non exercé'
        total+=count
    print(f'FLIP sur soi-même après destruction au combat : 2/2 identiques, {total} activations',flush=True)
