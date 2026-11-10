"""La génération et sa version Lua exercent le premier de deux monstres ciblés."""
from pathlib import Path
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'gen'), str(ROOT / 'oracle')]
import cardgen
import compare
import core
import driver
import tolua

NORMAL, SPELL = 600991001, 600991002
base = dict(ctype=17, atk=1000, df=1000, lv=4, attr=1, race=1, setc=[], lsc=0, rsc=0,
            effects=[], recipe=None)
clause = dict(verb='destroy', sel='target', f=dict(op=core.Z_MZONE), k=2, v=0, first_only=True)
effect = dict(kind='act', on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], cost=None,
              clauses=[clause])
specs = [dict(base, code=NORMAL), dict(base, code=SPELL, ctype=core.TYPE_SPELL, atk=0, df=0,
                                      lv=0, effects=[effect])]
bank = cardgen.to_bank(specs)
cmap = {c['code']: i for i, c in enumerate(specs)}
db = driver.CardDB()
db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs, folder)
    driver.SCRIPTS.insert(0, folder)
    for seed in (3400, 3401):
        def chooser(dec, indices):
            for kind in ('activate', 'summon', 'to_ep', 'pass'):
                choices = [i for i in indices if dec.options[i][0] == kind]
                if choices:
                    return choices[0]
            return indices[0]
        result = compare.play(db, bank, cmap, seed, step, legal, mode='gen', chooser=chooser,
                              given=([[NORMAL, NORMAL, SPELL, NORMAL] * 10, [NORMAL] * 40], [[], []]))
        assert result['ok'], result.get('err')
        resolved = sum(e['count'] for e in result['coverage']
                       if e['code'] == SPELL and e['stage'] == 'resolved')
        assert resolved > 0, result['coverage']
        print(seed, 'partie identique, effets résolus', resolved, flush=True)
