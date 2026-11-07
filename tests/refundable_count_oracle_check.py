"""Le cœur rend les compteurs OATH après une activation niée, mais pas les autres."""
from pathlib import Path
import collections, sys, tempfile
import jax
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'gen'), str(ROOT/'oracle')]
import core, cardgen, tolua, driver, compare

NORMAL, COUNTER, USED, ACTIVATED = range(600920000, 600920004)
base = dict(ctype=17, atk=500, df=500, lv=4, attr=1, race=1, setc=[], lsc=0, rsc=0, effects=[], recipe=None)
effect = dict(kind='act', on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], cost=None, clauses=[])
specs = [dict(base, code=NORMAL), dict(base, code=COUNTER, ctype=core.TYPE_TRAP|core.TYPE_COUNTER,
    atk=0, df=0, lv=0, effects=[dict(effect, conds=[('chain', core.TYPE_SPELL)], cost=('lp',500), clauses=[dict(verb='negate')])])]
for code, refund in ((USED, False), (ACTIVATED, True)):
    specs.append(dict(base, code=code, ctype=core.TYPE_SPELL, atk=0, df=0, lv=0,
        effects=[dict(effect, opt=core.OPT_NAME, refund=refund, clauses=[dict(verb='damage', v=200)])]))
bank = cardgen.to_bank(specs); cmap = {c['code']: i for i, c in enumerate(specs)}
db = driver.CardDB(); db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs, folder); driver.SCRIPTS.insert(0, folder)
    for code in (USED, ACTIVATED):
        maximum = replies = 0
        for seed in (1600,1601):
            casts = collections.Counter(); pending = False
            def chooser(dec, indices):
                global replies, pending
                for kind in ('sset', 'activate', 'chain', 'summon', 'attack', 'to_bp', 'to_ep', 'pass'):
                    for i in indices:
                        o = dec.options[i]
                        if o[0] != kind:
                            continue
                        if kind == 'sset' and o[1] != COUNTER:
                            continue
                        if kind in ('activate', 'chain'):
                            if o[1] == code:
                                turn = int(compare.LAST_STATE.turn)
                                if turn < 4:
                                    continue
                                pending = True; casts[turn] += 1
                            elif o[1] == COUNTER:
                                if not pending:
                                    continue
                                pending = False; replies += 1
                        return i
                return indices[0]
            result = compare.play(db, bank, cmap, seed, step, legal, mode='gen', chooser=chooser,
                given=([[code,NORMAL]*20, [COUNTER,NORMAL]*20], [[],[]]))
            assert result['ok'], (code, seed, result.get('err'))
            maximum = max(maximum, max(casts.values(), default=0))
        assert replies > 0, (code, 'aucune activation niée')
        assert maximum > 1 if code == ACTIVATED else maximum == 1, (code, maximum)
        print(f'{code}: 2/2 identiques, {replies} réponses, maximum {maximum} activations par tour', flush=True)
