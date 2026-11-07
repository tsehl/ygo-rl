"""Pièges procéduraux : accepter/refuser l'effet initial et réutiliser l'effet ensuite."""
from pathlib import Path
import sys, tempfile
import jax
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'gen'), str(ROOT/'oracle')]
import core, cardgen, tolua, driver, compare

base = dict(ctype=17, atk=500, df=500, lv=4, attr=1, race=1, setc=[], lsc=0, rsc=0, effects=[], recipe=None)
activation = dict(kind='act', on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], cost=None, clauses=[], inline=2)
quick = dict(kind='quick', on=0, opt=core.OPT_CARD, optional=False, rng=core.Z_SZONE, dmg=0, conds=[],
             cost=('lp', 500), clauses=[dict(verb='damage', v=200)])
NORMAL, TRAP = 600910000, 600910001
specs = [dict(base, code=NORMAL), dict(base, code=TRAP, ctype=core.TYPE_TRAP|core.TYPE_CONTINUOUS,
                                      atk=0, df=0, lv=0, effects=[activation, quick])]
bank = cardgen.to_bank(specs); cmap = {c['code']: i for i, c in enumerate(specs)}
db = driver.CardDB(); db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs, folder); driver.SCRIPTS.insert(0, folder)
    for accept in (True, False):
        offered = later = 0
        for seed in (1400, 1401):
            def chooser(dec, indices):
                global offered, later
                if dec.msg == driver.K['MSG_SELECT_EFFECTYN'] and any(
                    dec.options[i][0] == 'chain' and dec.options[i][5] == 94 for i in indices):
                    offered += 1
                    return next(i for i in indices if dec.options[i][0] == ('chain' if accept else 'pass'))
                for kind in ('sset', 'activate', 'chain', 'summon', 'attack', 'to_bp', 'to_ep', 'pass'):
                    for i in indices:
                        o = dec.options[i]
                        if o[0] == kind:
                            if kind in ('activate', 'chain') and o[1] == TRAP and o[5] == (TRAP<<20)+1:
                                later += 1
                            return i
                return indices[0]
            result = compare.play(db, bank, cmap, seed, step, legal, mode='gen', chooser=chooser,
                                 given=([[TRAP, NORMAL]*20, [NORMAL]*40], [[], []]))
            assert result['ok'], (accept, seed, result.get('err'))
        assert offered > 0 and later > 0, (accept, offered, later)
        print(f'accepter={accept}: 2/2 identiques, {offered} questions, {later} activations ultérieures payées', flush=True)
