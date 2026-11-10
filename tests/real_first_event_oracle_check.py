"""Trois pièges officiels ciblent le premier monstre du groupe d'invocation."""
from pathlib import Path
import sys
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'engine'), str(ROOT / 'translate'), str(ROOT / 'oracle')]
import compare
import core
from oracle_bank import build

TRAPS = (4206964, 15083728, 42578427)
LARGE, SMALL = 15303296, 8715625
db, bank, cmap = build([*TRAPS, LARGE, SMALL])
step, legal = jax.jit(core.step), jax.jit(core.legal)
for trap in TRAPS:
    monster = SMALL if trap == 42578427 else LARGE
    for seed in (3400, 3401):
        def chooser(dec, indices):
            for kind in ('chain', 'sset', 'summon', 'to_ep', 'pass'):
                choices = [i for i in indices if dec.options[i][0] == kind]
                if choices:
                    return choices[0]
            return indices[0]
        result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                              given=([[trap, monster] * 20, [monster] * 40], [[], []]))
        assert result['ok'], (trap, seed, result.get('err'))
        resolved = sum(e['count'] for e in result['coverage'] if e['code'] == trap and e['stage'] == 'resolved')
        assert resolved > 0, (trap, seed, result['coverage'])
        print(trap, seed, 'identique, effets résolus', resolved, flush=True)
