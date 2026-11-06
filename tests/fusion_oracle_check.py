"""Scénarios de fusion ciblés, suivis jusqu'à la fin contre l'oracle.
Usage : .venv/bin/python tests/fusion_oracle_check.py
Les cartes fictives servent uniquement à isoler les règles ; jamais à l'entraînement.
"""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'translate'), str(ROOT/'engine'), str(ROOT/'oracle')]
import translate as T
import cards
import core
import compare
import driver


def main():
    fixtures = Path(__file__).parent/'fixtures/fusion'
    ids = (990000004, 990000005, 990000006)
    for code in ids:
        driver.TEST_CARDS[code] = ((0, 0, core.TYPE_MONSTER | core.TYPE_FUSION, 3000, 2500, 8, 1, 1), f'Test Fusion {code}')
    driver.SCRIPTS.insert(0, str(fixtures))
    recipes = {}
    with patch.object(T, 'OFFICIAL', str(fixtures)):
        for code in ids:
            effects, _, proc, _ = T.translate(code)
            assert effects == []
            assert proc[0] == 'fusion'
            recipes[code] = cards.FusionRecipe(proc[1], proc[2])
    for code in (3544583, 96334243, 8692301):
        effects, _, proc, _ = T.translate(code)
        assert effects == []
        recipes[code] = cards.FusionRecipe(proc[1], proc[2])
    db = driver.CardDB()
    codes = sorted([*ids, 3544583, 96334243, 8692301, 487395, 76908448,
                    34460851, 44287299, 89631139, 46986414, 24094653])
    rows = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [r[4] for r in rows], [r[5] for r in rows], [r[6] & 255 for r in rows],
                          ctype=[r[3] for r in rows], race=[r[7] for r in rows], attr=[r[8] for r in rows],
                          setcode=[cards.setcodes(r[2]) for r in rows],
                          **cards.assemble(codes, cards={24094653: T.translate(24094653)[0]}, fusion=recipes,
                                           ritual={}, synchro={}, xyz={}, link={}))
    cmap = {c:i for i,c in enumerate(codes)}
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    cases = [('distincts', ids[0], [34460851, 44287299] * 17 + [24094653]*6, True),
             ('répétés', ids[1], [89631139]*34 + [24094653]*6, True),
             ('filtres qui se recouvrent', ids[2], [44287299]*34 + [24094653]*6, True),
             ('matériau nommé absent', ids[0], [34460851]*34 + [24094653]*6, False),
             ('un seul exemplaire disponible', ids[1], [46986414]*33 + [89631139] + [24094653]*6, False),
             ('Allvain (script officiel)', 3544583, [44287299]*34 + [24094653]*6, True),
             ('Theseus (script officiel)', 96334243, [487395]*34 + [24094653]*6, True),
             ('Zirconia (script officiel)', 8692301, [76908448]*34 + [24094653]*6, True)]
    for name, fusion, deck, expected in cases:
        summoned = activated = 0
        for seed in (72, 73):
            result = compare.play(db, bank, cmap, seed, step, legal, given=([deck, deck], [[fusion], [fusion]]))
            assert result['ok'], (name, result.get('err'))
            summoned += sum(e['count'] for e in result['coverage'] if e['code'] == fusion and e['stage'] == 'summoned')
            activated += sum(e['count'] for e in result['coverage'] if e['code'] == 24094653 and e['stage'] == 'resolved')
        assert (summoned > 0) == expected, (name, summoned)
        assert (activated > 0) == expected, (name, activated)
        print(f'{name} : 2/2 identiques ; {summoned} fusions invoquées, {activated} Polymerization résolues', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
