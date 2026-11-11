"""Scénarios différentiels de groupes et de conditions dans le traducteur."""
from pathlib import Path
import ast
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'engine'), str(ROOT / 'translate'), str(ROOT / 'oracle')]
import compare
import core
import driver
import translate
from oracle_bank import build

SPELL, MONSTER, NORMAL = 600930091, 600930092, 600930093
for code, row in ((SPELL, (0, 0, 2, 0, 0, 0, 0, 0)),
                  (MONSTER, (0, 0, 33, 1000, 1000, 4, 1, 1)),
                  (NORMAL, (0, 0, 17, 1000, 1000, 4, 1, 1))):
    driver.TEST_CARDS[code] = (row, str(code))

CASES = ('relation_target_indestructible_oracle_reference.py',
         'group_after_failed_destroy_oracle_reference.py',
         'group_after_failed_summon_oracle_reference.py',
         'condition_before_action_oracle_reference.py',
         'failed_summon_condition_oracle_reference.py')

step, legal = jax.jit(core.step), jax.jit(core.legal)
for case in CASES:
    module = ast.parse((ROOT / 'tests' / case).read_text())
    scripts = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id in ('spell', 'monster')}
    with tempfile.TemporaryDirectory() as folder:
        for code, name in ((SPELL, 'spell'), (MONSTER, 'monster')):
            Path(folder, f'c{code}.lua').write_text(scripts[name])
        old = translate.OFFICIAL
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([SPELL, MONSTER, NORMAL])
            def chooser(dec, indices):
                for kind in ('activate', 'summon', 'to_ep', 'pass'):
                    choice = next((i for i in indices if dec.options[i][0] == kind and
                                   (kind not in ('activate', 'summon') or
                                    dec.options[i][1] in (SPELL, MONSTER))), None)
                    if choice is not None:
                        return choice
                return indices[0]
            result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                                  given=([[SPELL, MONSTER] * 20, [NORMAL] * 40], [[], []]))
            assert result['ok'], (case, result.get('err'))
            assert any(e['code'] == SPELL and e['stage'] == 'resolved' and e['count']
                       for e in result['coverage']), (case, result['coverage'])
            print(case, ':', result['decisions'], 'décisions identiques', flush=True)
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
