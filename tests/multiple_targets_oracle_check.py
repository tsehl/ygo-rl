"""Deux cibles : sélectionner la première originale avant de tester sa relation."""
from pathlib import Path
import ast
import shutil
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
import translate
from oracle_bank import build

SPELL, NORMAL, MST = 600930051, 44287299, 5318639
driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Multiple targets probe')
module = ast.parse((ROOT / 'tests/relation_multiple_targets_oracle_reference.py').read_text())
script = next(ast.literal_eval(node.value) for node in module.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == 'script')
# Require two available targets, as the selection itself does.
script = script.replace('LOCATION_SZONE,0,1,e:GetHandler()', 'LOCATION_SZONE,0,2,e:GetHandler()')
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    shutil.copy(Path(translate.OFFICIAL, f'c{MST}.lua'), folder)
    old = translate.OFFICIAL
    translate.OFFICIAL = folder
    driver.SCRIPTS.insert(0, folder)
    try:
        for getter in ('Duel.GetFirstTarget()', 'Duel.GetChainInfo(0,CHAININFO_TARGET_CARDS):GetFirst()'):
            Path(folder, f'c{SPELL}.lua').write_text(script.replace('Duel.GetFirstTarget()', getter))
            db, bank, cmap = build([SPELL, NORMAL, MST])
            for hit in (None, 0, 1):
                state = dict(fired=False, replied=False, selected=False)
                observed_lp = set()
                def chooser(dec, indices):
                    turn = int(compare.LAST_STATE.turn)
                    observed_lp.add(int(compare.LAST_STATE.lp[0]))
                    def find(kind, code):
                        return next((i for i in indices if dec.options[i][0] == kind
                                     and dec.options[i][1] == code), None)
                    choice = None
                    if (turn == 1 and dec.player == 0) or (turn == 2 and dec.player == 1):
                        choice = find('sset', MST)
                    if turn >= 3 and dec.player == 0 and not state['fired']:
                        choice = find('activate', SPELL)
                        if choice is not None:
                            state['fired'] = True
                    if state['fired'] and hit is not None and not state['replied'] and dec.player == 1:
                        choice = find('chain', MST)
                        if choice is not None:
                            state['replied'] = True
                    if choice is None and dec.options[0][0] == 'cards':
                        candidates = [i for i in indices if any(c[0] == MST and c[1] == 0
                                                              for c in dec.options[i][1])]
                        if candidates:
                            choice = candidates[min(hit, len(candidates)-1)] if dec.player == 1 else candidates[0]
                            if dec.player == 0 and state['fired']:
                                state['selected'] = True
                    if choice is None:
                        choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
                    return choice
                result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                                      given=([[SPELL, MST, MST, NORMAL] * 10, [MST, NORMAL] * 20], [[], []]))
                assert result['ok'], (getter, hit, result.get('err'))
                assert state['fired'] and state['selected'], state
                assert state['replied'] == (hit is not None), state
                assert (9234 in observed_lp) == (hit != 1), (hit, observed_lp)
                for code in ((SPELL, MST) if hit is not None else (SPELL,)):
                    assert any(e['code'] == code and e['stage'] == 'resolved' and e['count']
                               for e in result['coverage']), result['coverage']
                print(getter, 'réponse', hit, ':', result['decisions'], 'décisions identiques', flush=True)
    finally:
        translate.OFFICIAL = old
        driver.SCRIPTS.remove(folder)
