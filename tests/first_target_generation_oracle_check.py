"""Génération : première cible originale, relation et seconde clause conditionnelle."""
from pathlib import Path
import sys
import tempfile
import jax
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'gen', 'oracle')]
import cardgen, cards, compare, core, driver, tolua
SPELL, NORMAL, MST = 600930061, 600930062, 600930063
base = dict(ctype=core.TYPE_MONSTER | core.TYPE_NORMAL, atk=500, df=500, lv=4,
            attr=1, race=1, setc=[], lsc=0, rsc=0, effects=[], recipe=None)
def act(clauses):
    return dict(kind='act', on=0, opt=0, optional=False, rng=0, dmg=0,
                conds=[], cost=None, clauses=clauses)
probe = dict(verb='bounce', sel='target', f=dict(me=core.Z_SZONE, not_self=1),
             k=2, v=0, first_original=True)
reply = dict(verb='destroy', sel='target', f=dict(me=core.Z_SZONE, op=core.Z_SZONE), k=1, v=0)
specs = [dict(base, code=SPELL, ctype=core.TYPE_SPELL, atk=0, df=0, lv=0,
              effects=[act([probe, dict(verb='recover', v=1234)])]),
         dict(base, code=NORMAL),
         dict(base, code=MST, ctype=core.TYPE_SPELL | cards.TYPE_QUICKPLAY, atk=0, df=0, lv=0,
              effects=[act([reply])])]
bank = cardgen.to_bank(specs)
cmap = {c['code']: i for i, c in enumerate(specs)}
db = driver.CardDB(); db.add(tolua.cdb_rows(specs))
compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']] for c in specs})
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    tolua.write(specs, folder)
    driver.SCRIPTS.insert(0, folder)
    try:
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
            result = compare.play(db, bank, cmap, 3400, step, legal, mode='gen', chooser=chooser,
                                  given=([[SPELL, MST, MST, NORMAL] * 10, [MST, NORMAL] * 20], [[], []]))
            assert result['ok'], (hit, result.get('err'))
            assert state['fired'] and state['selected'], state
            assert state['replied'] == (hit is not None), state
            assert (9234 in observed_lp) == (hit != 1), (hit, observed_lp)
            for code in ((SPELL, MST) if hit is not None else (SPELL,)):
                assert any(e['code'] == code and e['stage'] == 'resolved' and e['count']
                           for e in result['coverage']), result['coverage']
            print('réponse', hit, ':', result['decisions'], 'décisions identiques', flush=True)
    finally:
        driver.SCRIPTS.remove(folder)
