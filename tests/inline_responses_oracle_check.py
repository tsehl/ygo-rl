"""Un effet intégré reste une activation de piège et dépend de sa source continue."""
import jax
from oracle_bank import build
import core, compare, driver

TRAP, EARTH, ELF, COUNTER, MST = 1802450, 69247929, 44287299, 3819470, 5318639
db, bank, cmap = build([TRAP, EARTH, ELF, COUNTER, MST])
step, legal = jax.jit(core.step), jax.jit(core.legal)
for reply in (COUNTER, MST):
    offers = replies = 0
    for seed in (1500, 1501):
        pending = False
        source = None
        def chooser(dec, indices):
            global pending, source, offers, replies
            if dec.msg == driver.K['MSG_SELECT_EFFECTYN']:
                candidates = [i for i in indices if dec.options[i][0] == 'chain' and dec.options[i][5] == 94]
                if candidates:
                    o = dec.options[candidates[0]]
                    source = (o[1], o[2], o[3], o[4])
                    pending = True; offers += 1
                    return candidates[0]
            if dec.options[0][0] == 'cards' and source is not None:
                for i in indices:
                    if source in dec.options[i][1]:
                        return i
            for kind in ('sset', 'activate', 'chain', 'summon', 'attack', 'to_bp', 'to_ep', 'pass'):
                for i in indices:
                    o = dec.options[i]
                    if o[0] != kind:
                        continue
                    if kind in ('activate', 'chain'):
                        if o[1] == TRAP and o[5] == 0 and int(compare.LAST_STATE.turn) < 4:
                            continue
                        if o[1] == reply:
                            if not pending:
                                continue
                            pending = False; replies += 1
                    return i
            return indices[0]
        result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                             given=([[TRAP, EARTH]*20, [reply, ELF, ELF, ELF]*10], [[], []]))
        assert result['ok'], (reply, seed, result.get('err'))
    assert offers > 0 and replies > 0, (reply, offers, replies)
    print(f'{reply}: 2/2 identiques, {offers} effets initiaux acceptés, {replies} réponses ciblées', flush=True)
