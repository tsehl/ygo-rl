"""Activation d'un piège continu avec ou sans son effet optionnel initial."""
import jax
from oracle_bank import build
import core, compare, driver

TRAP, EARTH, ELF = 1802450, 69247929, 44287299
db, bank, cmap = build([TRAP, EARTH, ELF])
step, legal = jax.jit(core.step), jax.jit(core.legal)
for accept in (True, False):
    offered = chosen = later = 0
    for seed in (1300, 1301):
        def chooser(dec, indices):
            global offered, chosen, later
            if dec.msg == driver.K['MSG_SELECT_EFFECTYN'] and any(
                dec.options[i][0] == 'chain' and dec.options[i][5] == 94 for i in indices):
                offered += 1
                for i in indices:
                    if dec.options[i][0] == ('chain' if accept else 'pass'):
                        chosen += int(accept)
                        return i
            for kind in ('sset', 'activate', 'chain', 'summon', 'attack', 'to_bp', 'to_ep', 'pass'):
                for i in indices:
                    option = dec.options[i]
                    if option[0] == kind:
                        if kind in ('activate', 'chain') and option[1] == TRAP and option[5] == 0 and int(compare.LAST_STATE.turn) < 4:
                            continue
                        if kind in ('activate', 'chain') and option[1] == TRAP and option[5] != 0:
                            later += 1
                        return i
            return indices[0]
        result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                             given=([[TRAP, EARTH] * 20, [ELF] * 40], [[], []]))
        assert result['ok'], (accept, seed, result.get('err'))
    assert offered > 0, (accept, 'question jamais proposée')
    assert chosen > 0 if accept else later > 0, (accept, chosen, later)
    print(f'accepter={accept}: 2/2 identiques, {offered} questions, {chosen} choix oui, {later} effets ultérieurs', flush=True)
