"""Rejoue une partie de compare.py en mode verbeux : python replay.py GRAINE [MODE]."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import jax, compare, core, cards, driver
db = driver.CardDB()
bank, cmap = compare.make_bank(db, db.normal_monsters() + list(cards.CARDS))
r = compare.play(db, bank, cmap, int(sys.argv[1]), jax.jit(core.step), jax.jit(core.legal), verbose=True, mode=sys.argv[2] if len(sys.argv) > 2 else "mixed")
print(r)
