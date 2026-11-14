"""Zone libérée par le coût (ZX_FREE / ZX_FREEANY), « à moi ou face recto » (ZX_OPFU) et étiquettes constantes."""
import sys
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine'), str(ROOT/'translate')]
import core, cards as C

ALLZ = core.Z_DECK | core.Z_HAND | core.Z_MZONE | core.Z_SZONE | core.Z_GRAVE | core.Z_REMOVED | core.Z_EXTRA
NORMAL = core.TYPE_MONSTER | core.TYPE_NORMAL


class ZoneFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        codes = [1, 2]
        cls.bank = core.make_bank(np.array(codes), [0, 0], [0, 0], [4, 4], ctype=[NORMAL, NORMAL], **C.assemble(codes))
        d = jnp.ones(40, jnp.int32)
        s = jax.jit(core.new_game)(cls.bank, d, d)
        cls.base = s._replace(zone=jnp.full(80, core.Z_DECK, jnp.int32), ctrl=jnp.arange(80, dtype=jnp.int32) // 40)
        cls.mask = staticmethod(jax.jit(core.filter_mask, static_argnames=("bdes",)))

    def field(self, mine, theirs, hand=()):
        """mine / theirs : listes de (carte, case, position) en zone monstres."""
        s = self.base
        zone, seq, pos = s.zone, s.seq, s.pos
        for c, q, ps in list(mine) + list(theirs):
            zone, seq, pos = zone.at[c].set(core.Z_MZONE), seq.at[c].set(q), pos.at[c].set(ps)
        for c, q in hand:
            zone, seq = zone.at[c].set(core.Z_HAND), seq.at[c].set(q)
        return s._replace(zone=zone, seq=seq, pos=pos)

    def sel(self, s, zx, me=ALLZ, op=ALLZ):
        f = jnp.array(C.F(me=me, op=op, zx=zx), jnp.int32)
        return list(np.nonzero(np.asarray(self.mask(self.bank, s, f, 0, 0, bdes=False)))[0])

    def test_free_zone(self):
        fu = core.P_FU_ATK
        full = [(i, i, fu) for i in range(5)] + [(5, 5, fu)]  # cinq cases principales et une case extra
        s = self.field(full, [(40, 1, fu)], hand=[(6, 0)])
        # aucune case libre : seuls mes monstres des cases principales libèrent une case
        self.assertEqual(self.sel(s, core.ZX_FREE, me=core.Z_MZONE | core.Z_HAND, op=core.Z_MZONE), [0, 1, 2, 3, 4])
        # sans condition de contrôleur : le monstre adverse de la case 1 compte aussi
        self.assertEqual(self.sel(s, core.ZX_FREEANY, me=core.Z_MZONE, op=core.Z_MZONE), [0, 1, 2, 3, 4, 40])
        # une case libre : tout passe
        s2 = self.field(full[1:], [(40, 1, fu)], hand=[(6, 0)])
        self.assertEqual(self.sel(s2, core.ZX_FREE, me=core.Z_MZONE | core.Z_HAND, op=core.Z_MZONE), [1, 2, 3, 4, 5, 6, 40])

    def test_mine_or_faceup(self):
        s = self.field([(0, 0, core.P_FD_DEF)], [(40, 0, core.P_FU_ATK), (41, 1, core.P_FD_DEF)])
        self.assertEqual(self.sel(s, core.ZX_OPFU, me=core.Z_MZONE, op=core.Z_MZONE), [0, 40])


class LabelTranslationTests(unittest.TestCase):
    def test_constant_label_folded(self):
        import translate as T
        # Evolution Burst-like : coût e:SetLabel(1), cible chk==0 « if e:GetLabel()==1 then e:SetLabel(0) … else … end »
        effs, *_ = T.translate(5338223)
        flat = [ins for e in effs for prog in e[4:7] for ins in prog]
        zx = [ins[2][core.F_ZX] for ins in flat if ins[0] == core.OP_FILTER and isinstance(ins[2], tuple)]
        self.assertTrue(any(z & core.ZX_FREE and z & core.ZX_OPFU for z in zx))

    def test_dynamic_label_refused(self):
        import translate as T
        with self.assertRaises(T.Unsupported):
            T.translate(483)


if __name__ == "__main__":
    unittest.main()
