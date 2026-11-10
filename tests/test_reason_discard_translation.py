from pathlib import Path
import sys, tempfile, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T, core

class ReasonDiscardTranslationTests(unittest.TestCase):
    def test_discard_uses_explicit_lua_reason(self):
        path=Path(T.OFFICIAL,'c85034450.lua')
        original=path.read_text()
        for source,expected in ((original,core.REASON_COST|core.REASON_DISCARD),
                                (original.replace('REASON_COST|REASON_DISCARD','REASON_EFFECT'),core.REASON_EFFECT)):
            with tempfile.TemporaryDirectory() as folder:
                Path(folder,'c85034450.lua').write_text(source)
                with patch.object(T,'OFFICIAL',folder):effects,*_=T.translate(85034450)
            moves=[ins for effect in effects for ins in effect[4] if ins[0]==core.OP_MOVE]
            self.assertEqual(len(moves),1)
            self.assertEqual(moves[0][3],core.MOVE_REASON_EXPLICIT|(expected<<1))

if __name__=='__main__':unittest.main()
