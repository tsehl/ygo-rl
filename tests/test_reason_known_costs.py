from pathlib import Path
import sys, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T, core

class KnownCostReasonTests(unittest.TestCase):
    def test_helpers_keep_native_reason(self):
        cases={'Cost.SelfToGrave':core.REASON_COST,
               'Cost.SelfTribute':core.REASON_COST|core.REASON_RELEASE,
               'Cost.SelfDiscard':core.REASON_COST|core.REASON_DISCARD,
               'Cost.SelfDiscardToGrave':core.REASON_COST|core.REASON_DISCARD,
               'Cost.SelfBanish':core.REASON_COST,
               'Cost.DetachFromSelf':core.REASON_COST}
        for name,expected in cases.items():
            with self.subTest(name=name):
                tr=T.Translator.__new__(T.Translator);tr.nreg=0;tr.cur_fn=T.Fn(name=name,bound=[1])
                prog=T.Prog(tr,'cost');tr.run_known(name,prog)
                moves=[ins for ins in prog.ins if ins[0]==core.OP_MOVE]
                self.assertEqual(len(moves),1)
                self.assertEqual(moves[0][3],core.MOVE_REASON_EXPLICIT|(expected<<1))

if __name__=='__main__':unittest.main()
