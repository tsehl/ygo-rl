"""Les nouvelles procédures restent accessibles au modèle via des cartes procédurales."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'engine'),str(ROOT/'gen'),str(ROOT/'train')]
import core,cardgen,tolua,model,brick_audit,compare_gen


class ProcedureGenerationTests(unittest.TestCase):
    def test_generated_comparator_reports_failures_to_the_shell(self):
        for ok,expected in ((True,0),(False,1)):
            with patch.object(sys,'argv',['compare_gen.py','--banks','0','--games','1']), \
                 patch.object(compare_gen.cardgen.Gen,'make_set',return_value=[dict(code=600000000,effects=[])]), \
                 patch.object(compare_gen.cardgen,'to_bank',return_value=None), \
                 patch.object(compare_gen.tolua,'write',return_value='/tmp/no-generated-scripts'), \
                 patch.object(compare_gen.tolua,'cdb_rows',return_value={}), \
                 patch.object(compare_gen.driver,'CardDB'), \
                 patch.object(compare_gen.driver,'SCRIPTS',[]), \
                 patch.object(compare_gen,'decks',return_value=None), \
                 patch.object(compare_gen.compare,'play',return_value=dict(ok=ok,err='controlled failure')), \
                 patch.object(compare_gen.compare,'COVER',{}), \
                 patch.object(compare_gen.compare,'LAST_LOG',[]):
                self.assertEqual(compare_gen.main(),expected)

    def test_generated_procedures_keep_fixed_shapes_and_visible_programs(self):
        shapes=None
        found=0
        for seed in (0,1):
            specs=cardgen.Gen(seed).make_set(80,10)
            bank=cardgen.to_bank(specs)
            current=tuple(x.shape for x in bank)
            if shapes is not None:self.assertEqual(current,shapes)
            shapes=current
            tok,mask=model.card_tokens(bank)
            for index,c in enumerate(specs):
                for effect,e in enumerate(c['effects']):
                    if e['kind']!='hand_proc':continue
                    found+=1
                    self.assertEqual(int(tok['ekind'][index,effect]),core.K_PROC)
                    self.assertTrue(bool(mask['eff'][index,effect]))
                    self.assertTrue(bool(mask['ins'][index,effect].any()))
                    self.assertTrue(cardgen.fits(e))
                    self.assertIn('genre PROC',brick_audit.features(bank,index))
                    table=tolua.spec_table(c,{x['code']:x['lv'] for x in specs})
                    self.assertEqual(table['effects'][effect]['kind'],'hand_proc')
                    # Bornes supérieures représentées dans le bytecode, pas perdues à l'assemblage.
                    if any(mx==0 for f,mn,mx in e['counts']):
                        instructions=np.asarray(bank.prog[index,effect,core.PG_COST])
                        self.assertTrue(((instructions[:,0]==core.OP_REQ)&(instructions[:,3]==1)).any())
        self.assertGreater(found,0)


if __name__=='__main__':unittest.main()
