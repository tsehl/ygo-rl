"""Tests du banc de validation : faux succès, protocole oracle et traçabilité."""
import collections
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'oracle')]
import compare
import numpy as np
import driver
import validation_run


class ValidationTests(unittest.TestCase):
    def test_compare_exit_status(self):
        for result, expected in ((True, 0), (False, 1)):
            with patch.object(sys, 'argv', ['compare.py', '2']), patch.object(driver, 'CardDB'), \
                 patch.object(compare, 'make_bank', return_value=(None, {})), \
                 patch.object(compare, 'play', return_value={'ok': result}):
                self.assertEqual(compare.main(), expected)

    def test_regression_propagates_failures_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / 'runner'
            runner.write_text('#!/bin/sh\necho "$1" >> "$CALLS"\n[ "$1" != "$FAIL_TARGET" ]\n')
            runner.chmod(0o755)
            import os
            for target, code in (('compare.py', 1), ('obs_check.py', 1), ('vmap_check.py', 1), ('sb_check.py', 1), ('none', 0)):
                calls = Path(tmp) / 'calls'
                calls.unlink(missing_ok=True)
                env = dict(os.environ, YGO_PYTHON=str(runner), JOBS='2', CALLS=str(calls), FAIL_TARGET=target)
                out = subprocess.run(['bash', str(ROOT / 'engine/regress.sh'), tmp], env=env, capture_output=True)
                self.assertEqual(out.returncode, code, out.stderr)
                self.assertEqual(len(calls.read_text().splitlines()), 21)

    def test_ambiguous_effect_does_not_use_jax_legality(self):
        state = SimpleNamespace(cid=np.array([0]), zone=np.array([compare.core.Z_MZONE]), ctrl=np.array([0]), seq=np.array([0]))
        opt = ('activate', 123, 0, 'mzone', 0, 9)
        with patch.dict(compare.DESC, {123: [9, 9]}):
            with self.assertRaises(compare.Inconclusive):
                compare.translate(None, opt, state, [True] * 100)

    def test_disjoint_effect_ranges_identify_the_effect_without_legality(self):
        class ForbiddenLegality:
            def __getitem__(self,key):raise AssertionError("légalité JAX consultée")
        state=SimpleNamespace(cid=np.array([0]),zone=np.array([compare.core.Z_HAND]),ctrl=np.array([0]),seq=np.array([0]))
        with patch.dict(compare.DESC,{123:[None,None]}),patch.dict(compare.EFFECT_KINDS,{123:[compare.core.K_IGNITION]*2}), \
             patch.dict(compare.EFFECT_RANGES,{123:[compare.core.Z_HAND|compare.core.Z_MZONE,compare.core.Z_GRAVE]}):
            self.assertEqual(compare.translate(None,('activate',123,0,'hand',0,0),state,ForbiddenLegality()),[compare.core.V_ACT0])
            state.zone[0]=compare.core.Z_GRAVE
            self.assertEqual(compare.translate(None,('activate',123,0,'grave',0,0),state,ForbiddenLegality()),[compare.core.V_ACT0+1])

    def test_overlapping_effect_ranges_remain_ambiguous(self):
        state=SimpleNamespace(cid=np.array([0]),zone=np.array([compare.core.Z_MZONE]),ctrl=np.array([0]),seq=np.array([0]))
        with patch.dict(compare.DESC,{123:[None,None]}),patch.dict(compare.EFFECT_KINDS,{123:[compare.core.K_IGNITION]*2}), \
             patch.dict(compare.EFFECT_RANGES,{123:[compare.core.Z_MZONE]*2}),self.assertRaises(compare.Inconclusive):
            compare.translate(None,('activate',123,0,'mzone',0,0),state)

    def test_oracle_effect_events(self):
        duel = driver.Duel.__new__(driver.Duel)
        duel.coverage = collections.Counter()
        duel.chains = {}
        duel.summoning = collections.defaultdict(list)
        def msg(name, fmt='', *args):
            duel._handle(bytes([driver.K[name]]) + struct.pack('<' + fmt, *args))
        for index in (1, 2):
            msg('MSG_CHAINING', 'IBBIIBBIQI', 42, 0, 4, 0, 1, 0, 4, 0, 77, index)
        msg('MSG_CHAIN_DISABLED', 'B', 1)
        msg('MSG_CHAIN_SOLVED', 'B', 2)
        msg('MSG_CHAIN_SOLVED', 'B', 1)
        self.assertEqual(duel.coverage[(42, 'activated', 77)], 2)
        self.assertEqual(duel.coverage[(42, 'resolved', 77)], 1)
        msg('MSG_CHAIN_END')
        self.assertFalse(duel.chains)
        # Invocation annulée, puis invocation réussie à la même place.
        msg('MSG_SUMMONING', 'IBBII', 10, 0, 4, 0, 1)
        msg('MSG_MOVE', 'IBBII', 10, 0, 4, 0, 1)
        msg('MSG_SUMMONING', 'IBBII', 11, 0, 4, 0, 1)
        msg('MSG_SUMMONED')
        self.assertEqual(duel.coverage[(10, 'summoned', None)], 0)
        self.assertEqual(duel.coverage[(11, 'summoned', None)], 1)

    def test_run_files_preserve_identity_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {'signature': 'abc', 'args': {'seed0': 12}}
            first = validation_run.save_run(tmp, manifest, [{'seed': 12}])
            second = validation_run.save_run(tmp, manifest, [])
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text())['games'], [{'seed': 12}])


    def test_signature_tracks_content_and_dependency_versions(self):
        args = SimpleNamespace(games=2, seed0=72)
        with patch.object(validation_run, 'fingerprint', return_value={'engine/core.py': 'old'}), \
             patch.object(validation_run, 'git_head', return_value='commit'):
            first = validation_run.provenance(args)
        with patch.object(validation_run, 'fingerprint', return_value={'engine/core.py': 'new'}), \
             patch.object(validation_run, 'git_head', return_value='commit'):
            second = validation_run.provenance(args)
        self.assertNotEqual(first['signature'], second['signature'])
        self.assertEqual(first['args'], {'games': 2, 'seed0': 72})

    def test_journal_keeps_completed_games_before_final_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = validation_run.RunJournal(tmp, {'signature': 'abc'})
            journal.append({'seed': 72, 'result': {'ok': False}})
            records = [json.loads(s) for s in journal.path.read_text().splitlines()]
            self.assertEqual(records[0]['manifest']['signature'], 'abc')
            self.assertFalse(records[1]['game']['result']['ok'])


    def test_yesno_trigger_mapping_uses_only_static_metadata(self):
        state = SimpleNamespace(cid=np.array([0]), zone=np.array([compare.core.Z_MZONE]), ctrl=np.array([0]), seq=np.array([0]))
        opt = ('chain', 123, 0, 'mzone', 0, 221)
        with patch.dict(compare.DESC, {123: [None, None]}), \
             patch.dict(compare.EFFECT_KINDS, {123: [compare.core.K_IGNITION, compare.core.K_TRIG_O]}):
            self.assertEqual(compare.translate(None, opt, state), [(compare.core.V_ACT0 + 1)])
        with patch.dict(compare.DESC, {123: [None, None]}), \
             patch.dict(compare.EFFECT_KINDS, {123: [compare.core.K_TRIG_O, compare.core.K_TRIG_O]}):
            with self.assertRaises(compare.Inconclusive):
                compare.translate(None, opt, state)


    def test_continuous_effect_is_not_an_activation_candidate(self):
        state = SimpleNamespace(cid=np.array([0]), zone=np.array([compare.core.Z_SZONE]), ctrl=np.array([0]), seq=np.array([0]))
        opt = ('activate', 123, 0, 'szone', 0, 0)
        with patch.dict(compare.DESC, {123: [None, None]}), \
             patch.dict(compare.EFFECT_KINDS, {123: [compare.core.K_ACTIVATE, compare.core.K_CONT]}):
            self.assertEqual(compare.translate(None, opt, state), [compare.core.V_ACT0])


    def test_failed_serialization_leaves_no_complete_run_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                validation_run.save_run(tmp, {'invalid': object()}, [])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_incomplete_translation_cache_is_rebuilt(self):
        import pickle
        from unittest.mock import MagicMock
        sys.path.insert(0, str(ROOT / 'translate'))
        import validate
        import translate
        expected = {24094653: ('ok', [], [], None, [])}
        context = MagicMock()
        context.Pool.return_value.__enter__.return_value.imap_unordered.return_value = list(expected.items())
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'translated.pkl'
            cache.write_bytes(b'\x80')  # pickle interrompu
            with patch.object(validate, 'CACHE', str(cache)), patch.object(validate, '_sig', return_value='signature'), \
                 patch.object(translate, 'all_codes', return_value=[24094653]), \
                 patch.object(validate.multiprocessing, 'get_context', return_value=context):
                self.assertEqual(validate.translate_all(), expected)
            with cache.open('rb') as f:
                self.assertEqual(pickle.load(f)['res'], expected)
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])


    def test_translation_cache_key_tracks_dependency_versions(self):
        sys.path.insert(0, str(ROOT / 'translate'))
        import validate
        with patch.object(validation_run, 'fingerprint', return_value={'translate/translate.py': 'unchanged'}):
            with patch.object(validation_run, 'package_versions', return_value={'luaparser': 'old'}):
                before = validate._sig()
            with patch.object(validation_run, 'package_versions', return_value={'luaparser': 'new'}):
                after = validate._sig()
        self.assertNotEqual(before, after)


    def test_identical_options_are_disambiguated_by_their_oracle_index(self):
        state = SimpleNamespace(cid=np.array([0]), zone=np.array([compare.core.Z_SZONE]), ctrl=np.array([0]), seq=np.array([0]))
        option=('chain',123,0,'szone',0,0)
        calls=[]
        def query(dec,index):
            calls.append(index)
            return driver.K['EVENT_FLIP_SUMMON_SUCCESS'] if index==0 else driver.K['EVENT_SUMMON_SUCCESS']
        dec=SimpleNamespace(options=[option,option],raw={'effect_query':query})
        with patch.dict(compare.DESC,{123:[None,None]}), \
             patch.dict(compare.EFFECT_KINDS,{123:[compare.core.K_ACTIVATE]*2}), \
             patch.dict(compare.EFFECT_EVENTS,{123:[1<<compare.core.EV_SUMMON,1<<compare.core.EV_FLIP_SUMMON]}):
            self.assertEqual(compare.translate(dec,option,state,option_index=0),[compare.core.V_ACT0+1])
            self.assertEqual(compare.translate(dec,option,state,option_index=1),[compare.core.V_ACT0])
        self.assertEqual(calls,[0,1])

    def test_description_94_alone_does_not_identify_an_inline_question(self):
        state=SimpleNamespace(cid=np.array([0]),zone=np.array([compare.core.Z_SZONE]),
            ctrl=np.array([0]),seq=np.array([0]),dec=compare.core.D_CHAIN,zctx=compare.core.ZC_ACT)
        option=('chain',123,0,'szone',0,94)
        dec=SimpleNamespace(msg=driver.K['MSG_SELECT_EFFECTYN'])
        with patch.dict(compare.DESC,{123:[94]}),patch.dict(compare.EFFECT_KINDS,{123:[compare.core.K_QUICK]}):
            self.assertEqual(compare.translate(dec,option,state),[compare.core.V_ACT0])

    def test_inline_question_verifies_the_pending_source(self):
        state=SimpleNamespace(cid=np.array([0,0]),zone=np.array([compare.core.Z_SZONE]*2),
            ctrl=np.array([0,0]),seq=np.array([0,1]),dec=compare.core.D_YESNO,zctx=compare.core.ZC_INLINE,ch_src=0)
        dec=SimpleNamespace(msg=driver.K['MSG_SELECT_EFFECTYN'])
        with patch.object(compare,'BANK_CODE',np.array([123])):
            self.assertEqual(compare.translate(dec,('chain',123,0,'szone',0,94),state),
                             [compare.core.special(2,compare.core.A_YES)])
            with self.assertRaises(compare.Mismatch):
                compare.translate(dec,('chain',123,0,'szone',1,94),state)


    def test_inspector_recovers_a_truncated_build_stamp(self):
        import build_inspector
        build_inspector.load.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(build_inspector,'OUT',Path(tmp)):
                (Path(tmp)/'effect-inspector.json').write_text('{')
                loaded=build_inspector.load()
                self.assertTrue(hasattr(loaded,'YGO_EffectEvent'))
                self.assertIn('signature',json.loads((Path(tmp)/'effect-inspector.json').read_text()))
                self.assertFalse((Path(tmp)/'effect-inspector.tmp').exists())
        finally:
            build_inspector.load.cache_clear()


if __name__ == '__main__':
    unittest.main()
