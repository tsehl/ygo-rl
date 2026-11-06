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


if __name__ == '__main__':
    unittest.main()
