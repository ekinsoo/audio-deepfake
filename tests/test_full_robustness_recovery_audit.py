import copy
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import full_robustness_recovery_audit as audit


class RecoveryAuditTests(unittest.TestCase):
    def fixture(self):
        names = ['preflight', 'smoke', 'seed42_00000_00001', 'seed43_00000_00001', 'seed44_00000_00001']
        stages = [dict(name=name, status='complete', exit_code=0, elapsed_seconds=10,
                       budget_seconds=100, peak_rss_bytes=200*1024**2,
                       last_memory=dict(system_available_ram_bytes=4*1024**3, child_peak_rss_bytes=200*1024**2))
                  for name in names]
        failed = [dict(name='seed44_00000_02000', elapsed_seconds=90.844, status='stopped',
                       reason='child_exit_1', resource_failure='insufficient available system RAM')]
        resume = dict(original_job='outputs/jobs/original', original_status_sha256='old',
                      success_stages=stages[:-1], failed_stages=failed)
        execution = dict(source_sha256={'audit':'pin'})
        preflight = dict(input_sha256={'data':'hash'}, execution_source_sha256={'oldworker':'hash'})
        checked = dict(status='ok', audio_records_verified=50224, input_sha256=preflight['input_sha256'],
                       recovery_execution_sha256='hash', original_status_sha256='old', source_sha256=execution['source_sha256'])
        supervision = dict(status='scoring_complete', recovered=True, recovery_execution_sha256='hash',
                           recovery_preflight_sha256='hash', failure_history=failed,
                           original_job=resume['original_job'], original_status_sha256='old', stages=stages,
                           seed_smoke_seconds={'42':1,'43':1,'44':1},
                           seed_wall_seconds={'42':11,'43':11,'44':101.844},
                           source_sha256=preflight['execution_source_sha256'],
                           recovery_stages=[{**stages[0], 'name':'recovery_preflight'}])
        chunks = {str(seed):[dict(file=f'chunks/seed{seed}_00000_00001.json')] for seed in (42,43,44)}
        return resume, execution, preflight, checked, supervision, chunks

    def check(self, values):
        resume, execution, preflight, checked, supervision, chunks = values
        with patch.object(audit, 'validate_recovery_execution', return_value=(resume, execution)), \
             patch.object(audit, 'read', return_value=checked), patch.object(audit.full, 'digest', return_value='hash'):
            audit.recovery_supervision(supervision, preflight, chunks, {'42':1,'43':1,'44':1}, Path('output'), Path('original'))

    def test_recovery_keeps_failed_attempt_inside_budget(self):
        self.check(self.fixture())

    def test_failed_attempt_cannot_be_dropped_from_budget(self):
        values = self.fixture()
        values[4]['seed_wall_seconds']['44'] = 11
        with self.assertRaisesRegex(ValueError, 'failed-attempt budget'):
            self.check(values)

    def test_failure_history_and_original_success_must_survive(self):
        for alteration in ('failure', 'success', 'preflight'):
            values = self.fixture()
            if alteration == 'failure':
                values[4]['failure_history'] = []
            elif alteration == 'success':
                values[4]['stages'] = copy.deepcopy(values[4]['stages'])
                values[4]['stages'][0]['elapsed_seconds'] = 9
            else:
                values[3]['source_sha256'] = {}
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.check(values)

    def test_budget_nan_overflow_and_boolean_fail(self):
        for number in (float('nan'), 1800.1, True):
            values = self.fixture()
            values[4]['seed_wall_seconds']['44'] = number
            with self.subTest(number=number), self.assertRaises(ValueError):
                self.check(values)

    def test_resume_uses_saved_validator_when_finalizer_hook_installed(self):
        names = ['preflight','smoke'] + [f'seed{seed}_{start:05d}_{min(start+2000,24844):05d}'
                   for seed in (42,43) for start in range(0,24844,2000)]
        stages = [dict(name=name,elapsed_seconds=1) for name in names]
        stages.append(dict(name='seed44_00000_02000',status='stopped',exit_code=1,
                           reason='child_exit_1',elapsed_seconds=90,budget_seconds=1800))
        status = dict(status='stopped',exit_code=1,reason='RuntimeError: child_exit_1',
                      source_sha256={},stages=stages)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'job'
            original.mkdir()
            (original/'seed44_00000_02000.stderr.log').write_text('Resource stop: insufficient available system RAM')
            with patch.object(audit,'ROOT',root), \
                 patch.object(audit.worker,'frozen_inputs',return_value=(dict(execution_source_sha256={}),
                     dict(seeds={'42':dict(clean_scores_path='c'),'43':dict(clean_scores_path='d')}),{},dict(dev=[]),[],[])), \
                 patch.object(audit,'read',return_value=status), \
                 patch.object(audit,'smoke_checks',return_value={'42':1,'43':1,'44':1}), \
                 patch.object(audit,'validate_scores',return_value=[]), \
                 patch.object(audit.finalizer,'collect_chunks',return_value=([],[])), \
                 patch.object(audit.full,'digest',return_value='pin'), \
                 patch.object(audit,'ORIGINAL_SUPERVISION') as saved_validator, \
                 patch.object(audit.finalizer,'validate_supervision',side_effect=AssertionError('recursive hook')):
                result = audit.validate_resume(root,original)
                saved_validator.assert_called_once()
                self.assertEqual(result['seed_spent_seconds']['44'],91)

    def test_recovery_execution_pins_original_preflight_smoke_and_failure_history(self):
        resume = dict(original_job='original',original_status_sha256='status',chunk_sha256={},
                      failed_stages=[dict(reason='RAM')])
        execution = dict(status='frozen',source_sha256={name:'hash' for name in audit.RECOVERY_SOURCES},
                         original_job='original',original_status_sha256='status',chunk_sha256={},
                         preflight_sha256='hash',smoke_sha256='hash',failure_history=resume['failed_stages'])
        for key in (None,'preflight_sha256','smoke_sha256','failure_history'):
            changed = dict(execution)
            if key:
                changed[key] = 'changed'
            with self.subTest(key=key), patch.object(audit,'validate_resume',return_value=resume), \
                 patch.object(audit,'read',return_value=changed), patch.object(audit.full,'digest',return_value='hash'):
                if key:
                    with self.assertRaisesRegex(ValueError,'evidence changed'):
                        audit.validate_recovery_execution(Path('output'),Path('original'))
                else:
                    audit.validate_recovery_execution(Path('output'),Path('original'))


if __name__ == '__main__':
    unittest.main()
