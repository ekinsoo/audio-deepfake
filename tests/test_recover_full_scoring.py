"""Scoring recovery integrity boundaries; no data download or model fitting."""
import copy
import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audio_deepfake.baseline import GMMConfig
from audio_deepfake.features import LFCCConfig
from audio_deepfake.manifests import ManifestRow

SPEC = importlib.util.spec_from_file_location('recover_full_scoring', Path(__file__).resolve().parents[1]/'scripts/recover_full_scoring.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def rows(count=7):
    return [ManifestRow('dev', 'speaker_D', f'LA_D_{i}', 'A01', 'spoof', f'audio/{i}.flac') for i in range(count)]


def write_scores(path, selected):
    with path.open('w', encoding='utf-8', newline='') as target:
        writer=csv.writer(target, lineterminator='\n'); writer.writerow(recovery.FIELDS)
        for i,row in enumerate(selected):
            writer.writerow((row.utterance_id,row.split,row.label,row.attack_id,float(i)))


class ScoreRecoveryTests(unittest.TestCase):
    def test_score_metadata_order_count_and_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'scores.csv'; selected=rows(3)
            write_scores(path,selected)
            self.assertEqual(len(recovery.validate_scores(path,selected)),3)
            original=path.read_text()
            modifications=[original.replace('spoof','bonafide',1), original.replace('A01','A02',1),
                           original.replace('dev','eval',1), original.replace('LA_D_0','LA_D_1',1),
                           original.replace('0.0','nan',1), original.replace('0.0','inf',1),
                           original.replace('score','wrong',1), '\n'.join(original.splitlines()[:-1])+'\n',
                           original+original.splitlines()[1]+'\n']
            lines=original.splitlines(); modifications.append('\n'.join([lines[0],lines[2],lines[1],lines[3]])+'\n')
            for text in modifications:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError): recovery.validate_scores(path,selected)

    def test_unpickle_never_called_before_model_hash(self):
        with patch.object(recovery.full,'digest',return_value='bad'),patch.object(recovery,'load_model') as load:
            with self.assertRaisesRegex(ValueError,'refusing unpickle'):
                recovery.trusted_model({},LFCCConfig(),GMMConfig())
            load.assert_not_called()

    def model_fixture(self):
        lfcc,gmm=LFCCConfig(),GMMConfig()
        train=[SimpleNamespace(utterance_id='LA_T_1')]
        metadata=dict(format_version=1,lfcc_config=asdict(lfcc),gmm_config=asdict(gmm),training_ids=['LA_T_1'],
                      normalization='none',covariance_type='diag',
                      versions={name:recovery.version(name) for name in ('numpy','scipy','scikit-learn','joblib','soundfile')},
                      source_sha256={'baseline.py':recovery.MODEL_SHA256,'features.py':recovery.MODEL_SHA256})
        return {'train':train},lfcc,gmm,SimpleNamespace(lfcc_config=lfcc,gmm_config=gmm,metadata=metadata)

    def test_model_ids_configuration_versions_source(self):
        groups,lfcc,gmm,model=self.model_fixture()
        with patch.object(recovery.full,'digest',return_value=recovery.MODEL_SHA256),patch.object(recovery,'load_model',return_value=model):
            self.assertIs(recovery.trusted_model(groups,lfcc,gmm),model)
            for key,bad in [('training_ids',['LA_T_2']),('gmm_config',{}),('lfcc_config',{}),
                            ('versions',{}),('source_sha256',{}),('normalization','cmvn'),('format_version',2)]:
                saved=copy.deepcopy(model.metadata)
                model.metadata[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError): recovery.trusted_model(groups,lfcc,gmm)
                model.metadata=saved

    def original_fixture(self,root):
        original=root/'original'; original.mkdir()
        config={'experiment':{'seed':42}}
        prior_path=root/'prior.json'
        for name in recovery.CORE_SOURCES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        state=dict(status='failed',stage='dev-scoring',train_records_processed=25380,dev_records_processed=recovery.PREFIX_COUNT,
                   model_sha256=recovery.MODEL_SHA256,error_type='RuntimeError',error='Resource stop: process peak exceeds 2 GiB')
        (original/'run_state.json').write_text(json.dumps(state))
        (original/'config.json').write_text(json.dumps(config));(original/'model.joblib').write_bytes(b'checkpoint')
        (original/'scores.csv').write_text('prefix')
        prior=dict(status='stopped',seed=42,eval_enabled=False,reason='RuntimeError: child_peak_memory_budget_exceeded',
                   source_sha256={name:recovery.full.digest(root/name) for name in recovery.CORE_SOURCES})
        prior_path.write_text(json.dumps(prior))
        return original,prior_path,config,prior,state

    def test_original_state_config_source_and_model_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); original,prior_path,config,prior,state=self.original_fixture(root)
            resource_hash=recovery.full.digest(root/'src/audio_deepfake/resources.py')
            with patch.object(recovery,'ROOT',root),patch.object(recovery,'ORIGINAL',original),patch.object(recovery,'PRIOR_JOB',prior_path),\
                 patch.object(recovery.full,'validate_config'),patch.object(recovery,'MODEL_SHA256',recovery.full.digest(original/'model.joblib')),\
                 patch.object(recovery,'RESOURCE_ORIGINAL_SHA256',resource_hash),patch.object(recovery,'RESOURCE_FIXED_SHA256',resource_hash):
                # The state stores the configured artifact pin too.
                state['model_sha256']=recovery.MODEL_SHA256;(original/'run_state.json').write_text(json.dumps(state))
                self.assertIn('instrumentation_fix',recovery.validate_original(config))
                mutations=[('state','stage','training'),('state','train_records_processed',2000),('state','dev_records_processed',3),
                           ('prior','eval_enabled',True),('prior','reason','something else')]
                for group,key,value in mutations:
                    data=state if group=='state' else prior
                    path=original/'run_state.json' if group=='state' else prior_path
                    before=data[key];data[key]=value;path.write_text(json.dumps(data))
                    with self.subTest(group=group,key=key),self.assertRaises(ValueError):recovery.validate_original(config)
                    data[key]=before;path.write_text(json.dumps(data))
                (root/'src/audio_deepfake/features.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'source/config'):recovery.validate_original(config)
                (root/'src/audio_deepfake/features.py').write_text('src/audio_deepfake/features.py')
                (original/'config.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'configuration'):recovery.validate_original(config)
                (original/'config.json').write_text(json.dumps(config))
                (original/'model.joblib').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'refusing unpickle'):recovery.validate_original(config)

    def write_chunk(self,output,selected,start,stop):
        path=output/f'chunk-{start}-{stop}.csv';write_scores(path,selected[start:stop])
        record=dict(status='ok',start=start,stop=stop,count=stop-start,model_sha256=recovery.MODEL_SHA256,
                    score_sha256=recovery.full.digest(path),plan_sha256=recovery.full.digest(output/'plan.json'))
        path.with_suffix('.json').write_text(json.dumps(record))

    def test_collect_requires_complete_nonoverlapping_committed_chunks(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(recovery,'PREFIX_COUNT',3),patch.object(recovery,'TOTAL_DEV',7):
            output=Path(temp);(output/'plan.json').write_text('{}');selected=rows()
            with self.assertRaisesRegex(ValueError,'incomplete'):recovery.collect_chunks(output,selected)
            self.write_chunk(output,selected,3,5)
            with self.assertRaisesRegex(ValueError,'incomplete'):recovery.collect_chunks(output,selected)
            self.write_chunk(output,selected,5,7)
            pairs,records=recovery.collect_chunks(output,selected)
            self.assertEqual([pair[0] for pair in pairs],[row.utterance_id for row in selected[3:]])
            self.assertEqual(len(records),2)
            self.write_chunk(output,selected,4,6)
            with self.assertRaisesRegex(ValueError,'gap/overlap'):recovery.collect_chunks(output,selected)
            (output/'chunk-4-6.csv').unlink();(output/'chunk-4-6.json').unlink()
            (output/'chunk-5-7.csv').write_text('corruption')
            with self.assertRaisesRegex(ValueError,'hash differs'):recovery.collect_chunks(output,selected)

    def test_collect_rejects_orphan_wrongplan_and_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(recovery,'PREFIX_COUNT',3),patch.object(recovery,'TOTAL_DEV',7):
            output=Path(temp);(output/'plan.json').write_text('{}');selected=rows()
            self.write_chunk(output,selected,3,7)
            path=output/'chunk-3-7.json';record=json.loads(path.read_text());record['plan_sha256']='bad';path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Invalid chunk'):recovery.collect_chunks(output,selected)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'Uncommitted'):recovery.collect_chunks(output,selected)

    def test_invalid_chunk_ranges_refused_before_any_io(self):
        for start,stop in [(None,None),(0,10),(22815,24844),(22816,22815),(22815,24845),(True,2)]:
            with self.subTest(start=start,stop=stop),self.assertRaises(ValueError):recovery.chunk(Path('does-not-exist'),start,stop)

    def test_output_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            with self.assertRaisesRegex(ValueError,'already exists'):recovery.preflight(output)
            (output/'scores.csv').write_text('original')
            with self.assertRaisesRegex(ValueError,'already exists'):recovery.finalize(output)
            self.assertEqual((output/'scores.csv').read_text(),'original')

    def test_incomplete_finalize_never_evaluates_or_writes_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'plan.json').write_text('{}')
            with patch.object(recovery,'frozen_plan',return_value=({}, {}, {'dev':rows()})),\
                 patch.object(recovery.full,'evaluate_scores') as evaluate:
                with self.assertRaisesRegex(ValueError,'incomplete'):recovery.finalize(output)
                evaluate.assert_not_called();self.assertFalse((output/'scores.csv').exists())

    def test_overlap_reproduction_rejects_changed_scores(self):
        selected=rows(3);checked={'guard':SimpleNamespace(check=lambda:None),'groups':{'dev':selected},
                               'root':Path('.'),'lfcc':LFCCConfig()}
        with patch.object(recovery,'PREFIX_COUNT',3),patch.object(recovery,'load_audio'),patch.object(recovery,'extract_lfcc'),\
             patch.object(recovery,'score_features',return_value=1.0):
            prefix=[(row.utterance_id,1.0) for row in selected]
            self.assertEqual(len(recovery.validate_overlap(None,checked,prefix)),3)
            prefix[-1]=(selected[-1].utterance_id,1.01)
            with self.assertRaisesRegex(ValueError,'overlap score'):recovery.validate_overlap(None,checked,prefix)

    def test_chunk_scores_only_requested_suffix_and_commits_hashes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(recovery,'PREFIX_COUNT',3),patch.object(recovery,'TOTAL_DEV',7):
            output=Path(temp);(output/'plan.json').write_text('{}');selected=rows()
            journal={}
            for row in selected:
                path=output/row.audio_path;path.parent.mkdir(exist_ok=True);path.write_bytes(row.utterance_id.encode())
                journal[row.audio_path]={'size_bytes':path.stat().st_size,'sha256':recovery.full.digest(path)}
            journal_path=output/'journal.json';journal_path.write_text(json.dumps({'records':journal}))
            config={'paths':{'audio_root':str(output),'audio_journal':str(journal_path)}}
            guard=SimpleNamespace(check=lambda **kwargs:{'peak_rss_bytes':123},started=None)
            model=SimpleNamespace(metadata={'test':'unchanged'})
            with patch.object(recovery,'frozen_plan',return_value=({'model_metadata':model.metadata},config,{'dev':selected})),\
                 patch.object(recovery.full,'ResourceGuard',return_value=guard),\
                 patch.object(recovery.full,'validate_config',return_value=(LFCCConfig(),GMMConfig())),\
                 patch.object(recovery,'trusted_model',return_value=model),patch.object(recovery,'load_audio') as load,\
                 patch.object(recovery,'extract_lfcc'),patch.object(recovery,'score_features',return_value=2.5):
                result=recovery.chunk(output,3,5)
                self.assertEqual(result['count'],2);self.assertEqual(load.call_count,2)
                self.assertEqual([call.args[0] for call in load.call_args_list],[output/row.audio_path for row in selected[3:5]])
                self.assertEqual(recovery.validate_scores(output/'chunk-3-5.csv',selected[3:5]),[(row.utterance_id,2.5) for row in selected[3:5]])
                self.assertEqual(result['score_sha256'],recovery.full.digest(output/'chunk-3-5.csv'))
                self.assertTrue((output/'chunk-3-5.json').exists());self.assertFalse((output/'chunk-3-5.csv.tmp').exists())
                with self.assertRaisesRegex(ValueError,'already exists'):recovery.chunk(output,3,5)

    def test_finalize_combines_exact_scores_and_records_recovery_timing(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(recovery,'PREFIX_COUNT',3),patch.object(recovery,'TOTAL_DEV',7):
            output=Path(temp)/'recovery';output.mkdir();original=Path(temp)/'original';original.mkdir()
            (output/'plan.json').write_text('{}');selected=rows()
            write_scores(original/'scores.csv',selected[:3]);self.write_chunk(output,selected,3,7)
            record=json.loads((output/'chunk-3-7.json').read_text());record['recovery_seconds']=2.0
            (output/'chunk-3-7.json').write_text(json.dumps(record))
            plan=dict(input_sha256={'x':'hash'},model_metadata={},provenance={'source_sha256':{}},overlap_validation=[],preflight_seconds=1.0)
            config={'experiment':{'seed':42}};groups={'dev':selected}
            checked=dict(input_sha256=plan['input_sha256'],guard=SimpleNamespace(check=lambda:{},started=None),
                         asv=Path('asv'),reference=Path('reference'),archive={'verified':True})
            with patch.object(recovery,'ORIGINAL',original),patch.object(recovery,'frozen_plan',return_value=(plan,config,groups)),\
                 patch.object(recovery.full,'preflight',return_value=checked),patch.object(recovery,'environment_info',return_value={}),\
                 patch.object(recovery.full,'evaluate_scores',return_value={'eer':0.1}) as evaluate:
                summary=recovery.finalize(output)
                self.assertEqual(summary['status'],'ok');self.assertEqual(summary['recovery']['newly_scored_count'],4)
                self.assertEqual(summary['recovery']['chunk_seconds'],2.0)
                self.assertNotIn('training_seconds',summary['recovery'])
                self.assertEqual(len(recovery.validate_scores(output/'scores.csv',selected)),7)
                self.assertEqual(len(evaluate.call_args.args[1]),7)
                self.assertTrue((output/'summary.json').exists());self.assertFalse((output/'model.joblib').exists())


if __name__=='__main__':unittest.main()
