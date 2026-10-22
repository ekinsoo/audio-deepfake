"""Full-split boundaries and checkpoint/failure behavior, without dataset training."""
import copy
import csv
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audio_deepfake.baseline import GMMConfig
from audio_deepfake.features import LFCCConfig
from audio_deepfake.manifests import ManifestRow

SPEC = importlib.util.spec_from_file_location('run_full_baseline', Path(__file__).resolve().parents[1]/'scripts/run_full_baseline.py')
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


def fixture(root):
    groups = {}
    hashes = {}
    for split, code, labels in [('train','T',['bonafide','spoof','spoof']),('dev','D',['bonafide','spoof'])]:
        path = root/full.PROTOCOL_FILES[split]
        path.write_text(''.join(f'speaker_{split} LA_{code}_{i} - {"-" if label=="bonafide" else "A01"} {label}\n'
                                for i,label in enumerate(labels)),encoding='utf-8')
        hashes[split]=full.digest(path)
        groups[split]=full.parse_protocol(path,split)
    selected=root/'full_train_dev.csv'
    text=io.StringIO(newline='')
    writer=csv.DictWriter(text,fieldnames=tuple(ManifestRow.__dataclass_fields__),lineterminator='\n')
    writer.writeheader();writer.writerows(asdict(r) for r in groups['train']+groups['dev'])
    selected.write_bytes(text.getvalue().encode())
    counts={k:{'count':len(v),'labels':full.summarize(v)['labels']} for k,v in groups.items()}
    return groups,hashes,selected,counts


class FullRunnerTests(unittest.TestCase):
    def config(self):
        return tomllib.loads((full.ROOT/'configs/full_seed42.toml').read_text())

    def test_fixed_configuration_and_allowed_seeds(self):
        for seed in (42,43,44):
            config=self.config();config['experiment']['seed']=seed
            lfcc,gmm=full.validate_config(config)
            self.assertEqual(lfcc,LFCCConfig());self.assertEqual(gmm,GMMConfig(seed=seed))
        for section,key,value in [('experiment','seed',45),('experiment','seed',True),
                                  ('data','eval_enabled',True),('data','train_count',2000),
                                  ('model','n_components',64),('features','pre_emphasis',0.9),
                                  ('resources','workers',2),('resources','max_processing_seconds',3600)]:
            config=self.config();config[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):
                full.validate_config(config)

    def test_exact_official_metadata_order_and_natural_balance(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp);groups,hashes,selection,counts=fixture(root)
            with patch.object(full,'PROTOCOL_SHA256',hashes),patch.object(full,'COUNTS',counts),patch.object(full,'SELECTION_SHA256',full.digest(selection)):
                actual,_=full.official_selection(root,selection)
                self.assertEqual(actual,groups)
                self.assertEqual(full.summarize(actual['train'])['labels'],{'bonafide':1,'spoof':2})
                lines=selection.read_text().splitlines()
                selection.write_text('\n'.join([lines[0],lines[2],lines[1],*lines[3:]])+'\n')
                with self.assertRaisesRegex(ValueError,'exactly preserve'):
                    full.official_selection(root,selection)

    def test_protocol_hash_or_natural_count_changes_rejected(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp);groups,hashes,selection,counts=fixture(root)
            with patch.object(full,'PROTOCOL_SHA256',hashes),patch.object(full,'COUNTS',counts),patch.object(full,'SELECTION_SHA256',full.digest(selection)):
                counts['train']['labels']={'bonafide':2,'spoof':1}
                with self.assertRaisesRegex(ValueError,'natural class counts'):
                    full.official_selection(root,selection)
                hashes['train']='0'*64
                with self.assertRaisesRegex(ValueError,'protocol SHA'):
                    full.official_selection(root,selection)

    def test_resource_gates(self):
        base=dict(system_available_ram_bytes=4*full.GIB,peak_rss_bytes=100_000_000)
        with patch.object(full,'memory_snapshot',return_value=base),patch.object(full.shutil,'disk_usage',return_value=SimpleNamespace(free=30*full.GIB)):
            guard=full.ResourceGuard(Path('.'));guard.check(initial=True)
            base['system_available_ram_bytes']=2*full.GIB
            with self.assertRaisesRegex(RuntimeError,'available system RAM'):guard.check(initial=True)
            guard.check()
            base['system_available_ram_bytes']=full.GIB-1
            with self.assertRaisesRegex(RuntimeError,'available system RAM'):guard.check()
            base['system_available_ram_bytes']=4*full.GIB;base['peak_rss_bytes']=2*full.GIB+1
            with self.assertRaisesRegex(RuntimeError,'process peak'):guard.check()
            base['peak_rss_bytes']=100_000_000;guard.started=time.perf_counter()-1801
            with self.assertRaisesRegex(RuntimeError,'30 minutes'):guard.check()
        with patch.object(full,'memory_snapshot',return_value=base),patch.object(full.shutil,'disk_usage',return_value=SimpleNamespace(free=19*full.GIB)):
            with self.assertRaisesRegex(RuntimeError,'free disk'):full.ResourceGuard(Path('.')).check()

    def test_existing_output_and_wrong_selection_pin_rejected(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'Output already exists'):full.preflight(self.config(),root)
            config=self.config();config['paths']['selection_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'SHA-256 pin'):full.preflight(config,root/'new')

    def test_preflight_audio_journal_reference_and_exact_members(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp);groups,hashes,selection,counts=fixture(root)
            config=self.config();config['paths'].update(audio_root=str(root/'audio'),protocol_dir=str(root),
                selection_manifest=str(selection),selection_sha256=full.digest(selection),audio_journal=str(root/'journal.json'),
                asv_scores=str(root/'asv'),reference_evaluator=str(root/'reference'))
            records={}
            for row in groups['train']+groups['dev']:
                path=root/'audio'/row.audio_path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic bytes')
                records[row.audio_path]={'sha256':full.digest(path),'size_bytes':path.stat().st_size,
                    'sample_rate':16000,'channels':1,'subtype':'PCM_16'}
            journal=dict(status='ok',expected_records=5,verified_records=5,records=records,source_protocol_sha256=hashes,
                source_manifest_sha256={'full_train.csv':full.canonical_hash(groups['train']),
                    'full_dev.csv':full.canonical_hash(groups['dev']),'full_train_dev.csv':full.digest(selection)},
                archive={'verified':True,'size_bytes':full.ARCHIVE_SIZE,'md5':full.ARCHIVE_MD5})
            (root/'asv').write_bytes(b'asv fixture');(root/'reference').write_bytes(b'reference fixture')
            with patch.object(full,'PROTOCOL_SHA256',hashes),patch.object(full,'COUNTS',counts),\
                 patch.object(full,'SELECTION_SHA256',full.digest(selection)),\
                 patch.object(full,'ASV_SHA256',full.digest(root/'asv')),\
                 patch.object(full,'REFERENCE_SHA256',full.digest(root/'reference')),\
                 patch.object(full.ResourceGuard,'check',return_value={'system_available_ram_bytes':4*full.GIB,'peak_rss_bytes':100}):
                (root/'journal.json').write_text(json.dumps(journal))
                self.assertEqual(full.preflight(config,root/'new')['groups'],groups)
                for target in ('incomplete','extra','protocol','manifest','archive','audio-hash','audio-format','reference'):
                    changed=copy.deepcopy(journal)
                    if target=='incomplete':changed['status']='running'
                    elif target=='extra':changed['records']['ASVspoof2019_LA_eval/flac/LA_E_1.flac']={}
                    elif target=='protocol':changed['source_protocol_sha256']={}
                    elif target=='manifest':changed['source_manifest_sha256']={}
                    elif target=='archive':changed['archive']['verified']=False
                    elif target=='audio-hash':next(iter(changed['records'].values()))['sha256']='0'*64
                    elif target=='audio-format':next(iter(changed['records'].values()))['channels']=2
                    elif target=='reference':(root/'reference').write_bytes(b'modified reference')
                    (root/'journal.json').write_text(json.dumps(changed))
                    with self.subTest(target=target),self.assertRaises(ValueError):full.preflight(config,root/'new')

    def test_preflight_failure_is_persisted_before_any_training(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            output=Path(temp)/'new'
            with patch.object(full,'preflight',side_effect=RuntimeError('Resource stop before fit')),\
                 patch.object(full,'train_feature_stream') as train:
                with self.assertRaisesRegex(RuntimeError,'Resource stop'):
                    full.run_experiment(full.ROOT/'configs/full_seed42.toml',output)
            train.assert_not_called()
            state=json.loads((output/'run_state.json').read_text())
            self.assertEqual(state['status'],'failed');self.assertEqual(state['stage'],'preflight')

    def prepared(self,root):
        groups,_,_,_=fixture(root)
        guard=SimpleNamespace(started=None,check=lambda **kwargs:{'peak_rss_bytes':100,'system_available_ram_bytes':4*full.GIB})
        return dict(groups=groups,root=root,guard=guard,initial_memory={},lfcc=LFCCConfig(),gmm=GMMConfig(),
                    input_sha256={},archive={'verified':True},asv=root/'asv',reference=root/'reference')

    def test_checkpoint_precedes_scoring_and_complete_score_ids(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp);plan=self.prepared(root);output=root/'run'
            def train(rows,loader,**kwargs):
                for row in rows: loader(row)
                return SimpleNamespace(metadata={'synthetic':True})
            def save(model,path):path.write_bytes(b'synthetic model checkpoint')
            def score(model,features):
                self.assertTrue((output/'model.joblib').exists())
                self.assertEqual(json.loads((output/'run_state.json').read_text())['stage'],'dev-scoring')
                return float(features[0,0])
            with patch.object(full,'preflight',return_value=plan),patch.object(full,'train_feature_stream',side_effect=train),\
                 patch.object(full,'load_audio',return_value=np.ones(16,dtype=np.float32)),\
                 patch.object(full,'extract_lfcc',return_value=np.ones((1,60),dtype=np.float32)),\
                 patch.object(full,'save_model',side_effect=save),patch.object(full,'score_features',side_effect=score),\
                 patch.object(full,'evaluate_scores',return_value={'eer':.2}) as evaluate,\
                 patch.object(full,'environment_info',return_value={}),\
                 patch.object(full.shutil,'disk_usage',return_value=SimpleNamespace(free=30*full.GIB)):
                result=full.run_experiment(full.ROOT/'configs/full_seed42.toml',output)
            self.assertEqual(result['scope'],'full-official-train-dev-adaptation')
            self.assertEqual(evaluate.call_args.kwargs['evaluation_scope'],'full')
            self.assertEqual(len(evaluate.call_args.args[1]),len(plan['groups']['dev']))
            self.assertEqual(json.loads((output/'run_state.json').read_text())['status'],'ok')
            self.assertTrue((output/'summary.json').exists())

    def test_scoring_failure_retains_model_and_failure_state(self):
        with tempfile.TemporaryDirectory(dir=full.ROOT) as temp:
            root=Path(temp);plan=self.prepared(root);output=root/'run'
            with patch.object(full,'preflight',return_value=plan),\
                 patch.object(full,'train_feature_stream',return_value=SimpleNamespace(metadata={})),\
                 patch.object(full,'save_model',side_effect=lambda model,path:path.write_bytes(b'checkpoint')),\
                 patch.object(full,'load_audio',side_effect=RuntimeError('Resource stop during scoring')):
                with self.assertRaisesRegex(RuntimeError,'Resource stop'):
                    full.run_experiment(full.ROOT/'configs/full_seed42.toml',output)
            self.assertTrue((output/'model.joblib').exists())
            state=json.loads((output/'run_state.json').read_text())
            self.assertEqual(state['status'],'failed');self.assertEqual(state['stage'],'dev-scoring')
            self.assertNotIn('summary.json',[p.name for p in output.iterdir()])


if __name__=='__main__':unittest.main()
