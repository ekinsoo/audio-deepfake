"""Offline tests for complete official train/dev preparation."""
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import soundfile as sf

SPEC = importlib.util.spec_from_file_location('prepare_full_data',Path(__file__).resolve().parents[1]/'scripts/prepare_full_data.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class FullPreparationTests(unittest.TestCase):
    def setUp(self):
        # These fixtures contain tiny synthetic audio; resource pressure has
        # its own explicit failure test below. Do not depend on host RAM or
        # the availability of a Windows system-memory probe on Linux.
        memory = patch.object(p, 'memory_snapshot', return_value={
            'system_available_ram_bytes': 4 * 1024**3})
        memory.start()
        self.addCleanup(memory.stop)

    def fixture(self,root):
        protocol_dir=root/'keys'
        protocol_dir.mkdir()
        specs={}
        for split,prefix,speaker in [('train','T','Tspeaker'),('dev','D','Dspeaker')]:
            path=protocol_dir/f'{split}.txt'
            lines=[]
            for index in range(8):
                label='bonafide' if index<2 else 'spoof'
                attack='-' if index<2 else f'A{index-1:02d}'
                lines.append(f'{speaker} LA_{prefix}_{index} - {attack} {label}\n')
            path.write_text(''.join(lines),encoding='utf-8')
            specs[split]={'filename':path.name,'sha256':p.digest(path),'size_bytes':path.stat().st_size,'count':8,'bonafide':2,'spoof':6,'speakers':1,'bonafide_speakers':1,'spoof_speakers':1,'spoof_per_attack':1}
        waveform=io.BytesIO()
        sf.write(waveform,np.zeros(160,dtype=np.float32),16000,format='FLAC',subtype='PCM_16')
        archive=root/'LA.zip'
        with zipfile.ZipFile(archive,'w') as source:
            for split,prefix in [('train','T'),('dev','D')]:
                for index in range(8):
                    source.writestr(f'LA/ASVspoof2019_LA_{split}/flac/LA_{prefix}_{index}.flac',waveform.getvalue())
            source.writestr('LA/ASVspoof2019_LA_dev/flac/LA_D_Aextra.flac',waveform.getvalue())
            source.writestr('LA/ASVspoof2019_LA_eval/flac/LA_E_1.flac',waveform.getvalue())
        return archive,protocol_dir,root/'audio',root/'output',specs

    def patches(self,fixture):
        rows = p.parse_protocol(fixture[1]/'train.txt','train') + p.parse_protocol(fixture[1]/'dev.txt','dev')
        combined_hash = hashlib.sha256(p.manifest_payload(rows)).hexdigest()
        return patch.multiple(p,SPECS=fixture[4],SIZE=fixture[0].stat().st_size,MD5=p.digest(fixture[0],'md5'),RESERVE_BYTES=0,MIN_RAM_BYTES=1,MIN_RUNNING_RAM_BYTES=1,SAVE_EVERY=3,COMBINED_MANIFEST_SHA256=combined_hash)

    def test_plan_no_writes_and_execute_exact_keys_without_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                plan=p.prepare_full(*fixture[:4])
                self.assertFalse(fixture[2].exists())
                self.assertFalse(fixture[3].exists())
                self.assertEqual(plan['expected_records'],16)
                self.assertEqual(plan['excluded_unkeyed_flac_counts'],{'dev':1})
                report=p.prepare_full(*fixture[:4],execute=True)
                self.assertEqual(report['status'],'ok')
                self.assertEqual(report['verified_records'],16)
                self.assertEqual(len(list(fixture[2].rglob('*.flac'))),16)
                self.assertFalse((fixture[2]/'ASVspoof2019_LA_eval').exists())
                combined=fixture[3]/'full_train_dev.csv'
                self.assertEqual(p.digest(combined),report['source_manifest_sha256']['full_train_dev.csv'])
                self.assertNotIn(b'\r',combined.read_bytes())
                self.assertEqual(report['source_protocol_sha256'],{key:spec['sha256'] for key,spec in fixture[4].items()})

    def test_changed_protocol_and_extra_rows_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                key=fixture[1]/'dev.txt'
                key.write_text(key.read_text()+'Dspeaker LA_D_999 - - bonafide\n')
                with self.assertRaisesRegex(ValueError,'protocol size/SHA'):
                    p.prepare_full(*fixture[:4])
                p.SPECS['dev']['sha256']=p.digest(key)
                p.SPECS['dev']['size_bytes']=key.stat().st_size
                with self.assertRaisesRegex(ValueError,'natural class counts'):
                    p.prepare_full(*fixture[:4])

    def test_bad_archive_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                fixture[0].write_bytes(fixture[0].read_bytes()+b'bad')
                with self.assertRaisesRegex(ValueError,'published size/MD5'):
                    p.prepare_full(*fixture[:4],execute=True)
                self.assertFalse(fixture[3].exists())

    def test_eval_and_noncanonical_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                splits,_=p.read_official_splits(fixture[1])
                row=splits['train'][0]
                for bad in [replace(row,split='eval'),replace(row,audio_path='../outside.flac')]:
                    with self.assertRaisesRegex(ValueError,'canonical train/dev'):
                        p.audio_path(fixture[2],bad)

    def test_interrupted_rename_gap_recovers_by_archive_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            original_replace=Path.replace
            interrupted=[]
            def interrupt(path,target):
                result=original_replace(path,target)
                if path.suffix=='.part' and not interrupted:
                    interrupted.append(True)
                    raise KeyboardInterrupt
                return result
            with self.patches(fixture):
                with patch.object(Path,'replace',interrupt),self.assertRaises(KeyboardInterrupt):
                    p.prepare_full(*fixture[:4],execute=True)
                report=json.loads((fixture[3]/'prepare_report.json').read_text())
                self.assertEqual(report['status'],'error')
                self.assertEqual(report['records'],{})
                self.assertEqual(len(list(fixture[2].rglob('*.flac'))),1)
                report=p.prepare_full(*fixture[:4],execute=True)
                self.assertEqual(report['status'],'ok')
                self.assertEqual(len(report['records']),16)

    def test_journal_tamper_refused_and_lock_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                p.prepare_full(*fixture[:4],execute=True)
                journal=fixture[3]/'prepare_report.json'
                report=json.loads(journal.read_text())
                first=next(iter(report['records']))
                report['records'][first]['sha256']='0'*64
                journal.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'Changed journaled audio'):
                    p.prepare_full(*fixture[:4],execute=True)
                self.assertFalse((fixture[3]/'prepare.lock').exists())
                self.assertEqual(json.loads(journal.read_text())['status'],'error')

    def test_unknown_existing_audio_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                splits,_=p.read_official_splits(fixture[1])
                row=splits['train'][0]
                path=p.audio_path(fixture[2],row)
                path.parent.mkdir(parents=True)
                with zipfile.ZipFile(fixture[0]) as source:
                    body=bytearray(source.read('LA/'+row.audio_path))
                body[-1]^=1
                path.write_bytes(body)
                with self.assertRaisesRegex(ValueError,'differs from verified archive'):
                    p.prepare_full(*fixture[:4],execute=True)

    def test_ram_gate_and_single_preparer_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                with patch.object(p,'memory_snapshot',return_value={'system_available_ram_bytes':0}),self.assertRaisesRegex(ValueError,'3 GiB'):
                    p.prepare_full(*fixture[:4])
                fixture[3].mkdir()
                (fixture[3]/'prepare.lock').write_text('active')
                with self.assertRaises(FileExistsError):
                    p.prepare_full(*fixture[:4],execute=True)
                self.assertEqual((fixture[3]/'prepare.lock').read_text(),'active')

    def test_pilot_journal_remains_unchanged_and_metadata_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture):
                p.prepare_full(*fixture[:4],execute=True)
                journal=fixture[3]/'prepare_report.json'
                report=json.loads(journal.read_text())
                first=next(iter(report['records']))
                pilot=Path(temporary)/'pilot.json'
                pilot.write_text(json.dumps({'records':{first:report['records'][first]}}))
                before=pilot.read_bytes()
                p.prepare_full(*fixture[:4],pilot_journal=pilot,execute=True)
                self.assertEqual(pilot.read_bytes(),before)
                known=json.loads(before)
                known['records'][first]['channels']=2
                pilot.write_text(json.dumps(known))
                with self.assertRaisesRegex(ValueError,'metadata'):
                    p.prepare_full(*fixture[:4],pilot_journal=pilot,execute=True)

    def test_pilot_journal_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture),self.assertRaisesRegex(ValueError,'overwrite the pilot journal'):
                p.prepare_full(*fixture[:4],pilot_journal=fixture[3]/'prepare_report.json',execute=True)
            self.assertFalse(fixture[3].exists())

    def test_report_temporary_symlink_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'prepare_report.json'
            real_is_symlink=Path.is_symlink
            def pretend_symlink(item):
                return item==root/'prepare_report.json.tmp' or real_is_symlink(item)
            with patch.object(Path,'is_symlink',pretend_symlink),self.assertRaisesRegex(ValueError,'symlinks'):
                p.save_full_report({'status':'running'},path)
            self.assertFalse(path.exists())

    def test_canonical_combined_manifest_pin_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=self.fixture(Path(temporary))
            with self.patches(fixture),patch.object(p,'COMBINED_MANIFEST_SHA256','0'*64),self.assertRaisesRegex(ValueError,'Canonical full manifest'):
                p.prepare_full(*fixture[:4])


if __name__=='__main__':
    unittest.main()
