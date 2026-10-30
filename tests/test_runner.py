"""Behavioral integration checks for the assembled single-file launch path."""
from pathlib import Path
from contextlib import closing
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.io as sio

SCRIPT=Path(__file__).resolve().parents[1]/'run_three_scenarios.py'
spec=importlib.util.spec_from_file_location('single_file_runner_under_test',SCRIPT)
runner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runner
spec.loader.exec_module(runner)


def sample_raw(one=False):
    times=np.array([0.]) if one else np.array([0.,.1])
    data={name:np.column_stack((times,np.zeros(len(times)))) for name in
          ('accelerator','brake','parking','selector','motor_left','motor_right')}
    if not one:
        data['accelerator'][-1,1]=.5
    return data


class RunnerTests(unittest.TestCase):
    def test_requested_scenario_settings_and_channel_minimality(self):
        expected=[(['pedal','parking','selector'],.1,10),
                  (['pedal','parking','selector','doors'],.01,5),
                  (['pedal','parking','selector','steering','doors'],.01,5)]
        self.assertEqual(len(runner.SCENARIOS),3)
        for cfg,(signals,dt,n) in zip(runner.SCENARIOS,expected):
            self.assertEqual(cfg['driver_signals'],signals)
            self.assertEqual((cfg['dt'],cfg['n'],cfg['pedal_step'],cfg['speed_step']),(dt,n,.5,.5))
        self.assertEqual(runner.SCENARIOS[2]['steering_step'],10.)
        # A valid first-scenario log need not contain the additional channels.
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'minimal.mat'
            raw=sample_raw()
            sio.savemat(file,{runner.MAT_SIGNALS[k]:v for k,v in raw.items()})
            loaded=runner.read_raw(file,runner.SCENARIOS[0],{})
            self.assertEqual(set(loaded),set(raw))
            with self.assertRaises(ValueError):
                runner.read_raw(file,runner.SCENARIOS[1],{})

    def test_log_basename_groups_never_cross_split_and_seed_is_stable(self):
        files=[Path('source1/a.mat'),Path('source2/a.mat'),Path('source1/b.mat'),
               Path('source2/c.mat'),Path('source2/d.mat')]
        train,test=runner.make_file_split(files,.5,2026)
        self.assertEqual(set(train)|set(test),set(files))
        self.assertFalse({p.stem for p in train}&{p.stem for p in test})
        reverse_train,reverse_test=runner.make_file_split(list(reversed(files)),.5,2026)
        self.assertEqual(set(train),set(reverse_train))
        self.assertEqual(set(test),set(reverse_test))
        with self.assertRaises(ValueError):
            runner.make_file_split([Path('source1/a.mat'),Path('source2/a.mat')],.2,2026)

    def test_failed_input_log_does_not_abort_later_logs_or_inflate_counts(self):
        cfg=runner.validate_scenario(dict(name='test',driver_signals=['pedal','parking','selector'],
                                          dt=.1,n=2,pedal_step=.5,speed_step=.5))
        settings={'max_gap':1.,'chunk_samples':1,'speed_factor':1.}
        def reader(path,*args):
            if str(path)=='bad.mat':
                raise OSError('deliberately unavailable input')
            return sample_raw()
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            with mock.patch.object(runner,'_export_training_progress'), \
                 mock.patch.object(runner,'evaluate_scenario',return_value={'test_events':1}):
                summary=runner.run_scenario(cfg,['a.mat','bad.mat','c.mat'],[],settings,folder,reader)
            self.assertEqual(summary['status'],'complete')
            with closing(sqlite3.connect(folder/'model.sqlite')) as db:
                self.assertEqual(db.execute('SELECT SUM(c) FROM edges').fetchone()[0],2)
                self.assertEqual(db.execute('SELECT c FROM retained_edges').fetchone()[0],2)
                self.assertEqual(dict(db.execute('SELECT path,status FROM logs')),
                                 {'a.mat':'ok','bad.mat':'error','c.mat':'ok'})

    def test_fitted_model_without_valid_test_is_explicitly_incomplete(self):
        cfg=runner.validate_scenario(dict(name='no_test',driver_signals=['pedal','parking','selector'],
                                          dt=.1,n=1,pedal_step=.5,speed_step=.5))
        settings={'max_gap':1.,'chunk_samples':1,'speed_factor':1.}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(runner,'_export_training_progress'), \
                 mock.patch.object(runner,'evaluate_scenario',return_value={'test_events':0}):
                summary=runner.run_scenario(cfg,['train.mat'],[],settings,Path(td),lambda *args:sample_raw())
            self.assertEqual(summary['status'],'trained_without_valid_test')
            with closing(sqlite3.connect(Path(td)/'model.sqlite')) as db:
                self.assertEqual(db.execute('SELECT SUM(c) FROM edges').fetchone()[0],1)

    def test_frozen_reader_rejects_mutation_before_loading(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'log.mat'; file.write_bytes(b'a')
            stat=file.stat()
            manifest=[dict(path=str(file),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)]
            loader=mock.Mock(return_value={'sentinel':'raw'})
            frozen=runner.freeze_input_reader(manifest,loader)
            self.assertEqual(frozen(file,{},{}),{'sentinel':'raw'})
            loader.reset_mock()
            file.write_bytes(b'changed length')
            with self.assertRaises(runner.InputChangedError):
                frozen(file,{},{})
            loader.assert_not_called()

    def test_frozen_reader_rejects_mutation_during_loading(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'log.mat'; file.write_bytes(b'a')
            stat=file.stat()
            manifest=[dict(path=str(file),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)]
            def mutate(*args):
                file.write_bytes(b'changed while reading')
                return {'not_used':'raw'}
            frozen=runner.freeze_input_reader(manifest,mutate)
            with self.assertRaises(runner.InputChangedError):
                frozen(file,{},{})

    def test_no_valid_training_transition_cannot_be_reported_complete(self):
        cfg=runner.validate_scenario(dict(name='empty',driver_signals=['pedal','parking','selector'],
                                          dt=.1,n=2,pedal_step=.5,speed_step=.5))
        settings={'max_gap':1.,'chunk_samples':1,'speed_factor':1.}
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            with mock.patch.object(runner,'_export_training_progress'), \
                 mock.patch.object(runner,'evaluate_scenario') as evaluate:
                with self.assertRaisesRegex(ValueError,'No valid training transitions'):
                    runner.run_scenario(cfg,['single.mat'],[],settings,folder,
                                        lambda *args:sample_raw(one=True))
                evaluate.assert_not_called()
            with closing(sqlite3.connect(folder/'model.sqlite')) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM edges').fetchone()[0],0)
                self.assertEqual(db.execute('SELECT status FROM logs').fetchone()[0],'no_valid_transitions')


if __name__=='__main__':
    unittest.main()
