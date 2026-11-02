from pathlib import Path
from types import SimpleNamespace
import csv
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_fourth_scenario as s4
import numpy as np


def config(dt=.1,n=0):
    return s4.validate_scenario(dict(s4.SCENARIOS[0],dt=dt,n=n))


def raw(pedals=(.1,.11,.11,.1,.1),steering=(0.,1.,1.,0.,-1.),times=None):
    times=np.arange(len(pedals))*.1 if times is None else np.asarray(times)
    values={name:np.column_stack((times,np.zeros(len(times)))) for name in
        ('accelerator','brake','parking','selector','steering','door1','door2','door3','motor_left','motor_right')}
    values['accelerator'][:,1]=pedals
    values['steering'][:,1]=np.deg2rad(steering)
    return values


def blocks(data,cfg=None,chunk=200000,**settings):
    return list(s4.iter_aligned_blocks(data,cfg or config(),dict(max_gap=1.,
        chunk_samples=chunk,steering_limit_degrees=1260.) | settings))


def flatten(result,key):
    return np.concatenate([b[key][b['new_sample_start']:] for b in result])


class DirectionsTests(unittest.TestCase):
    def test_only_fourth_scenario_and_requested_settings(self):
        self.assertEqual(len(s4.SCENARIOS),1)
        cfg=s4.SCENARIOS[0]
        self.assertTrue(cfg['name'].startswith('04_'))
        self.assertEqual((cfg['dt'],cfg['n'],cfg['pedal_step'],cfg['speed_step'],cfg['steering_step']),(.01,5,.5,.5,10.))
        self.assertEqual(cfg['driver_signals'],['pedal','parking','selector','steering','doors','pedal_direction','steering_direction'])
        self.assertNotIn('pedal_direction',s4.required_sources(cfg))
        self.assertNotIn('steering_direction',s4.required_sources(cfg))
        self.assertEqual(set(s4.required_sources(cfg)),set(raw()))

    def test_derivatives_require_their_physical_control(self):
        for feature in ('pedal_direction','steering_direction'):
            with self.assertRaisesRegex(ValueError,'requires'):
                s4.validate_scenario(dict(config(),driver_signals=[feature]))

    def test_signed_changes_before_quantization_include_zero(self):
        b=blocks(raw())[0]
        np.testing.assert_array_equal(b['state'][:,0],np.zeros(5))
        np.testing.assert_array_equal(b['state'][:,3],np.zeros(5))
        np.testing.assert_array_equal(b['state'][:,5],[0,1,0,-1,0])
        np.testing.assert_array_equal(b['state'][:,6],[0,1,0,-1,-1])
        np.testing.assert_array_equal(b['valid_samples'],[False,True,True,True,True])
        np.testing.assert_array_equal(b['transition_valid'],[False,True,True,True])
        np.testing.assert_array_equal(b['state_values'][:,5:],b['state'][:,5:])

    def test_pedal_direction_is_accelerator_minus_brake(self):
        data=raw(pedals=(10.,)*5)
        data['brake'][:,1]=[0,1,1,0,0]
        np.testing.assert_array_equal(blocks(data)[0]['state'][:,5],[0,-1,0,1,0])

    def test_direction_uses_only_raw_measurements_available_now(self):
        data=raw(pedals=(0.,)*7,steering=(0.,)*7)
        data['steering']=np.column_stack(([0.,.2,.4,.6],np.deg2rad([0.,20.,40.,60.])))
        a=blocks(data)[0]
        changed={k:v.copy() for k,v in data.items()}
        changed['steering'][1:,1]=np.deg2rad([-20.,-40.,-60.])
        b=blocks(changed)[0]
        self.assertEqual(a['state'][1,6],0)
        self.assertEqual(b['state'][1,6],0)
        # S3 offline angle interpolation is retained, separately from causal signs.
        self.assertNotEqual(a['state'][1,3],b['state'][1,3])
        self.assertEqual(a['state'][2,6],1)
        self.assertEqual(b['state'][2,6],-1)

    def test_zero_tolerance_is_only_floating_point_roundoff(self):
        values=np.array([1.,np.nextafter(1.,np.inf),1.+1e-8,1.+1e-8])
        np.testing.assert_array_equal(s4._s4_backward_sign(values),[0,0,1,0])

    def test_chunk_overlap_does_not_reset_or_double_count_derivatives(self):
        data=raw(pedals=(0,1,2,1,1,0,1,0,2),steering=(0,1,2,1,1,0,1,0,2))
        reference=blocks(data)
        for chunk in (1,2,3,4):
            result=blocks(data,chunk=chunk)
            for key in ('time','state','state_values','valid_samples','direction_history_valid'):
                np.testing.assert_array_equal(flatten(result,key),flatten(reference,key))
            np.testing.assert_array_equal(np.concatenate([b['transition_valid'] for b in result]),reference[0]['transition_valid'])

    def test_history_resets_at_every_new_log(self):
        for _ in range(2):
            result=blocks(raw(),chunk=1)
            self.assertFalse(result[0]['valid_samples'][0])
            self.assertEqual(int(np.concatenate([b['transition_valid'] for b in result]).sum()),3)

    def test_steering_outlier_is_removed_before_direction_calculation(self):
        data=raw(pedals=(0.,)*6,steering=(0,1,2000,3,4,5))
        b=blocks(data)[0]
        np.testing.assert_array_equal(b['valid_samples'],[False,True,False,False,True,True])
        self.assertEqual(b['steering_outliers_removed'],1)
        self.assertFalse(b['transition_valid'][2])
        self.assertFalse(b['transition_valid'][3])
        self.assertTrue(b['transition_valid'][4])

    def test_invalid_pedals_do_not_cancel_and_create_history(self):
        data=raw(pedals=(0.,110.,0.,0.,0.))
        data['brake'][1,1]=110.
        b=blocks(data)[0]
        np.testing.assert_array_equal(b['valid_samples'],[False,False,False,True,True])
        np.testing.assert_array_equal(b['transition_valid'],[False,False,False,True])

    def test_long_gap_does_not_supply_direction_history(self):
        data=raw(pedals=(0.,)*6,steering=(0,1,2,3,4,5),times=(0,.1,.2,2.,2.1,2.2))
        b=blocks(data,max_gap=.3)[0]
        t=b['time']
        ix=np.flatnonzero(np.isclose(t,2.))[0]
        self.assertFalse(b['valid_samples'][ix])
        self.assertTrue(b['valid_samples'][ix+1])

    def test_first_row_without_history_is_never_exported_as_observed_zero(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=config(n=2)
            settings=dict(max_gap=1.,chunk_samples=1,steering_limit_degrees=1260.)
            db=s4.open_training_db(Path(td)/'model.sqlite',cfg,settings)
            try:
                for name in ('a.mat','b.mat'):
                    report=s4.train_one_log(db,name,raw(),cfg,settings)
                    self.assertEqual(report['valid_transitions'],3)
                self.assertEqual(db.execute('SELECT SUM(c) FROM edges').fetchone()[0],6)
                s4.finalize_model(db,2)
                self.assertTrue(all(len(json.loads(key))==7 for key, in db.execute('SELECT value_key FROM states')))
                self.assertTrue(all(c>=2 for c, in db.execute('SELECT c FROM retained_edges')))
                before=db.execute('SELECT SUM(c) FROM edges').fetchone()[0]
                self.assertEqual(s4.train_one_log(db,'a.mat',raw(),cfg,settings)['status'],'already_complete')
                self.assertEqual(db.execute('SELECT SUM(c) FROM edges').fetchone()[0],before)
            finally:
                db.close()

    def test_decoded_diagnostics_distinguish_physical_and_direction_accuracy(self):
        cfg=config()
        state_map=s4._EvStates([(1,'[0,0,1,0,0,0,0]'),(2,'[0,0,1,0,0,1,-1]')],cfg)
        row=s4._s4_diagnostic_record()
        s4._s4_diagnostic_add(row,np.array([1]),np.array([1]),np.array([2]),np.array([7]),state_map,cfg)
        self.assertEqual(row['common_correct'],7)
        self.assertEqual(row['pedal_confusion'][2,1],7)
        self.assertEqual(row['steering_confusion'][0,1],7)
        self.assertEqual(row['events'],7)

    def test_unknown_test_states_keep_distinct_signs_in_diagnostics(self):
        cfg=config()
        state_map=s4._EvStates([(1,'[0,0,1,0,0,0,0]')],cfg)
        ids=state_map.ids(np.array([[0,0,1,0,0,-1,1]]))
        self.assertLess(ids[0],0)
        row=s4._s4_diagnostic_record()
        s4._s4_diagnostic_add(row,ids,ids,ids,np.array([3]),state_map,cfg)
        self.assertEqual(row['pedal_confusion'][0,0],3)
        self.assertEqual(row['steering_confusion'][2,2],3)


class SplitReuseTests(unittest.TestCase):
    def manifest(self,folder):
        result=[]
        for name,split in (('a.mat','train'),('b.mat','test')):
            p=folder/name;p.write_bytes(name.encode())
            stat=p.stat()
            result.append(dict(path=str(p),group_id=p.stem,split=split,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        path=folder/'split.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=result[0].keys());writer.writeheader();writer.writerows(result)
        return path,result

    def test_exact_split_is_reused_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);path,rows=self.manifest(folder)
            loaded=s4._s4_load_split_manifest(path)
            self.assertEqual([(x['path'],x['split']) for x in loaded],[(x['path'],x['split']) for x in rows])
            (folder/'a.mat').write_bytes(b'changed size')
            with self.assertRaises(s4.InputChangedError):
                s4._s4_load_split_manifest(path)

    def test_conflicting_input_options_are_rejected(self):
        for data,new in (([Path('other')],False),(None,True)):
            with self.assertRaisesRegex(ValueError,'cannot be combined'):
                s4._s4_inputs(SimpleNamespace(split_manifest=Path('split.csv'),data=data,new_split=new))


if __name__=='__main__':
    unittest.main()
