from pathlib import Path
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import run_three_scenarios as t


def scenario(signals=None,dt=.1,n=0):
    return dict(name='test',driver_signals=signals or ['pedal','parking','selector'],
                dt=dt,n=n,pedal_step=.5,speed_step=.5,steering_step=10.)


def raw(pedal=(0.,.5,1.),time=None):
    time=np.arange(len(pedal),dtype=float)*.1 if time is None else np.asarray(time,dtype=float)
    values={name:np.column_stack((time,np.zeros(len(time)))) for name in
            ('accelerator','brake','parking','selector','steering','door1','door2','door3','motor_left','motor_right')}
    values['accelerator'][:,1]=pedal
    return values


def collect(data,cfg,settings=None):
    settings={"max_gap":1.,"chunk_samples":200000,**(settings or {})}
    return list(t.iter_aligned_blocks(data,cfg,settings))


class AlignmentTests(unittest.TestCase):
    def test_duplicate_last_nan_and_sorted(self):
        prepared=t.prepare_raw_signal(np.array([[1.,1.],[0.,0.],[1.,np.nan],[2.,2.]]))
        np.testing.assert_array_equal(prepared['time'],[0.,1.,2.])
        self.assertTrue(np.isnan(prepared['value'][1]))
        self.assertEqual(prepared['duplicates_removed'],1)
        self.assertEqual(prepared['segment'].tolist(),[1,0,2])

    def test_sorted_fast_path_preserves_numeric_values(self):
        signal=np.column_stack((np.arange(100),np.arange(100)*2.))
        prepared=t.prepare_raw_signal(signal,max_gap=None)
        self.assertTrue(np.shares_memory(prepared['time'],signal))
        self.assertTrue(np.shares_memory(prepared['value'],signal))

    def test_gap_finite_exact_endpoints_cannot_bridge(self):
        data=raw((0.,1.),time=(0.,2.))
        blocks=collect(data,scenario(dt=2.))
        self.assertEqual(blocks[0]['valid_samples'].tolist(),[True,True])
        self.assertEqual(blocks[0]['transition_valid'].tolist(),[False])

    def test_internal_nan_skipped_by_grid_still_cuts_transition(self):
        data=raw((0.,np.nan,1.),time=(0.,.1,.2))
        blocks=collect(data,scenario(dt=.2))
        self.assertEqual(blocks[0]['transition_valid'].tolist(),[False])

    def test_internal_bad_category_skipped_by_grid_still_cuts_transition(self):
        data=raw((0.,0.,0.),time=(0.,.1,.2))
        data['parking'][:,1]=[0.,9.,0.]
        blocks=collect(data,scenario(dt=.2))
        self.assertEqual(blocks[0]['transition_valid'].tolist(),[False])

    def test_individual_invalid_pedals_do_not_cancel_into_valid_difference(self):
        data=raw((0.,110.,0.),time=(0.,.1,.2))
        data['brake'][:,1]=[0.,110.,0.]
        blocks=collect(data,scenario(dt=.2))
        self.assertEqual(blocks[0]['transition_valid'].tolist(),[False])

    def test_doors_or_after_independent_previous_interpolation(self):
        data=raw((0.,0.,0.,0.),time=(0.,.1,.2,.3))
        data['door1']=np.array([[0.,0.],[.2,1.],[.3,1.]])
        data['door2']=np.array([[.05,1.],[.15,0.],[.3,0.]])
        data['door3']=np.array([[0.,0.],[.3,0.]])
        blocks=collect(data,scenario(['doors'],dt=.1))
        np.testing.assert_allclose(blocks[0]['time'],[.05,.15,.25])
        np.testing.assert_array_equal(blocks[0]['state'][:,0],[1,0,1])

    def test_unknown_door_is_not_silently_closed(self):
        data=raw((0.,0.,0.))
        data['door2'][1,1]=np.nan
        blocks=collect(data,scenario(['doors']))
        self.assertFalse(blocks[0]['valid_samples'][1])
        self.assertFalse(np.any(blocks[0]['transition_valid']))

    def test_radians_to_unbounded_degrees_midpoints_up(self):
        data=raw((0.,0.,0.))
        data['steering'][:,1]=np.deg2rad([15.,-15.,1145.])
        blocks=collect(data,scenario(['steering']))
        np.testing.assert_array_equal(blocks[0]['state'][:,0],[2,-1,115])
        # Floating deg/rad round-trip at15 can lie below midpoint by 1ulp;
        # the numerical midpoint tolerance stabilizes that conversion.
        np.testing.assert_array_equal(t._round_ticks(np.array([15.,-15.]),10.),[2,-1])
        self.assertEqual(blocks[0]['state_values'][-1,0],1150.)

    def test_steering_limit_keeps_boundaries_and_rejects_outside(self):
        data=raw((0.,)*5)
        data['steering'][:,1]=np.deg2rad([-1260.,1260.,-1260.001,1260.001,1145.])
        block=collect(data,scenario(['steering']),dict(steering_limit_degrees=1260.))[0]
        np.testing.assert_array_equal(block['valid_samples'],[True,True,False,False,True])
        np.testing.assert_array_equal(block['state_values'][[0,1,4],0],[-1260.,1260.,1150.])
        self.assertEqual(block['steering_outliers_removed'],2)
        self.assertTrue(np.all(np.abs(block['state_values'][:,0])<=1260.))

    def test_steering_raw_radian_boundary_is_inclusive_and_nextafter_is_rejected(self):
        data=raw((0.,)*4)
        limit=np.deg2rad(1260.)
        data['steering'][:,1]=[-limit,limit,-np.nextafter(limit,np.inf),np.nextafter(limit,np.inf)]
        block=collect(data,scenario(['steering']),dict(steering_limit_degrees=1260.))[0]
        self.assertEqual(block['valid_samples'].tolist(),[True,True,False,False])

    def test_off_grid_steering_outlier_does_not_form_transition(self):
        data=raw((0.,0.,0.))
        data['steering'][:,1]=np.deg2rad([0.,2000.,0.])
        block=collect(data,scenario(['steering'],dt=.2),dict(steering_limit_degrees=1260.))[0]
        self.assertEqual(block['valid_samples'].tolist(),[True,True])
        self.assertEqual(block['transition_valid'].tolist(),[False])

    def test_steering_spike_cannot_create_interpolated_or_boundary_states(self):
        data=raw((0.,0.,0.))
        data['steering'][:,1]=np.deg2rad([0.,2000.,0.])
        block=collect(data,scenario(['steering'],dt=.05),dict(steering_limit_degrees=1260.))[0]
        self.assertEqual(block['valid_samples'].tolist(),[True,False,False,False,True])
        self.assertFalse(np.any(block['transition_valid']))
        np.testing.assert_array_equal(block['state_values'][block['valid_samples'],0],[0.,0.])

    def test_steering_limit_does_not_affect_scenarios_without_steering(self):
        data=raw((0.,0.,0.))
        data['steering'][:,1]=[0.,1e100,0.]
        for signals in (['pedal','parking','selector'],['pedal','parking','selector','doors']):
            block=collect(data,scenario(signals),dict(steering_limit_degrees=1260.))[0]
            self.assertTrue(np.all(block['transition_valid']))
            self.assertEqual(block['steering_outliers_removed'],0)

    def test_speed_floor_bin_and_current_environment(self):
        data=raw((0.,.5,1.))
        data['motor_left'][:,1]=[-.1,.1,.6]
        blocks=collect(data,scenario(),dict(speed_factor=1.))
        self.assertEqual(blocks[0]['environment'].tolist(),[-1,0,1])

    def test_chunk_boundaries_no_duplicated_or_missing_transitions(self):
        data=raw(tuple(np.arange(7,dtype=float)*.5))
        small=collect(data,scenario(),dict(chunk_samples=2))
        large=collect(data,scenario(),dict(chunk_samples=20))
        self.assertEqual(sum(np.count_nonzero(b['transition_valid']) for b in small),6)
        points=np.concatenate([b['state'][b['new_sample_start']:] for b in small])
        np.testing.assert_array_equal(points,large[0]['state'])

    def test_one_sample_and_nonoverlap_are_safe(self):
        one=collect(raw((0.,)),scenario())
        self.assertEqual(len(one[0]['time']),1)
        self.assertEqual(len(one[0]['transition_valid']),0)
        data=raw((0.,.5))
        data['parking'][:,0]+=10.
        self.assertEqual(collect(data,scenario()),[])


class CountingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.cfg=scenario(n=2)
        self.settings={'max_gap':1.,'chunk_samples':2,'speed_factor':1.}
        self.connection=t.open_training_db(Path(self.temp.name)/'model.sqlite',self.cfg,self.settings)

    def tearDown(self):
        self.connection.close(); self.temp.cleanup()

    def test_global_counts_strict_threshold_and_resume(self):
        t.train_one_log(self.connection,'a',raw((0.,.5)),self.cfg,self.settings)
        t.train_one_log(self.connection,'b',raw((0.,.5,1.)),self.cfg,self.settings)
        report=t.finalize_model(self.connection,2)
        self.assertEqual(report['raw_events'],3)
        self.assertEqual(report['kept_events'],2)
        self.assertEqual(report['raw_edges'],2)
        self.assertEqual(report['kept_edges'],1)
        self.assertEqual(t.train_one_log(self.connection,'a',raw((0.,.5)),self.cfg,self.settings)['status'],'already_complete')
        self.assertEqual(self.connection.execute('SELECT SUM(c) FROM edges').fetchone()[0],3)
        self.assertEqual(self.connection.execute('SELECT total FROM row_totals').fetchone()[0],2)

    def test_environment_not_pooled_and_no_cross_logs(self):
        first=raw((0.,.5)); second=raw((0.,.5))
        second['motor_left'][:,1]=1.
        t.train_one_log(self.connection,'a',first,self.cfg,self.settings)
        t.train_one_log(self.connection,'b',second,self.cfg,self.settings)
        report=t.finalize_model(self.connection,2)
        self.assertEqual(report['raw_events'],2)
        self.assertEqual(report['kept_events'],0)
        self.assertEqual(report['raw_edges'],2)

    def test_steering_outlier_audit_is_counted_once_across_chunks_and_resume(self):
        cfg=scenario(['steering'],n=0)
        settings={**self.settings,'chunk_samples':1,'steering_limit_degrees':1260.}
        data=raw((0.,)*7)
        data['steering'][:,1]=np.deg2rad([0.,0.,1261.,0.,-1261.,0.,0.])
        conn=t.open_training_db(Path(self.temp.name)/'limited.sqlite',cfg,settings)
        try:
            report=t.train_one_log(conn,'limited.mat',data,cfg,settings)
            self.assertEqual(report['steering_outliers'],2)
            self.assertEqual(report['valid_transitions'],2)
            self.assertEqual(t.finalize_model(conn,0)['steering_raw_outlier_samples'],2)
            self.assertEqual(t.train_one_log(conn,'limited.mat',data,cfg,settings)['status'],'already_complete')
            self.assertEqual(conn.execute('SELECT SUM(steering_outliers) FROM logs').fetchone()[0],2)
        finally:
            conn.close()

    def test_atomic_rollback_and_state_cache_invalidation(self):
        cfg=scenario(['steering'],n=0)
        settings={**self.settings,'chunk_samples':1}
        data=raw((0.,0.,0.,0.))
        data['steering'][:,1]=[0.,.1,.2,1e16]
        cache={}
        with self.assertRaises(ValueError):
            t.train_one_log(self.connection,'bad',data,cfg,settings,cache)
        self.assertEqual(cache,{})
        self.assertEqual(self.connection.execute('SELECT COUNT(*) FROM states').fetchone()[0],0)
        self.assertEqual(self.connection.execute('SELECT COUNT(*) FROM edges').fetchone()[0],0)
        self.assertFalse(t.log_is_complete(self.connection,'bad'))

    def test_empty_log_is_complete_but_not_fake_model_evidence(self):
        report=t.train_one_log(self.connection,'one',raw((0.,)),self.cfg,self.settings)
        self.assertEqual(report['status'],'no_valid_transitions')
        self.assertTrue(t.log_is_complete(self.connection,'one'))
        summary=t.finalize_model(self.connection,2)
        self.assertEqual(summary['raw_events'],0)

    def test_config_mismatch_rejected_and_n0_keeps_everything(self):
        path=Path(self.temp.name)/'model.sqlite'
        with self.assertRaises(ValueError):
            t.open_training_db(path,{**self.cfg,'dt':.2},self.settings)
        t.train_one_log(self.connection,'a',raw((0.,.5)),self.cfg,self.settings)
        self.assertEqual(t.finalize_model(self.connection,0)['kept_events'],1)

    def test_two_connection_race_rechecks_completion_after_write_lock(self):
        other=t.open_training_db(Path(self.temp.name)/'model.sqlite',self.cfg,self.settings)
        original=t.log_is_complete
        first=[True]
        def interleaved(connection,path):
            if connection is self.connection and first[0]:
                first[0]=False
                stale=original(connection,path)
                # Deterministically reproduce the second process committing
                # between the first process's outer check and BEGIN IMMEDIATE.
                t.train_one_log(other,path,raw((0.,.5)),self.cfg,self.settings)
                return stale
            return original(connection,path)
        try:
            with mock.patch.object(t,'log_is_complete',side_effect=interleaved):
                result=t.train_one_log(self.connection,'same.mat',raw((0.,.5)),self.cfg,self.settings)
            self.assertEqual(result['status'],'already_complete')
            self.assertEqual(self.connection.execute('SELECT SUM(c) FROM edges').fetchone()[0],1)
            self.assertEqual(self.connection.execute('SELECT COUNT(*) FROM logs').fetchone()[0],1)
        finally:
            other.close()


if __name__=='__main__':
    unittest.main()
