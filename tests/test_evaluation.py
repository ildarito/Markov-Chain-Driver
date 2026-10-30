import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

import run_three_scenarios as ev
tr=ev


def scenario(n=0,drivers=None):
    return dict(name='fixture',driver_signals=drivers or ['pedal','parking','selector'],dt=.1,
                n=n,pedal_step=.5,speed_step=.5,steering_step=10.)

def raw_sequence(pedals=(0,0,1,1,2,2,1,1,0,0),**extra):
    t=np.arange(len(pedals))*.1
    raw={'accelerator':np.column_stack((t,pedals)),
         'brake':np.column_stack((t,np.zeros(len(t)))),
         'parking':np.column_stack((t,np.zeros(len(t)))),
         'selector':np.column_stack((t,np.ones(len(t)))),
         'motor_left':np.column_stack((t,np.zeros(len(t)))),
         'motor_right':np.column_stack((t,np.zeros(len(t))))}
    raw.update(extra)
    return raw

class SparseRowsTests(unittest.TestCase):
    def test_exact_normalization_profiles_and_query(self):
        r=ev._EvRows([[1,1,1,1],[1,1,2,3],[2,2,1,4]])
        s=r.scores(np.array([1,1,2]),np.array([1,1,2]),np.array([1,2,1]))
        np.testing.assert_allclose(s['p'],[.25,.75,1])
        np.testing.assert_array_equal(s['winner'],[2,2,1])
        np.testing.assert_allclose(s['sumsq'],[.625,.625,1])
        self.assertTrue(np.all(s['top5']))
        self.assertEqual(len(r.rowkeys),2)

    def test_duplicate_edges_aggregate_before_normalization(self):
        r=ev._EvRows([[0,1,2,1],[0,1,1,2],[0,1,2,1]])
        self.assertEqual(len(r.edges),2)
        self.assertAlmostEqual(r.scores([0],[1],[2])['p'][0],.5)
        self.assertEqual(r.scores([0],[1],[2])['winner'][0],1)

    def test_unseen_negative_state_fallback_is_selfloop(self):
        r=ev._EvRows([[0,1,1,5]])
        s=r.scores([0,7],[-1,1],[-1,2])
        np.testing.assert_array_equal(s['p'],[1,0])
        np.testing.assert_array_equal(s['fallback'],[True,True])

    def test_context_query_preserves_actual_current_for_missing_row(self):
        r=ev._EvRows([[0,0,2,3]])
        s=r.scores([4],[0],[7],fallback_i=np.array([7]))
        self.assertEqual(s['p'][0],1)
        self.assertEqual(s['winner'][0],7)

    def test_sampler_and_empty_row(self):
        class Fixed:
            def __init__(self,x):self.x=x
            def random(self):return self.x
        r=ev._EvRows([[0,1,1,1],[0,1,2,3]])
        self.assertEqual(r.sample(0,1,Fixed(0)),(1,False))
        self.assertEqual(r.sample(0,1,Fixed(.25)),(2,False))
        self.assertEqual(r.sample(0,2,Fixed(.9)),(2,True))

    def test_empty_sparse_model(self):
        r=ev._EvRows([])
        self.assertEqual(r.scores([2],[7],[7])['p'][0],1)
        self.assertEqual(r.sample(2,7,np.random.default_rng(3)),(7,True))

class MetricsTests(unittest.TestCase):
    def test_weighted_known_nll_brier_and_classification(self):
        r=ev._EvRows([[0,1,1,1],[0,1,2,3]])
        s=r.scores([0,0],[1,1],[1,2])
        record=ev._ev_record()
        ev._ev_add(record,s,np.array([1,1]),np.array([1,2]),np.array([1,3]))
        summary=ev._ev_summary([record])
        self.assertAlmostEqual(summary['strict_nll_nats'],-(.25*math.log(.25)+.75*math.log(.75)))
        self.assertAlmostEqual(summary['brier'],.375)
        self.assertAlmostEqual(summary['top1_accuracy'],.75)
        self.assertAlmostEqual(summary['change_precision'],.75)
        self.assertAlmostEqual(summary['change_recall'],1)

    def test_strict_zero_probability_is_infinite_not_clipped(self):
        r=ev._EvRows([[0,1,1,2]])
        record=ev._ev_record()
        ev._ev_add(record,r.scores([0],[1],[2]),np.array([1]),np.array([2]),np.array([7]))
        summary=ev._ev_summary([record])
        self.assertTrue(summary['strict_nll_is_infinite'])
        self.assertIsNone(summary['strict_nll_nats'])
        self.assertAlmostEqual(summary['clipped_nll_nats'],-math.log(1e-12))
        self.assertEqual(summary['brier'],2)

    def test_strict_positive_tiny_probability_not_clipped(self):
        r=ev._EvRows([[0,1,1,1],[0,1,2,10**14]])
        record=ev._ev_record()
        ev._ev_add(record,r.scores([0],[1],[1]),np.array([1]),np.array([1]),np.array([1]))
        summary=ev._ev_summary([record])
        self.assertGreater(summary['strict_nll_nats'],summary['clipped_nll_nats'])
        self.assertFalse(summary['strict_nll_is_infinite'])

    def test_dwell_excludes_both_segment_boundaries(self):
        self.assertEqual(ev._ev_complete_dwells([1,1,2,2,3,3],[True]*5),[2])
        self.assertEqual(ev._ev_complete_dwells([1,1,2,2,3,3],[True,True,False,True,True]),[])
        self.assertEqual(ev._ev_complete_dwells([1,1],[True]),[])

    def test_survival_handles_duplicate_durations_and_ends_at_zero(self):
        x,y=ev._ev_survival({2:2,3:1},.1)
        np.testing.assert_allclose(x,[0,.2,.3])
        np.testing.assert_allclose(y,[1,1/3,0])

    def test_acf_never_crosses_gap(self):
        result=ev._ev_acf([0,1,100,99],[True,False,True],[1,2])
        self.assertIsNotNone(result[0]);self.assertIsNone(result[1])

    def test_wasserstein_physical_units(self):
        d=ev._ev_distribution({0:1},{10:1},True)
        self.assertEqual(d['tv'],1)
        self.assertAlmostEqual(d['js_nats'],math.log(2))
        self.assertEqual(d['wasserstein'],10)
        self.assertIsNone(ev._ev_distribution({0:1},{10:1},False)['wasserstein'])

    def test_replay_chunk_overlap_counted_once(self):
        raw=raw_sequence()
        data=ev._ev_collect_replay(raw,scenario(),dict(chunk_samples=2,max_gap=1,max_replay_samples=7),tr.iter_aligned_blocks)
        self.assertEqual(len(data['time']),7)
        self.assertEqual(len(data['transition_valid']),6)
        np.testing.assert_allclose(np.diff(data['time']),.1)

class IntegrationTests(unittest.TestCase):
    def make_database(self,folder,n=0):
        db=Path(folder)/'counts.sqlite'
        cfg=scenario(n)
        settings=dict(max_gap=1,chunk_samples=3,make_figures=False,eval_rollouts=1,
                      max_replay_files=1,max_replay_samples=20,bootstrap=10,seed=4)
        c=tr.open_training_db(db,cfg,settings)
        tr.train_one_log(c,str(Path(folder)/'train.mat'),raw_sequence(),cfg,settings)
        tr.finalize_model(c);c.close()
        return db,cfg,settings

    def test_end_to_end_model_and_metrics(self):
        from scipy.io import loadmat
        with tempfile.TemporaryDirectory() as folder:
            db,cfg,settings=self.make_database(folder)
            result=ev.evaluate_scenario(db,[Path(folder)/'test.mat'],cfg,settings,
                lambda *args:raw_sequence(),Path(folder)/'out',block_iterator=tr.iter_aligned_blocks)
            self.assertEqual(result['test_events'],9)
            self.assertEqual(result['training_observations'],9)
            mat=loadmat(Path(folder)/'out'/'model_compact.mat',squeeze_me=True)
            self.assertEqual(int(mat['row_start_1based'][0]),1)
            self.assertEqual(int(mat['row_start_1based'][-1]),len(np.atleast_1d(mat['probability']))+1)
            self.assertTrue((Path(folder)/'out'/'scenario_summary.json').is_file())

    def test_train_test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            db,cfg,settings=self.make_database(folder)
            with self.assertRaisesRegex(ValueError,'both training and test'):
                ev.evaluate_scenario(db,[Path(folder)/'train.mat'],cfg,settings,
                    lambda *args:raw_sequence(),Path(folder)/'out',block_iterator=tr.iter_aligned_blocks)

    def test_pruned_empty_rows_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            db,cfg,settings=self.make_database(folder,n=100)
            result=ev.evaluate_scenario(db,[Path(folder)/'test.mat'],cfg,settings,
                lambda *args:raw_sequence(),Path(folder)/'out',block_iterator=tr.iter_aligned_blocks)
            self.assertEqual(result['retained_edge_types'],0)
            self.assertEqual(result['test_fallback_fraction'],1)

    def test_failed_partial_log_rolls_back_scores_and_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            db,cfg,settings=self.make_database(folder)
            settings['eval_rollouts']=0
            def failing(raw,cfg,settings):
                blocks=tr.iter_aligned_blocks(raw,cfg,settings)
                yield next(blocks)
                raise ValueError('simulated second-chunk failure')
            result=ev.evaluate_scenario(db,[Path(folder)/'test.mat'],cfg,settings,
                lambda *args:raw_sequence(),Path(folder)/'out',block_iterator=failing)
            metrics=json.loads((Path(folder)/'out'/'metrics.json').read_text(encoding='utf8'))
            self.assertEqual(result['test_events'],0)
            self.assertEqual(sum(x['events'] for x in metrics['calibration']),0)
            self.assertEqual(metrics['by_speed'],[])

    def test_shared_band_permutation_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=np.array([[0,1,4,1],[1,4,2,10],[1,2,3,7],[0,3,1,2]])
            db=Path(folder)/'fake.sqlite';db.write_bytes(b'fixture')
            summary,plot=ev._ev_structural(raw,4,db,scenario(5),Path(folder))
            self.assertTrue(all(x['band_coordinate_roundtrip'].startswith('PASS') for x in summary))
            selected=next(x for x in summary if x['n']==5)
            self.assertEqual(selected['retained_edge_types'],2)
            self.assertEqual(selected['removed_observations'],3)


if __name__=='__main__':
    unittest.main()
