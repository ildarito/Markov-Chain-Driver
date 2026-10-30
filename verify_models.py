"""Verify distributed MAT exports without access to operational logs."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.io import loadmat


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / 'models' / 'manifest.json').read_text(encoding='utf-8'))
    reports = []
    for item in manifest['models']:
        path = root / item['file']
        assert path.stat().st_size == item['bytes'], (path.name, 'size')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item['sha256'], (path.name, 'checksum')
        m = loadmat(path, simplify_cells=True)
        assert m['schema'] == 'vasiliy-observed-state-csr-v1'
        ids = np.asarray(m['state_ids']).reshape(-1)
        n = len(ids)
        assert np.array_equal(ids, np.arange(1, n + 1))
        signals = np.asarray(m['driver_signal_names']).reshape(-1)
        assert np.asarray(m['state_values']).shape == (n, len(signals))
        assert np.all(np.isfinite(m['state_values']))
        e = np.asarray(m['row_speed_bin']).reshape(-1)
        i = np.asarray(m['row_current_state']).reshape(-1)
        offsets = np.asarray(m['row_start_1based']).reshape(-1).astype(np.int64) - 1
        targets = np.asarray(m['target_state']).reshape(-1)
        p = np.asarray(m['probability']).reshape(-1)
        count = np.asarray(m['retained_count']).reshape(-1)
        assert len(e) == len(i) == len(offsets) - 1
        assert len(targets) == len(p) == len(count)
        assert offsets[0] == 0 and offsets[-1] == len(p)
        assert np.all(np.diff(offsets) > 0)
        assert np.all((i >= 1) & (i <= n))
        assert np.all((targets >= 1) & (targets <= n))
        assert np.all((e[1:] > e[:-1]) | ((e[1:] == e[:-1]) & (i[1:] > i[:-1])))
        assert np.all(np.isfinite(p)) and np.all((p > 0) & (p <= 1))
        assert np.all(count >= int(m['minimum_count']))
        sums = np.add.reduceat(p, offsets[:-1])
        error = float(np.max(np.abs(sums - 1)))
        assert error < 1e-12, error
        totals = np.add.reduceat(count.astype(np.float64), offsets[:-1])
        expected = count / np.repeat(totals, np.diff(offsets))
        assert np.allclose(p, expected, rtol=1e-13, atol=1e-15)
        assert float(m['dt']) > 0 and float(m['speed_step_kmh']) > 0
        # Configuration is distributable metadata, without internal log paths.
        config = str(m['configuration_json'])
        assert 'ic-server' not in config and '\\Users\\' not in config
        reports.append(dict(scenario=item['scenario'], sha256=digest, states=n,
                            observed_rows=len(i), retained_edges=len(p),
                            max_row_sum_error=error, status='PASS'))
    return reports


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
