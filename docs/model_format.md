# Compact trained-model format

Schema: `vasiliy-observed-state-csr-v1`. Load with `scipy.io.loadmat(..., simplify_cells=True)` or MATLAB `load`.

| Field | Meaning |
| --- | --- |
| state_ids | Contiguous observed state IDs, 1 through N |
| state_ticks, state_values | N-by-D discrete tuples and physical signal values |
| driver_signal_names | Column order in the tuples |
| row_speed_bin, row_current_state | Lexicographically sorted keys (speed bin, current state ID) |
| row_start_1based | R+1 offsets into retained edges, first=1 and last=E+1 |
| target_state, probability, retained_count | Parallel E-element vectors for retained edges |
| initial_state_ids, initial_state_counts | Training initial-state distribution |
| dt, minimum_count | Sampling step in seconds and global training count threshold |
| speed_step_kmh, pedal_step_percent, steering_step_degrees | Quantization steps |
| configuration_json | Scenario configuration, without input-log paths |

For speed v use `e=floor(v/speed_step_kmh)`; a signed speed bin e represents `[e*step,(e+1)*step)`. It is not a one-based context ID. Find the observed row `(e,current_state_id)`; if none exists, hold the current state. For a found row r, sample one target using its normalized probability entries. MATLAB edge indices are `row_start_1based(r):row_start_1based(r+1)-1`. Python subtracts one from both offsets to form a half-open slice. Decode with `state_values(target_id,:)` in MATLAB or `state_values[target_id-1]` in Python. Advance at the saved dt; changing the sampling time does not preserve the learned process.

Only observed rows are stored. Do not reshape this representation into an N-by-N-by-K tensor. No speed extrapolation or nearest-row substitution is implicit in the format.
