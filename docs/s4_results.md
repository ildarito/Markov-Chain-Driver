# S4: model and evaluation results

The distributed S4 model is identified by SHA-256 `d8d951208fd6a2f427d9379708f40a5178fac10fdd7f163d808552343da16c15`. Its export metadata and checksum are recorded in `models/manifest.json`.

The driver-state components are: pedal, parking, selector, steering, doors, pedal_direction, steering_direction. The signs are ternary -1/0/+1 backward changes before quantization, using previous-held raw controls and roundoff-only zero tolerance. Base physical steering retains the S3 offline linear interpolation.

Accepted file cohorts match S3 exactly: 1770 training and 408 nonempty test logs. Of the fixed 4512 training and 1134 test entries, 2712 and 711 lack required channels; 30 and 15 successfully loaded logs are empty. S4 contains 1420933277 training and 316766519 held-out transitions, 2848 and 583 fewer than S3 because of history validity. All accepted S3/S4 file identities were cross-checked, together with per-log transition counts.

The model has 851541 observed states, 41024322 raw conditional edge types and 9390957 retained edges. Pruning count<5 removes 3.47848% of training events and 77.10881% of edge types; 6294746 previously supported rows become empty. The numeric sparse payload is 232.07 MiB, excluding counts, decoder and process overhead. The exact shared band would require 27651852461520 bytes (25.149 TiB) with lower and upper bandwidth 8527. The MAT export is 77380055 bytes. A read-only audit reconciles all retained edges with SQLite counts and probabilities; maximum row-sum error is 5.55e-16.

Filtered joint Brier is 0.346851 versus 0.333945 unfiltered, 0.330051 without speed and 0.619023 for persistence. Whole-log paired Brier intervals, based on 408 logs and 200 draws, are [0.011521, 0.014363] versus unfiltered and [0.015364, 0.018666] versus the unconditional chain. Strict NLL is infinite; zero-probability outcomes account for 4.17818% of filtered test events. The separate clipped NLL is 1.75038 with epsilon=1e-12.

Joint accuracy is 78.8799% and joint-state change F1 is 0.680100. Decoding the original five physical controls gives 95.4769% accuracy and change F1 0.000362, compared with persistence accuracy 95.4778%. These are decoded joint-MAP decisions, not marginal-distribution scores. The accepted file coverage matches S3, but target alphabets and history-valid transition sets still differ; an exact paired physical comparison is not supplied by the saved aggregates.

Zero signs occupy 96.8811% of pedal and 82.1968% of steering targets. Filtered balanced sign accuracies are 33.1176% and 37.1149%, so ordinary accuracy and joint-state F1 do not establish intent recognition. Recorded-speed replay remains limited to three usable selected logs and three realizations, with joint occupancy TV 0.88751, fallback 23.1288%, pedal Wasserstein distance 6.234 percentage points and steering distance 117.979 degrees. It is not a closed-loop vehicle test.

Operational logs and original path-bearing manifests remain undistributed. All public tables, current weights, and checksum metadata have been updated together.
