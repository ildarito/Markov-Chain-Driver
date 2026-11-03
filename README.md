# Markov-Chain-Driver

Speed-conditioned, count-based Markov models of joint driver controls for urban electric transport.
Companion implementation and **four trained models (S1–S4)** for *Environment-Dependent Markov Models of Driver Actions for Urban Electric Transport*, by Ildar Ahtamovich Akhmetshin.

## Quick start

Python 3.9 or later:

```powershell
python -m pip install -r requirements.txt
```

Train and evaluate S1, S2 and S3 sequentially:

```powershell
python run_three_scenarios.py --data "D:\my_logs" --output "results/runs"
```

Train and evaluate **S4 only**:

```powershell
python run_fourth_scenario.py --data "D:\my_logs" --output "results/runs_s4"
```

The first script runs three scenarios; the second runs the fourth scenario separately. Both save compact MAT models, JSON/CSV metrics and PNG/PDF figures. Repeating a command with unchanged input files and settings resumes completed training logs. Raw logs are not included. Without `--data`, the scripts use `data/` or the `MARKOV_DATA_DIR` environment variable.

| Scenario | Driver state | dt, s | Minimum count n | Runner |
| --- | --- | ---: | ---: | --- |
| S1 | signed pedal, parking brake, selector | 0.1 | 10 | `run_three_scenarios.py` |
| S2 | S1 + any door open | 0.01 | 5 | `run_three_scenarios.py` |
| S3 | S2 + steering angle | 0.01 | 5 | `run_three_scenarios.py` |
| **S4** | **S3 + pedal-change sign + steering-change sign** | **0.01** | **5** | **`run_fourth_scenario.py`** |

Speed is the environment signal in all four scenarios. Pedal bins are 0.5 percentage points, speed bins are 0.5 km/h and steering bins are 10 degrees. Steering is supplied in radians; samples outside ±1260 degrees are rejected before interpolation, breaking affected transitions. Three door channels are combined by logical OR. The signed pedal is accelerator minus brake in percent; equal differences do not distinguish simultaneous accelerator/brake inputs.

Counts are pooled across training logs before removing transition types observed strictly fewer than `n` times. A count equal to `n` is retained. Empty or absent state–speed rows hold the current state. Signal names, channels, steps and thresholds can be changed in the settings at the top of each runner.

## Input and outputs

Each MAT signal must be a numeric `[time_seconds, value]` array or a supported time/value structure. The `MAT_SIGNALS` mapping names the required channels. Speed is derived from left/right motor rpm using the configured wheel radius and gear ratio; adjust these for your vehicle. Files lacking required channels are reported and excluded per scenario.

The 80/20 split groups matching log basenames. This does not establish independence between drivers, vehicles, routes or days. Outputs include six prediction rules, Brier score, strict and clipped NLL, accuracy, zero-probability and fallback rates, change detection, calibration, whole-log bootstrap intervals, speed-specific scores, storage/band diagnostics and limited recorded-speed generation. Scores across different state definitions or time steps are not directly comparable.

## S4: recent pedal and steering changes

S4 adds two **ternary features, −1, 0 and +1**, representing decrease, no change and increase. They are signs of one-step backward differences of previous-held raw signed-pedal and steering measurements, computed before quantization. These are recent-control-change indicators rather than Boolean flags or measured psychological intentions.

Samples without continuous preceding history are excluded. Base physical steering retains S3's offline linear interpolation; the two added signs use previous-held measurements. The signs are fitted as components of the joint state and are not recomputed from generated quantized controls.

To reuse the exact split produced by an S1–S3 run, pass its `split_manifest.csv`:

```powershell
python run_fourth_scenario.py --split-manifest "results/runs/run_ID/split_manifest.csv" --output "results/runs_s4"
```

Replace `run_ID` with the actual run directory. `--split-manifest` checks file integrity and is used instead of `--data`. The original experimental split contains restricted operational paths and is not distributed.

The reported S3 and S4 models share 1770 accepted training logs and 408 nonempty test logs. S4 requires valid preceding history, so their valid transition sets differ slightly. The additional labels change the prediction target; joint-state scores must be distinguished from physical-control scores. See [S4 diagnostics](docs/s4_results.md) and `results/s4_coverage_audit.json` for detailed results.

## Trained models and reported results

`models/` contains compact MAT exports for **S1, S2, S3 and S4**. `models/manifest.json` identifies their SHA-256 checksums and experimental provenance. [The format guide](docs/model_format.md) and `models/format.json` describe one-based state IDs and compressed observed rows. MATLAB can load the files using `load`; these exports are not full Cartesian transition matrices.

| Scenario | Numeric sparse probability storage, MiB |
| --- | ---: |
| S1 | 6.58 |
| S2 | 9.51 |
| S3 | 98.57 |
| **S4** | **232.07** |

These values describe numeric probability/index buffers before MAT compression and exclude counts, decoders and process overhead. Sparse storage is smaller than the corresponding exact bands. Aggregate results are in `results/scenario_table.csv` and `results/baseline_table.csv`.

For S4, joint-state Brier is 0.3469 versus 0.6190 for holding and 0.3301 for the chain without speed. Decoded five-control accuracy rounds to 95.48% for both the filtered conditional model and holding; the unrounded results show no physical-action accuracy advantage. Zero-probability test outcomes make strict NLL infinite. Recorded-speed generation shows distributional and temporal discrepancies; coupled-vehicle behavior has not been validated.

Raw operational logs, internal directory names, per-log audit files and the original split manifest are not distributed. Reproducing the original fit requires those inputs. A new training run creates its own manifest locally.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python verify_models.py
```

The checks cover counting, discretization, filtering, normalization, band/permutation equivalence, evaluation, splitting and runner behavior. Model verification checks hashes, indices and row normalization without accessing the original logs.

## Author and availability

Ildar Ahtamovich Akhmetshin — Moscow Polytechnic University; Innovation Center KAMAZ LLC.
Contact: ildarakhmetshin2112@gmail.com.
Repository: https://github.com/ildarito/Markov-Chain-Driver

No open-source license has been selected for this release. Public visibility permits viewing; it does not itself grant a software or data license. No redistribution license for raw operational records is provided.
