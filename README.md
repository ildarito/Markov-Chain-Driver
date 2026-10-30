# Markov-Chain-Driver

Speed-conditioned, count-based Markov models of joint driver controls for urban electric transport.
Companion implementation and three trained models for *Environment-Dependent Markov Models of Driver Actions for Urban Electric Transport*, by Akhmetshin Ildar Ahtamovich.

## Quick start

Python 3.9 or later:

```powershell
python -m pip install -r requirements.txt
python run_three_scenarios.py --data "D:\my_logs" --output "results/runs"
```

The command trains all three scenarios sequentially, evaluates held-out whole logs and saves MAT models, JSON/CSV metrics, and PNG/PDF figures. Edit the settings at the top of the single script to change signal names, driver channels, time steps and thresholds. Raw logs are not included. Without `--data`, the script reads `data/` or the `MARKOV_DATA_DIR` environment variable. Repeating the same command with unchanged files/settings resumes committed training logs.

| Scenario | Driver state | dt, s | Minimum count n |
| --- | --- | ---: | ---: |
| S1 | signed pedal, parking brake, selector | 0.1 | 10 |
| S2 | S1 + any door open | 0.01 | 5 |
| S3 | S2 + steering angle | 0.01 | 5 |

Speed is the environment signal in every scenario. Pedal bins are 0.5 percentage points, speed bins 0.5 km/h, and steering bins 10 degrees. Steering is supplied in radians; samples outside ±1260 degrees are rejected before interpolation, breaking affected transitions. Three door channels are combined by logical OR. Pedal is accelerator minus brake in percent; this representation cannot distinguish simultaneous accelerator/brake inputs that have the same difference. Counts are pooled across training logs before removing edges with count strictly less than n. Empty or absent rows hold the current state.

## Input and outputs

Each MAT signal must be a numeric `[time_seconds, value]` array or a supported time/value structure. The `MAT_SIGNALS` mapping names the required channels. Speed is derived from left/right motor rpm using the wheel radius and gear ratio in the settings; adjust these for your vehicle. Files lacking required channels are reported and excluded per scenario.

The 80/20 split groups matching log basenames; it does not establish independence between drivers, routes or days. Results include six predictors, exact-state Brier score, strict and clipped NLL, accuracy, support/fallback rates, change detection, calibration, log bootstrap intervals, speed-conditioned scores, storage/band diagnostics and limited recorded-speed rollouts. NLL and exact-state accuracy across different state definitions or dt are not directly comparable.

## Trained models and reported results

`models/` contains the original compact MAT exports for S1, S2 and S3. `models/manifest.json` records SHA-256 checksums, run ID and the original experimental program checksum. The portable runner changes only default paths, dependency discovery and a missing-input message; changing the program hash produces a new run directory. Model parameters are not refitted. `models/format.json` and [the format guide](docs/model_format.md) describe one-based state IDs and compressed observed rows. MATLAB can load these files using `load`; they are not full Cartesian transition matrices.

`results/scenario_table.csv` and `results/baseline_table.csv` contain the article's aggregate results. Probability payloads occupy approximately 6.58, 9.51 and 98.57 MiB before MAT compression. They are smaller than the corresponding exact bands. High one-step accuracy largely reflects persistence. Zero-probability test events make strict NLL infinite; clipped NLL uses epsilon=1e-12. Recorded-speed rollouts show substantial distributional mismatch. These models have not been validated as realistic closed-loop virtual drivers.

Raw operational logs, internal directory names, per-log audit files and the original split manifest are not distributed. Exact reproduction of the reported training run requires those restricted inputs. A fresh run creates its own manifest locally.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python verify_models.py
```

Tests cover counting, discretization, pruning, normalization, band/permutation equivalence, evaluation, splitting and runner behavior. Model verification checks hashes, indexing and row normalization without accessing the logs.

## Author and availability

Akhmetshin Ildar Ahtamovich — Moscow Polytechnic University; Innovation Center KAMAZ LLC.
Contact: ildarakhmetshin2112@gmail.com.
Repository: https://github.com/ildarito/Markov-Chain-Driver

No open-source license has been selected for this release. Public visibility permits viewing; it does not itself grant a software or data license. No redistribution license for raw operational logs is provided.
