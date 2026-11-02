# S4 experimental checkpoint

Run `run_d81e43f1ced8`; model schema remains `vasiliy-observed-state-csr-v1`.
State column order: pedal, parking, selector, steering, doors, pedal_direction, steering_direction.
Direction columns are categorical -1/0/+1, with zero tolerance limited to floating-point roundoff. They describe recent observed movement, not measured psychological intent. Previous-value hold can produce zero signs between sensor updates; small fluctuations can change labels without changing quantized controls.

The exported model contains 687888 observed states and 5854920 retained transitions from 726371238 accepted training events. Count and probability checks reconcile every retained edge against the original SQLite database; maximum row-sum error is 4.44e-16. The exact band with a shared permutation would require about 10.680 TiB of numeric probability values; the compressed observed-row payload is 150.37 MiB. Neither estimate includes the decoder, counts or process overhead.

The predefined 4512/1134 train/test split matches S1-S3. Nevertheless, 2226 training paths and 553 test paths failed network access. Only 1005 training and 201 nonempty test logs were used. This model is a traceable exploratory checkpoint, not a complete-corpus S4 fit. Restoring access and resuming failed training paths changes the fitted model and requires new evaluation and checksums.

On the accepted 161497667 held-out transitions, filtered S4 has joint accuracy 77.6910% and joint-state change F1 0.671956. Decoding only the five physical controls gives 95.5851% accuracy and change F1 0.000527, compared with persistence accuracy 95.5886%. These decoded decisions are projections of the joint MAP state; no marginalized five-control Brier/NLL or paired physical-accuracy confidence interval is claimed. Cross-configuration joint scores cannot be ranked because the output alphabet and usable data differ.

True zero signs account for 97.3466% of pedal-direction and 80.7730% of steering-direction test labels. Filtered balanced sign accuracies are 33.0793% and 38.2361%. Ordinary sign accuracy is therefore insufficient to show meaningful intent prediction. Strict NLL remains infinite because test events have zero fitted probability; clipped NLL uses epsilon=1e-12.

Recorded-speed replay remains a limited diagnostic, with three usable selected logs and three realizations, not a closed-loop vehicle test. Joint occupancy TV is 0.8074; generated fallback is 22.6828%; pedal and steering Wasserstein distances are 9.580 percentage points and 95.082 degrees. Direction labels are sampled jointly and are not a deterministic derivative constraint on the generated quantized physical path.

Only aggregate tables, source code, compact models and checksums are distributed. Operational logs, internal path names and original per-log manifests are excluded. No additional open-source license is assigned by this update.
