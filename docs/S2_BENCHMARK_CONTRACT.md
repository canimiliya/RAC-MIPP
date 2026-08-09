# S2 Benchmark and Evaluation Contract

Status: frozen on `benchmark/s2-r0` for controller review. This contract changes no reward, environment, policy, or communication semantics.

## Split and seed boundary

All algorithms use the exact seed lists in `artifacts/s2/r0/benchmark_contract.json`. Train, validation, IID test, and OOD test are disjoint. Validation alone may select checkpoints or tune hyperparameters. Test roles require an explicit final-test acknowledgement in code and are rejected for checkpoint selection, early stopping, tuning, or development. A poor result never permits seed replacement.

The S1 COMA run predates this contract and remains a preserved parity anchor. Its IID test seeds `20001..20050` become the common IID test set; its training seed is not silently reclassified as a future formal baseline seed.

Communication-condition fields exist for every split, but their values remain `INTERFACE_RESERVED_FOR_S3`. S2 implements no packet loss, delay, or communication noise.

## Common evaluator

`scripts/s2/run_benchmark_evaluation.py` invokes the algorithm-neutral driver in `rac_mipp.benchmark.evaluator`. A policy adapter returns raw episode outcomes; the driver owns split validation, common metric names, aggregation, CSV and JSON serialization. The current `OriginalCOMAAdapter` validates the S1 checkpoint hash and runs the pinned upstream environment. Future algorithms must implement the same small adapter boundary instead of duplicating statistics.

Current metrics are entropy, positive-class F1, mission return, episode length, measurement count, and total team path length. Communication load, packet delivery, tail risk, and OOD gap are schema fields with `NOT_AVAILABLE_YET`; zero is not a missing value.

## Communication observation

The hook API can observe copied position snapshots and can receive future attempted/delivered-message events. It returns no actions or state to the environment. Hook-off and passive-hook modes must produce identical policy/environment metrics. Proximity is an observation, not proof of a sent packet; message and byte fields therefore remain unavailable until S3 supplies real events.

## Fair baseline budget

The primary equalizer is environment agent interactions per formal training seed. COMA, IPPO, MAPPO, RMAPPO, and Graph-MAPPO must use the same formal training seeds, interaction budget, environment/splits, validation-based checkpoint rule, and final evaluation seeds/trials. Optimizer updates, parameter count, training wall-clock, inference latency, and hardware are recorded separately. Parameter counts need not be identical, but no method may receive extra interactions after its result is known.

## Required run record

Every formal run must satisfy `artifacts/s2/r0/run_manifest_schema.json` and the runtime validator. Required fields are RUN_ID, TASK_ID, GIT_HEAD, UPSTREAM_COMMIT, CONFIG_HASH, SEED, ALGORITHM, ENVIRONMENT, TEAM_SIZE, COMM_DROP, COMM_DELAY, START_TIME, END_TIME, STATUS, PRIMARY_METRICS, and ARTIFACT_PATHS. Failed and interrupted runs remain recorded.

## Determinism and parity gates

The same checkpoint, config, and seed must repeat within absolute tolerance `1e-12` for the present CPU-environment metrics. The complete 50-trial unified-evaluator output is compared per trial and in aggregate with the preserved S1 CSV/summary at the same tolerance. If future GPU kernels require a wider tolerance, it must be preregistered before results are inspected.
