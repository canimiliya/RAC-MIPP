# S1 Formal Original-COMA Reproduction Protocol

This document preregisters protocol revision 1 for task `S1-R1-ORIGINAL-COMA-FORMAL-REPRODUCTION-R1`. It is frozen before the first formal long-training run. The formal test results may not be used to revise this run's configuration, checkpoint choice, seeds, evaluator, or acceptance gate.

## Scope and provenance

- Method: original `ipp-marl` COMA only, frozen upstream commit `1e9bdc3ba90f707ce79797468f533f5733c65e4b`.
- Primary target: P01 Table I, four UAVs, 25 m communication radius, 15 measurements per UAV, 50 synthetic trials.
- Paper target: entropy `0.2842 +/- 0.0408`; F1 `0.7858 +/- 0.0308`.
- No RAC-MIPP algorithm, packet/drop/delay model, IPPO, MAPPO, RMAPPO, GAT, graph policy, or risk objective is in scope.

## Training protocol

- Formal training seed: `20260809`; exactly one training seed is preregistered.
- Total budget: 1,500 collection/update cycles, 3,000 agent transitions per collection, totaling 4,500,000 agent transitions and 75,000 missions.
- Budget provenance: `CODE_DERIVED`, not `PAPER_STATED`. The original uploaded `params.yaml` annotated `n_episodes: 1500` as `full batch fillings`; the original paper-aligned configuration used batch size 600 and five batches per collection. The current outer-loop formula therefore expands 1,500 fillings into 75,000 missions.
- Each mission has four agents and 15 measurement/action timesteps (`budget=14` in the inclusive upstream loop), yielding 60 agent transitions.
- After every 50 missions, optimize five epochs over five randomly shuffled batches of 600 transitions.
- Optimizers: upstream Adam; actor learning rate `1e-5`; critic learning rate `1e-4`.
- Returns: upstream `gamma=0.99`, TD `lambda=0.8`.
- Exploration: epsilon linearly decreases from `0.5` to `0.02` over the first 10,000 training missions, then remains at `0.02`.
- Target critic: hard copy from the online critic every ten collection cycles, equal to 30,000 agent transitions. TD targets use that synchronized target. This resolves the upstream duplicate-target-object bug in favor of the paper-stated 30,000-transition semantics.
- Initial positions: fixed paper-stated four-corner positions `(10,10,15)`, `(40,10,15)`, `(40,40,15)`, `(10,40,15)` metres. This resolves the active-code/generated-position discrepancy in favor of the paper.
- Stopping: stop after exactly 1,500 completed update cycles. There is no performance-based early stopping.
- Model selection: final actor checkpoint only. Periodic checkpoints exist solely for crash recovery and cannot be selected by Table I performance.

## Compatibility and performance adapters

The upstream checkout remains unmodified. The runner applies only recorded in-process adapters:

1. correct `BatchMemory.get("mask")` from nonexistent `transition.masks` to `transition.mask`;
2. bind TD-target construction to the target critic that is actually synchronized by `CriticLearner`;
3. implement the paper-stated fixed initial positions;
4. redirect temp, cache, checkpoints, TensorBoard, and logs to `D:\AgentData\RAC-MIPP\S1-R1`;
5. cache/vectorize the episode-independent spectral amplitude used by the synthetic-map generator only after exact-array parity checks against the original generator;
6. disable autograd anomaly tracing for the long run; this changes diagnostics, not gradients or updates.

None changes the COMA loss, actor/critic architecture, reward, action space, observation channels, map update, sensor model, or evaluator definition.

## Evaluation protocol and leakage barrier

- The final checkpoint is evaluated exactly once after training on seeds `20001` through `20050`, in ascending order.
- Evaluation uses greedy argmax actions, four agents, fixed paper positions, 25 m communication, zero communication failures, and 15 measurements per UAV.
- Each trial records seed, final weighted target-region entropy, positive-class F1, mission return, absolute return, path length, communication events when observable, and episode steps.
- Entropy and F1 follow the upstream `COMATest` definitions. The final global belief fuses all four measurements at every timestep. Entropy is mean binary Shannon entropy over positive ground-truth cells. F1 is positive-class F1 after thresholding belief at 0.5.
- Test seeds are not used during checkpoint selection, training stopping, or protocol revision. No intermediate checkpoint is evaluated on the Table I test set.
- Trial 1 supplies the qualitative ground-truth, final-belief, and four-UAV trajectory figure; it is a real formal rollout, not a schematic.

## Acceptance criterion

Let each standardized gap be the absolute reproduced-mean minus paper-mean gap divided by the corresponding paper-reported standard deviation.

- `STRONG_PARITY`: entropy gap at most 1.0 and F1 gap at most 1.0.
- `ACCEPTABLE_PARITY`: both gaps at most 2.0 and no obvious qualitative contradiction.
- `MAJOR_GAP`: either primary gap exceeds 2.0.

The sample standard deviation (`ddof=1`) across 50 trials is reported for the reproduction. The paper-reported standard deviations are used only for standardized-gap gates.

## Failure and revision policy

Execution failures may be repaired without changing this scientific contract. If a semantic protocol change is required, revision 1 and all completed evidence remain immutable; a separately preregistered `PROTOCOL_REVISION` is required. A `MAJOR_GAP` triggers a bounded root-cause audit, not unregistered hyperparameter search, reward/evaluator edits, best-seed selection, or test-driven checkpoint selection.
