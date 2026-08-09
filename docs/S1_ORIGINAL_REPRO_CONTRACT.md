# S1 Original ipp-marl Reproduction Contract

This contract freezes what counts as reproducing the original method. Every item is labelled `PAPER_STATED`, `CODE_DERIVED`, `INFERRED`, or `UNKNOWN`; code defaults are not presented as paper claims.

## Paper provenance

- `P01`: Westheider, Rückin, and Popović, *Multi-UAV Adaptive Path Planning Using Deep Reinforcement Learning*, IROS 2023. Local PDF reviewed in full (8 pages).
- `P03`: Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018. Local PDF reviewed for the CTDE, counterfactual advantage, architecture, and training definitions.

## Code provenance

- Repository: `https://github.com/dmar-bonn/ipp-marl.git`
- Frozen commit: `1e9bdc3ba90f707ce79797468f533f5733c65e4b`
- Local path: `.deps/ipp-marl` (ignored and not redistributed)
- Upstream source remained unmodified during S1-R0.

## Environment definition

`PAPER_STATED`: a 50 m x 50 m flat terrain stores an initially unknown binary target variable. The posterior is a 10 cm occupancy-style grid. Synthetic regions of interest cover 30%-60% of terrain. Planning occurs on a 5 m grid at 5 m, 10 m, or 15 m altitude; altitude changes footprint and simulated sensor accuracy.

`CODE_DERIVED`: `params.yaml` implements the same dimensions, three altitude levels, an altitude-dependent noise model, and deterministic episode-indexed synthetic map generation. The current ground-truth generator uses axis-aligned splits selected from four cases.

## Agent/team setup

`PAPER_STATED`: training uses four homogeneous UAVs. The trained actor is evaluated without retraining with 2, 4, and 8 UAVs. The standard limited-communication case uses a 25 m radius.

## Observation

`PAPER_STATED`: the actor sees agent ID, remaining budget, an egocentric position/boundary map with communicated neighbour positions and altitudes, local belief, weighted belief entropy, current measurement entropy, and communicated footprint coverage. The critic receives those local features plus global positions, global belief/entropy, global footprint coverage, and the other agents' actions.

`CODE_DERIVED`: the actor tensor has seven spatial channels; the critic tensor adds five channels. Agent ID and budget are expanded as constant feature maps. A single CNN actor is parameter-shared across agents.

## Action

`PAPER_STATED`: six discrete fixed-step actions - up, north, east, south, west, down - with out-of-bounds and conflicting 2D positions masked. The upstream default also uses six actions.

## Reward

`PAPER_STATED`: all agents share a global normalized weighted entropy-reduction reward with affine scaling. The paper does not state numeric `alpha` and `beta`.

`CODE_DERIVED`: the optimized reward is `22 * relative_weighted_entropy_reduction - 0.5`; a second logged absolute reward uses `10 * absolute_reduction - 0.17`. These constants must not be described as paper-stated.

## Communication model

`PAPER_STATED`: a message contains UAV ID, position, and current measurement. It is received when inter-UAV distance is within radius `D`, then fused into the recipient's local belief. The paper evaluates 0 m, 25 m, and unlimited communication.

`CODE_DERIVED`: the upstream additionally has a `failure_rate` Bernoulli gate, defaulting to zero, and an optional sampled range mode. These are not part of the frozen P01 result contract.

## Original algorithm

`PAPER_STATED`: COMA uses centralized training and decentralized execution. Its critic estimates a joint-action Q value; the per-agent counterfactual baseline marginalizes only that agent's action while holding the other actions fixed. The actor uses only locally available information at execution. The critic is trained on-policy with TD(lambda).

P03 establishes the same information boundary and counterfactual advantage. P01 adapts it to image-based informative path planning with CNN feature maps; it does not inherit P03's StarCraft task or GRU architecture.

## Training budget and hyperparameters

`PAPER_STATED`:

- collect 3,000 on-policy environment interactions, then optimize for five epochs;
- batch size 600;
- Adam; actor LR `1e-5`, critic LR `1e-4`;
- `gamma=0.99`, `lambda=0.8`;
- epsilon decreases from 0.5 to 0.02 over 10,000 missions;
- target critic copied every 30,000 environment interactions.

`UNKNOWN`: P01 does not state a final total number of training interactions, wall-clock budget, seed count, or exact model-selection rule.

## Evaluation protocol

`PAPER_STATED`: 50 terrain-monitoring missions per experiment; 50 m x 50 m terrain; four agents and 25 m communication unless varied; 15 measurements per UAV; report mean and standard deviation of weighted map entropy and F1 score in ground-truth target regions.

## Primary quantitative reproduction target

The first formal target is P01 Table I, four agents with 25 m limited communication at 100% mission time: entropy `0.2842 +/- 0.0408`, F1 `0.7858 +/- 0.0308`, over 50 trials. This is selected because it uses the default synthetic domain and all required generation code is present.

## Qualitative reproduction target

`INFERRED`: generate a four-agent synthetic-terrain trajectory plus final belief and ground-truth map under 25 m communication. The real-world thermal-data example is not selected because that raster is absent from the upstream checkout.

## Paper-vs-code differences

1. The paper's collection batch is 3,000 interactions and optimization batch is 600. The checked-in default accumulates `60 x 5 = 300` transitions before an update.
2. The paper says target copy every 30,000 interactions. The config uses `copy_rate=10`; at the default 300-transition update cadence this corresponds to 3,000 interactions. In addition, `BatchMemory` receives `COMAWrapper.target_critic_network`, while `CriticLearner` updates a separate target copy.
3. The paper says initial UAV positions are fixed. The active code generates positions from `seed * episode * agent_id`; the explicitly fixed corner positions are commented out.
4. Paper batch `600` differs from code batch `60`; paper five epochs agrees with `data_passes=5`.
5. `params.yaml` sets `n_episodes=1500`, but the outer-loop count multiplies this value by the transition batch and divides by episode length/team size; it is not a direct mission count.
6. The paper leaves reward scaling coefficients unspecified; code fixes `22/-0.5` for the optimized relative reward and `10/-0.17` for an absolute log metric.
7. The upstream README lists only five core packages, while imports also require PyYAML, seaborn, scikit-learn, and TensorBoard.
8. Upstream constants hard-code a developer Linux path. The S1 adapter redirects runtime output without changing the algorithm.
9. `BatchMemory.get("mask")` refers to nonexistent `transition.masks`; the actual namedtuple field is `mask`. S1-R0 fixes this typo only in-process.
10. There is no standalone evaluation CLI, published checkpoint, real-world thermal raster, or exact published experiment config in the checkout. Evaluation exists only inside the training loop every 50 training steps.

## Unknowns

- Total original training budget and formal seed protocol.
- Exact paper reward scaling coefficients.
- Which checkpoint generated Table I.
- Real-world thermal dataset location.
- Whether the target-network split in code is intentional or an incomplete cleanup artifact.

## What counts as successful S1 reproduction

S1-R0 success means the real environment, action/observation/reward pipeline, one bounded forward/loss/backward/optimizer update, checkpoint write/load, and evaluation episode execute with finite evidence. It is not a paper result.

Full S1 success requires a later authorized formal run against the frozen 50-trial Table I target plus an auditable synthetic trajectory/map. Any unexplained gap remains a reproduction finding; metrics must not be changed to force agreement.
