# S3 Uncertain Communication Model Contract

Task: `S3-R0-UNCERTAIN-COMMUNICATION-ENVIRONMENT-AND-VALIDATION-R1`

Status: frozen for S3 validation. This is an environment contract, not a new MARL algorithm.

## Scientific basis and scope

- P08 Message-Dropout models another agent's message as one semantic block and never drops an agent's own observation. We adopt the message-level unit and protected self observation, but model physical execution-time packet loss rather than P08's training regularizer.
- P09 DACOM shows that delay is a network property that changes what information is available for a decision and represents delay in environment time units. Our discrete IPP simulator therefore uses integer environment steps and exposes message age. Unlike DACOM, S3 does not learn waiting time and does not delay actions.
- P13 Communication-Constrained Priors uses binary communication links and evaluates distance-based and lossy constraints. The UAV IPP channel composes a send-time distance link with seeded Bernoulli packet survival. S3 uses an IID channel; correlated/Markov fading remains future work and is not silently claimed.

Local evidence: `.papers/02_COMMUNICATION_GRAPH/P08_Kim_2019_Message_Dropout.pdf`, `.papers/02_COMMUNICATION_GRAPH/P09_Yuan_2022_DACOM_Delay-Aware_Communication.pdf`, and `.papers/04_RECENT_RELATED/P13_Yang_2025_NeurIPS_Communication-Constrained_MARL.pdf` (verified by the S0 literature manifest).

## Frozen event semantics

1. A `message attempt` is one directed, non-self sender/receiver pair at one environment step. Broadcast to three peers is three attempts.
2. Range eligibility is checked at send time using inclusive Euclidean distance `distance <= communication_radius` in metres.
3. Only range-eligible attempts reach the Bernoulli drop stage. A dropped packet is permanently removed.
4. `delay_steps` is an integer number of environment decision steps. `delay=3` means a packet submitted at step `t` becomes available while constructing the receiver observation at step `t+3`.
5. Enqueued payloads are deep-copied send-time snapshots. Delivery never rereads the sender's current state.
6. Queue order is `(delivery_step, send_step, sender_id, receiver_id, sequence)`. If multiple snapshots from one sender arrive together, the last item in this deterministic order is the dictionary-compatible value; the full event log retains all deliveries.
7. Every episode owns a new channel and queue. Pending messages are counted and discarded at episode end; `reset()` clears the queue, event log, sequence, and seeded RNG.
8. During disconnection, each agent keeps its own local belief and any previously fused knowledge. No synthetic zero message overwrites belief.
9. Self information is always present as the agent's local observation, but no self-message is attempted, dropped, delivered, or charged.
10. All agents publish their step-`t` snapshot before any receiver fuses deliveries. Agent iteration therefore cannot reveal a later within-step state.

## Randomness and pairing

- `ENVIRONMENT_SEED` continues to seed the frozen upstream map, initial state, sensor, and policy evaluation path.
- `CHANNEL_SEED` initializes a dedicated `numpy.random.Generator(PCG64)` owned only by `ChannelModel`.
- The COMA bridge consumes the exact legacy upstream global-NumPy draws at every receiver/sender check, independently of S3 configuration. This preserves the original zero-noise environment RNG trajectory while actual drop decisions use only `CHANNEL_SEED`.
- The validation rule is `CHANNEL_SEED = ENVIRONMENT_SEED + 300000`; paired diagnostics may hold `ENVIRONMENT_SEED` fixed and vary only `CHANNEL_SEED`.

## Metrics and interfaces

Real events produce `messages_attempted`, `messages_range_eligible`, `messages_dropped`, `messages_delayed`, `messages_delivered`, `packet_delivery_ratio`, `effective_neighbor_degree`, `message_age_mean`, `message_age_max`, `communication_radius`, and normalized load in `MESSAGE_UNITS`. Bytes are not reported because Python object serialization is not a scientifically stable radio payload definition.

`packet_delivery_ratio` is delivered packets divided by range-eligible attempts within the episode. With delay, packets still pending at the terminal boundary correctly reduce this finite-episode ratio and are separately reported as pending/discarded.

The reusable channel exposes delivered-message sets, delivery status through events, message age, sender/receiver IDs, and the radius-derived neighbor relation. Graph aggregation, recurrent memory, risk objectives, communication curriculum, and all S4/S5 algorithms are explicitly out of scope.

`ChannelModel.neighbor_mask(positions)` exposes the boolean directed send-time mask used by the actual channel; its diagonal is always false and the radius boundary is inclusive.

## Frozen validation conditions

- Zero noise: `drop=0`, `delay=0`, `radius=25 m`.
- IID characterization: drop `{0, 0.1, 0.3, 0.5}` at delay 0; delay `{0, 1, 3, 5}` at drop 0; joint `(0.3,3)` and `(0.5,5)`.
- Future OOD placeholders remain `drop=0.5`, `delay=5`; S3 characterization uses only frozen `VALIDATION` seeds and is not a final paper result.
