# RAC-MIPP — PROJECT_PROGRESS

> **文件角色：项目唯一总控纲领 / Single Source of Truth**
>
> 本文件用于管理 RAC-MIPP 从空仓库到可投稿论文的全过程。任何新对话中的“总控 AI”在分派任务前，必须优先读取本文件与最新 Git 历史；任何低级执行 AI 不得自行改变研究目标、阶段门槛或实验口径。
>
> **创建日期：2026-08-08**
>
> **项目状态：ACTIVE / S2_IN_PROGRESS**

---

## 0. 项目标识

```text
PROJECT_CODE=RAC-MIPP
PROJECT_FULL_NAME=Risk-Aware Communication-Constrained Multi-UAV Informative Path Planning
LOCAL_PATH=D:\Desktop\my_project\RAC-MIPP
REMOTE_REPO=https://github.com/canimiliya/RAC-MIPP.git
REMOTE_FULL_NAME=canimiliya/RAC-MIPP
DEFAULT_BRANCH=main

TARGET_OUTPUT=中科院二区及以上目标的小论文（纯仿真）
PAPER_MODE=SIMULATION_ONLY
PROJECT_PRIORITY=FAST_PUBLISHABLE_RESEARCH
PROJECT_STATUS=ACTIVE
CURRENT_STAGE=S2
CURRENT_STAGE_STATUS=IN_PROGRESS
FORMAL_PROGRESS=2/9≈22%
CURRENT_BLOCKER=NONE
LAST_CLOSED_TASK=S1-R1-ORIGINAL-COMA-FORMAL-REPRODUCTION-R1
LAST_CLOSED_TASK_END_HEAD=aaf3f24e3a1c073d958b3056f338d6206dadca14
```

### 0.1 研究依据

本项目由 2026-08-08 的 UAV 路径/任务规划 + 强化学习调研收敛而来。调研给出的“快速产出”首选路线为：

- 以 IROS 2023 的 `dmar-bonn/ipp-marl` 为公开研究底座；
- 研究多 UAV informative path planning；
- 将通信从简单“有/无通信”升级为不可靠通信；
- 引入 recurrent / graph-based MARL；
- 将平均性能目标升级为风险敏感 / tail-risk 目标；
- 做通信失效、队伍规模等 OOD 泛化；
- 保持纯仿真，不从零搭建飞控或视觉系统。

本项目不承诺某个期刊一定录用，也不把 JCR 分区等同于中科院分区。真正投稿前必须用所在单位当年最新版中科院分区表复核。

---

# 1. 最终论文定位

## 1.1 暂定英文题目

**Risk-Aware Multi-UAV Informative Path Planning under Uncertain Communication via Recurrent Graph-MAPPO**

备选：

**RAC-MIPP: Risk-Aware Cooperative Informative Path Planning for Multi-UAV Systems under Unreliable Communication**

最终标题在 S7 结束后根据实际方法与结果冻结，不允许在实验尚未形成前为了“包装”而夸大。

## 1.2 中文题目

**面向不可靠通信的风险感知多无人机信息路径规划：基于循环图多智能体强化学习的方法**

## 1.3 核心研究问题

不是：

> “给 PPO/COMA 加一个注意力模块，看看奖励能不能更高。”

而是：

> **在多 UAV 信息路径规划中，当地图认知存在不确定性、无人机之间存在丢包/延迟/有限通信范围等通信不确定性时，如何让去中心化策略不仅追求平均信息采集效率，还能降低低概率但高损失的尾部失败风险，并在未见通信条件和未见 UAV 数量下保持可扩展泛化？**

论文必须围绕下面三个变量的耦合展开：

```text
ENVIRONMENT / MAP UNCERTAINTY
        +
COMMUNICATION UNCERTAINTY
        +
RISK-SENSITIVE COOPERATIVE PLANNING
```

## 1.4 论文必须回答的三个主问题

### RQ1 — 平均性能

在相同训练预算下，RAC-MIPP 是否能在不可靠通信环境中优于原始 COMA、IPPO、MAPPO、RMAPPO 等基线的平均任务表现？

### RQ2 — 尾部风险

在高丢包、高延迟、局部通信断裂等困难条件下，RAC-MIPP 是否能降低 worst-case / worst-10% performance degradation，而不是只提高 mean return？

### RQ3 — OOD 泛化

训练时只见到较小队伍和有限通信扰动时，RAC-MIPP 是否能泛化到：

- 更大的 UAV team size；
- 更严重的 packet drop；
- 未见的 delay；
- 不同通信半径；
- 必要时更强的 sensor noise / map uncertainty？

---

# 2. 研究边界冻结

为了“尽快形成一篇可投稿的小论文”，第一版项目边界必须严格控制。

## 2.1 第一版必须做

- 多 UAV informative path planning；
- 基于公开 `ipp-marl` 研究环境；
- 原始方法真实复现；
- 至少一个现代 PPO 系 MARL 基线；
- 通信丢包；
- 通信延迟；
- recurrent memory；
- graph / neighborhood-based coordination；
- 风险敏感目标（CVaR / distributional / tail-risk 方向之一，S5 冻结具体实现）；
- OOD team-size；
- OOD communication；
- 至少 5 个独立随机种子，目标 8–10；
- 完整 ablation；
- 推理开销和通信开销；
- 可复现实验脚本。

## 2.2 第一版明确不做

除非 S0–S7 已完成且总控明确开启扩展阶段，否则禁止加入：

- 真机；
- AirSim / ROS / Gazebo 全栈重构；
- 自研飞控；
- 图像感知；
- SLAM；
- 端到端视觉导航；
- Meta-RL；
- Diffusion Policy；
- LLM；
- Transformer 大模型；
- 多任务分配；
- 追逃任务；
- 机械臂；
- continuous 3D dynamics 平台迁移；
- 同时更换任务、平台和算法。

原因：这些内容会扩大问题维度，破坏“快速形成可投稿结果”的目标。

## 2.3 可选扩展

只有当 S7 已经获得稳定结论后，才允许评估：

```text
EXT-A: 更真实 UAV kinematic / energy constraints
EXT-B: continuous 3D cross-platform validation
EXT-C: dynamic information source
EXT-D: stronger bandwidth/AoI communication model
```

扩展不能阻止第一版论文冻结。

---

# 3. 上游研究底座与知识产权边界

## 3.1 上游仓库

```text
UPSTREAM_NAME=ipp-marl
UPSTREAM_REPO=https://github.com/dmar-bonn/ipp-marl.git
UPSTREAM_DEFAULT_BRANCH=master
UPSTREAM_PINNED_COMMIT=1e9bdc3ba90f707ce79797468f533f5733c65e4b
UPSTREAM_COMMIT_OBSERVED_DATE=2026-08-08
```

该 commit 是总控在 2026-08-08 通过 GitHub 看到的 `master` 最新提交。

## 3.2 上游代码使用规则

截至 2026-08-08 的仓库根目录审计未看到独立 `LICENSE` 文件。因此在明确授权条款前：

1. **不得把上游源代码整仓复制进 RAC-MIPP 后再公开发布；**
2. 上游代码优先作为本地依赖放入 `.deps/ipp-marl/`；
3. `.deps/` 必须进入 `.gitignore`；
4. 所有实验必须记录 upstream URL + commit SHA；
5. RAC-MIPP 仓库优先保存：
   - 我们自己的 adapter；
   - channel model；
   - algorithm implementation；
   - configs；
   - evaluation；
   - tests；
   - docs；
   - reproducibility scripts；
6. 若后续确认许可允许再分发，再由总控单独决定是否 vendor / fork。

本段是项目工程与发布策略，不代替正式法律意见。

---

# 4. 方法假设与暂定架构

## 4.1 Agent 侧信息流

暂定结构：

```text
local observation
      |
      +--> map / belief features
      |
      +--> received neighbor messages
                  |
            graph aggregation
                  |
                 GRU
                  |
                Actor
                  |
              action
```

Actor 在执行期只使用可获得的局部信息。

## 4.2 Centralized critic

训练期允许 centralized critic 使用全局训练信息，但执行期禁止依赖全局状态。

风险 critic 的具体实现允许在 S5 从以下方案中择一：

```text
A. Quantile / distributional critic + CVaR objective
B. Ensemble critic + downside-risk objective
C. Explicit CVaR estimator
```

不允许同时堆多个风险模块只为了增加“创新点”。

## 4.3 暂定目标函数

方法层面的研究方向为：

```text
Expected mission utility
- lambda_r * tail risk
- lambda_c * communication cost
- lambda_e * motion / energy proxy
```

最终损失函数、符号与系数在 S5 方法冻结时确定。

## 4.4 三个核心机制

第一版论文最多保留三个主要机制：

1. **Recurrent Graph Coordination**
   - 局部邻接；
   - 处理可变 team size；
   - GRU 记忆通信缺失/延迟历史。

2. **Risk-Sensitive Critic / Objective**
   - 优化 tail robustness；
   - 不能只换 reward shaping。

3. **Communication Uncertainty Curriculum**
   - 训练时逐步增加丢包/延迟；
   - 测试时做更困难 OOD。

如果最终实验表明其中某个机制无稳定贡献，应删除，而不是强行保留。

---

# 5. 实验合同

## 5.1 Training / IID 初始协议

初始候选值：

```text
TRAIN_TEAM_SIZES={2,4,6}
IID_PACKET_DROP={0.0,0.1,0.3}
IID_DELAY_STEPS={0,1,3}
GAMMA=0.99
GAE_LAMBDA=0.95
PPO_CLIP=0.2
ACTOR_LR=3e-4
CRITIC_LR=3e-4
GRU_HIDDEN=128
GAT_LAYERS=2
GAT_HEADS=4
```

这些是工程起点，不是论文最终超参数。

## 5.2 OOD 协议

目标测试条件：

```text
OOD_TEAM_SIZES={8,12}
OOD_PACKET_DROP={0.5}
OOD_DELAY_STEPS={5}
```

根据上游环境支持情况，可加入：

```text
OOD_COMM_RADIUS
OOD_SENSOR_NOISE
OOD_MAP_DISTRIBUTION
```

但每新增一个 OOD 维度必须有明确论文动机。

## 5.3 Baselines

第一版最低基线矩阵：

```text
B0  Greedy / heuristic informative planner
B1  Original ipp-marl / COMA
B2  IPPO
B3  MAPPO
B4  Recurrent MAPPO
B5  Graph-MAPPO without risk
M0  RAC-MIPP (full)
```

如实现和算力允许，可补：

```text
B6  QMIX / discrete-value baseline
```

禁止只与 DQN/DDPG 等弱基线比较后宣称 SOTA。

## 5.4 核心指标

### Mission / Mapping

- information gain；
- entropy reduction；
- mapping RMSE / reconstruction error；
- mission success / completion。

### Efficiency

- total path length；
- steps / mission time proxy；
- energy proxy；
- sample efficiency。

### Communication

- messages sent；
- bytes / normalized communication load；
- effective neighbor degree；
- packet delivery ratio。

### Risk

- mean return；
- median；
- worst-10%；
- CVaR；
- failure probability；
- severe degradation rate。

### Generalization

- unseen team-size success；
- unseen communication success；
- generalization gap。

### Computation

- policy parameter count；
- inference latency；
- GPU memory（若可靠可测）；
- training wall-clock（若可靠可测）。

## 5.5 随机种子与统计要求

```text
MIN_SEEDS=5
TARGET_SEEDS=8-10
REPORT_MEAN_STD=true
REPORT_95CI=true
```

主结论至少要有：

- 95% CI；
- 至少一种适当的显著性检验或 bootstrap；
- effect size 或等价的实际提升量；
- 不允许只挑最好 seed。

## 5.6 Ablation 最低要求

必须至少覆盖：

```text
A0 FULL RAC-MIPP
A1 w/o Graph aggregation
A2 w/o GRU
A3 w/o Risk objective
A4 w/o Communication curriculum
A5 fixed team-size training
```

视最终方法增加：

```text
A6 risk alpha sensitivity
A7 communication budget sensitivity
A8 neighborhood size / communication radius
```

---

# 6. 证据与可复现性制度

任何“PASS”都不能只凭文字描述。

## 6.1 必须保留

优先提交 Git 的轻量产物：

```text
configs/
docs/
scripts/
src/
tests/
artifacts/**/summary.json
artifacts/**/metrics.csv
artifacts/**/manifest.json
artifacts/**/small_plots/
```

## 6.2 默认不得提交

```text
.deps/
checkpoints/
wandb/
tensorboard/
videos/raw/
large_npz/
large_pkl/
datasets/generated/
cache/
```

大文件只能保存摘要、hash、manifest 和生成命令。

## 6.3 每个训练实验最低元数据

```text
RUN_ID
TASK_ID
GIT_HEAD
UPSTREAM_COMMIT
CONFIG_HASH
SEED
ALGORITHM
ENVIRONMENT
TEAM_SIZE
COMM_DROP
COMM_DELAY
START_TIME
END_TIME
STATUS
PRIMARY_METRICS
ARTIFACT_PATHS
```

## 6.4 不允许的科研行为

- 不允许修改评测代码后继续沿用旧结果；
- 不允许因为结果差偷偷换 seed；
- 不允许先看 test 再调训练参数；
- 不允许删除失败实验而不记录；
- 不允许更改 baseline 超参预算以故意削弱 baseline；
- 不允许将 smoke test 结果当正式论文结果；
- 不允许把仿真结果描述成真机能力；
- 不允许把“未验证”写成“已证明”。

---

# 7. Git 与任务管理制度

## 7.1 分支命名

```text
bootstrap/s0-*
repro/s1-*
benchmark/s2-*
env/s3-*
baseline/s4-*
method/s5-*
exp/s6-*
analysis/s7-*
paper/s8-*
fix/*
```

## 7.2 Task ID 格式

```text
S{stage}-R{round}-{SHORT-DESCRIPTION}-R{revision}
```

示例：

```text
S1-R0-UPSTREAM-COMA-REPRODUCTION-R1
S3-R1-PACKET-DROP-DELAY-CHANNEL-R1
S5-R0-RISK-CRITIC-CVAR-R1
```

## 7.3 任务结束强制回报格式

低级执行 AI 每次必须返回：

```text
TASK_ID=
STATUS=COMPLETED|BLOCKED|PARTIAL
FINAL_LABEL=

START_HEAD=
END_HEAD=
REMOTE_BRANCH_HEAD=
BRANCH=

GOAL=
IMPLEMENTED=
TESTS=
RESULTS=
ARTIFACTS=
REGRESSIONS=
UNRESOLVED=

FORMAL_PROGRESS=
CURRENT_BLOCKER=
UNIQUE_NEXT_TASK=
```

缺少 `START_HEAD / END_HEAD / TESTS / ARTIFACTS` 的“完成”不视为正式完成。

## 7.4 合并规则

- 任务分支先完成；
- 测试 PASS；
- 产物可复核；
- 总控审计；
- 才能合并到 `main`；
- 不允许低级 AI 自行宣布阶段关闭；
- 不允许跨阶段提前实现“顺手的功能”。

---

# 8. 阶段路线图

项目正式划分为 S0–S8。

---

## S0 — Project Bootstrap & Research Contract

**状态：CLOSED**

目标：把空项目变成可复现、可审计、不会乱改方向的研究仓库。

### 必须完成

- 本地目录与远程 `main` 对齐；
- 上游仓库 provenance 冻结；
- 上游依赖本地化但不误提交；
- `.gitignore`；
- 环境信息记录；
- 项目目录骨架；
- 最小 pytest / import smoke；
- README 指向本文件；
- 不运行长训练。

### S0 PASS 判据

```text
LOCAL_REPO_SYNCED=true
ORIGIN_CORRECT=true
UPSTREAM_COMMIT_PINNED=true
UPSTREAM_LICENSE_AUDITED=true
DEPS_IGNORED=true
ENVIRONMENT_RECORDED=true
REPO_SKELETON_READY=true
SMOKE_TEST_PASS=true
LONG_TRAINING_STARTED=false
```

---

## S1 — Original ipp-marl Reproduction

**状态：CLOSED**

`S1-R1-ORIGINAL-COMA-FORMAL-REPRODUCTION-R1` 已通过总控审计并安全 fast-forward 集成到 main，原版 ipp-marl / COMA 正式长训练、50-trial Table I 评测与真实 synthetic rollout 均保留为 S2 parity anchor。

目标：先复现真实原方法，再做改进。

### 必须完成

- 阅读上游 README 和原论文关键实验定义；
- 冻结 original experiment contract；
- 跑通原始训练/评估；
- 至少复现一个关键定量结果；
- 至少复现一个行为/地图结果；
- 记录与论文差异；
- 不修改算法后声称“原版结果”。

### S1 PASS 判据

```text
UPSTREAM_ENV_RUNS=true
ORIGINAL_TRAINING_RUNS=true
ORIGINAL_EVAL_RUNS=true
ORIGINAL_METRIC_CONTRACT_FROZEN=true
REPRO_RESULT_AUDITED=true
UNEXPLAINED_MAJOR_GAP=false
```

如果论文数字不能精确复现，应记录差异并判断是否“功能复现可接受”，不得造假对齐。

---

## S2 — Benchmark & Evaluation Freeze

**状态：IN_PROGRESS**

目标：在任何新算法之前冻结公平评测接口。

### 必须完成

- seed contract；
- train/val/test 或 train/IID/OOD split；
- metrics；
- eval script；
- logging schema；
- baseline compute budget；
- communication metric hooks；
- deterministic smoke tests。

### S2 PASS 判据

```text
SEED_CONTRACT=true
EVAL_PIPELINE=true
METRIC_PARITY=true
IID_OOD_SPLIT_FROZEN=true
BASELINE_BUDGET_FROZEN=true
LOG_SCHEMA_FROZEN=true
```

---

## S3 — Uncertain Communication Environment

**状态：FROZEN**

目标：把“通信有/无”升级为参数化通信不确定性。

### 第一版最低能力

- packet drop；
- message delay；
- communication radius / neighbor mask；
- reproducible channel RNG；
- channel statistics logging。

### 可选

- bandwidth budget；
- stale information age；
- sensor noise coupling。

### S3 PASS 判据

```text
DROP_MODEL=true
DELAY_MODEL=true
NEIGHBOR_MASK=true
CHANNEL_SEEDED=true
CHANNEL_METRICS=true
NO_FUTURE_INFORMATION_LEAKAGE=true
DETERMINISM_SMOKE_PASS=true
ORIGINAL_ZERO_NOISE_PARITY=true
```

最后一项极重要：通信扰动关闭时，环境行为必须与原版兼容或解释差异。

---

## S4 — Strong Modern Baselines

**状态：FROZEN**

目标：建立审稿人认可的现代 MARL 对照。

最低完成：

- IPPO；
- MAPPO；
- recurrent MAPPO；
- Graph-MAPPO without risk；
- original COMA；
- heuristic。

### S4 PASS 判据

```text
ALL_REQUIRED_BASELINES_RUN=true
COMMON_EVAL=true
COMMON_BUDGET=true
BASELINE_SANITY_PASS=true
NO_INTENTIONAL_BASELINE_WEAKENING=true
```

---

## S5 — RAC-MIPP Core Method

**状态：FROZEN**

目标：实现并冻结论文方法。

### 必须决策

- graph aggregation 形式；
- recurrent state；
- risk estimator；
- CVaR / downside objective；
- communication curriculum；
- final actor/critic inputs；
- final loss；
- inference information boundary。

### S5 PASS 判据

```text
METHOD_CONTRACT_FROZEN=true
GRAPH_MODULE=true
RECURRENT_MODULE=true
RISK_MODULE=true
CURRICULUM=true
DECENTRALIZED_EXECUTION=true
UNIT_TESTS=true
SMOKE_TRAINING=true
NO_TEST_LEAKAGE=true
```

S5 结束后除 bug 修复外，不允许继续“加模块”。

---

## S6 — Main Experiments

**状态：FROZEN**

目标：完成论文主表的 IID 对比。

最低要求：

- 所有主要 baseline；
- full RAC-MIPP；
- 相同正式 seeds；
- 相同训练预算；
- 主要通信条件；
- 指标全部由统一 evaluator 生成。

### S6 PASS 判据

```text
MAIN_TABLE_COMPLETE=true
MIN_SEEDS_MET=true
ALL_RUNS_TRACEABLE=true
NO_MISSING_PRIMARY_METRICS=true
COMPUTE_BUDGET_AUDITED=true
```

---

## S7 — OOD, Ablation, Statistics

**状态：FROZEN**

目标：决定这是不是一篇真正能投稿的论文。

必须完成：

- unseen team size；
- severe packet loss；
- unseen delay；
- full ablation；
- risk sensitivity；
- statistics；
- runtime；
- failure case；
- qualitative visualization。

### S7 PASS 判据

```text
OOD_TEAM_SIZE=true
OOD_COMMUNICATION=true
ABLATION_COMPLETE=true
RISK_ANALYSIS=true
STATISTICS_COMPLETE=true
RUNTIME_ANALYSIS=true
FAILURE_CASE_ANALYSIS=true
PLOTS_READY=true
```

### S7 决策门

只有在这里判断论文强度：

```text
D1 STRONG:
    进入 S8，按二区以上目标包装投稿。

D2 MODERATE:
    补 1 个有科学动机的实验，不加新算法堆砌。

D3 WEAK:
    复盘假设；允许回退到 S5-Rx。
```

禁止在结果弱时直接虚构“一区创新”。

---

## S8 — Paper, Reproducibility & Freeze

**状态：FROZEN**

目标：把研究成果变成可投稿、可公开、可复现的完整包。

必须完成：

- final tables；
- figures；
- abstract；
- introduction；
- related work；
- method；
- experiments；
- limitations；
- reproducibility README；
- clean code；
- configs；
- selected lightweight result artifacts；
- release tag；
- final project report。

### S8 PASS 判据

```text
PAPER_DRAFT_COMPLETE=true
ALL_CLAIMS_TRACEABLE=true
FIGURE_SOURCE_TRACEABLE=true
PUBLIC_REPRO_GUIDE=true
REPO_CLEAN=true
RELEASE_TAGGED=true
PROJECT_FROZEN=true
```

---

# 9. 投稿策略

第一版 RAC-MIPP 的投稿逻辑是：

## 9.1 现实目标

更匹配纯仿真、AI 方法 + 工程问题的候选：

- Engineering Applications of Artificial Intelligence；
- Applied Soft Computing；
- Neural Computing and Applications。

是否满足“中科院二区及以上”必须在投稿当时用所在单位最新版分区表重新核验。

## 9.2 冲高条件

若 S7 同时满足：

- 明显 OOD 泛化；
- tail-risk 优势稳定；
- 强 baseline；
- 统计充分；
- 通信开销合理；
- 方法不是简单模块堆叠；
- UAV 任务建模足够扎实；

则可以进一步评估 Chinese Journal of Aeronautics 等更高目标。

## 9.3 不作为第一版默认目标

IEEE RA-L 不设为“快速保底”。纯仿真稿若缺少更强动力学/系统性证据，风险较高。

---

# 10. 项目目录建议

S0 完成后期望至少形成：

```text
RAC-MIPP/
├─ PROJECT_PROGRESS.md
├─ README.md
├─ .gitignore
├─ pyproject.toml / requirements*.txt
├─ configs/
│  ├─ baseline/
│  ├─ communication/
│  └─ rac_mipp/
├─ src/
│  └─ rac_mipp/
│     ├─ communication/
│     ├─ algorithms/
│     ├─ models/
│     ├─ evaluation/
│     └─ utils/
├─ scripts/
│  ├─ reproduce_upstream.*
│  ├─ train.*
│  └─ evaluate.*
├─ tests/
├─ docs/
│  ├─ UPSTREAM_PROVENANCE.md
│  ├─ ENVIRONMENT.md
│  ├─ EXPERIMENT_CONTRACT.md
│  └─ PAPER_NOTES.md
├─ artifacts/
│  ├─ s0/
│  ├─ s1/
│  └─ ...
└─ .deps/                  # LOCAL ONLY, GITIGNORED
   └─ ipp-marl/
```

目录可在 S0 根据上游实际结构轻微调整，但不允许随意演化成复杂 monorepo。

---

# 11. 总控 AI 职责

总控 AI 必须：

1. 维护本文件；
2. 每次只给低级 AI 一个唯一主任务；
3. 审核任务证据，不凭口头“已完成”关闭阶段；
4. 防止过早长训练；
5. 防止 scope creep；
6. 审查 baseline 是否公平；
7. 审查 train/test leakage；
8. 审查结果是否可复现；
9. 在任何大规模训练前冻结实验合同；
10. 维护 `FORMAL_PROGRESS / BLOCKER / UNIQUE_NEXT_TASK`；
11. 必要时暂停并回退，而不是让错误继续积累；
12. 不因“想发二区”而夸大结果。

---

# 12. 低级执行 AI 职责

低级执行 AI：

- 只执行当前 Task Card；
- 不自行规划下一阶段；
- 不主动扩大算法；
- 不修改本文件的研究目标；
- 不私自 merge；
- 不启动未授权的长训练；
- 遇到失败必须报告真实日志；
- 修改代码必须有最小测试；
- 每个实验必须可追溯到 Git commit + config + seed；
- 大文件不得误提交。

如果发现任务卡本身存在错误：

```text
STOP
REPORT_CONFLICT
PROPOSE_MINIMAL_CORRECTION
WAIT_FOR_CONTROLLER_DECISION
```

---

# 13. 当前正式状态

```text
DATE=2026-08-09

PROJECT=RAC-MIPP
PROJECT_STATUS=ACTIVE

LOCAL_PATH=D:\Desktop\my_project\RAC-MIPP
REMOTE_REPO=https://github.com/canimiliya/RAC-MIPP.git
DEFAULT_BRANCH=main

S0=CLOSED
S1=CLOSED
S2=IN_PROGRESS
S3=FROZEN
S4=FROZEN
S5=FROZEN
S6=FROZEN
S7=FROZEN
S8=FROZEN

FORMAL_PROGRESS=2/9≈22%
CURRENT_BLOCKER=NONE

LAST_CLOSED_TASK=S1-R1-ORIGINAL-COMA-FORMAL-REPRODUCTION-R1
LAST_CLOSED_TASK_END_HEAD=aaf3f24e3a1c073d958b3056f338d6206dadca14
```

---

# 14. UNIQUE NEXT TASK

```text
UNIQUE_NEXT_TASK=WAIT_FOR_CONTROLLER_DECISION
```

`S2-R0-BENCHMARK-EVALUATION-CONTRACT-FREEZE-R1` 已在任务分支完成统一 benchmark/evaluation contract、50-trial COMA parity 与 determinism smoke，当前等待总控审计。S2 尚未由总控正式关闭；S3-S8 与所有 RAC-MIPP 新算法工作仍冻结。

以下 S0-R0 Task Goal 保留为历史任务卡，不再是当前执行授权。

## Task Goal

在**不进行算法改进、不启动训练**的前提下，把本地空目录与远程 RAC-MIPP 建立成可复现研究仓库，并冻结上游 `ipp-marl` 的 provenance 与本地依赖方式。

## 执行要求

1. 在：

```text
D:\Desktop\my_project\RAC-MIPP
```

建立/同步 Git 仓库。

2. `origin` 必须为：

```text
https://github.com/canimiliya/RAC-MIPP.git
```

3. 从远程 `main` 最新 HEAD 创建任务分支：

```text
bootstrap/s0-r0
```

4. 创建 `.gitignore`，至少忽略：

```text
.deps/
__pycache__/
.pytest_cache/
.venv/
env/
*.pyc
wandb/
runs/
checkpoints/
*.pt
*.pth
*.ckpt
```

5. 在本地：

```text
.deps/ipp-marl
```

克隆上游：

```text
https://github.com/dmar-bonn/ipp-marl.git
```

并 checkout：

```text
1e9bdc3ba90f707ce79797468f533f5733c65e4b
```

6. 不提交 `.deps/ipp-marl`。

7. 创建：

```text
docs/UPSTREAM_PROVENANCE.md
```

至少记录：

```text
repo
branch
commit
clone date
local dependency path
license audit observation
redistribution policy
```

8. 创建：

```text
docs/ENVIRONMENT.md
```

记录当前：

```text
OS
CPU
GPU
RAM
Python
Conda
CUDA
NVIDIA driver
Git
```

未知项写 `UNKNOWN`，不猜。

9. 创建最小项目骨架：

```text
src/
tests/
configs/
scripts/
artifacts/s0/
```

10. 创建一个不依赖长训练的 smoke test，证明项目 Python/test 基础可执行。

11. 生成：

```text
artifacts/s0/bootstrap_summary.json
```

至少记录：

```text
task_id
start_head
end_head
origin
upstream_repo
upstream_commit
dependency_ignored
smoke_test
long_training_started=false
```

12. 提交并推送任务分支。

## 禁止事项

```text
NO_ALGORITHM_CHANGE
NO_LONG_TRAINING
NO_UPSTREAM_SOURCE_COPY_TO_PUBLIC_REPO
NO_S1_WORK
NO_PR_MERGE
```

## PASS Gate

```text
ORIGIN_CORRECT=true
BRANCH=bootstrap/s0-r0
UPSTREAM_REPO_CORRECT=true
UPSTREAM_COMMIT_CORRECT=true
DEPS_IGNORED=true
UPSTREAM_PROVENANCE_RECORDED=true
ENVIRONMENT_RECORDED=true
PROJECT_SKELETON_READY=true
SMOKE_TEST_PASS=true
LONG_TRAINING_STARTED=false
REMOTE_BRANCH_PUSHED=true
```

## 预期最终标签

```text
PASS_S0_R0_REPO_BOOTSTRAP_UPSTREAM_PROVENANCE
```

S0-R0 完成后，**不要自行开始 S0-R1 或 S1**。将完整结果交回总控 AI 审计，由总控决定唯一下一任务。

---

# 15. 总控接管提示词

新对话可直接给总控 AI：

> 你现在是 RAC-MIPP 项目的高级总控 AI。先完整读取 `PROJECT_PROGRESS.md`，再检查 GitHub `canimiliya/RAC-MIPP` 当前状态与最新任务结果。项目目标是尽快形成一篇中科院二区及以上目标的纯仿真多无人机强化学习规划论文。你不能跳阶段、不能擅自长训练、不能仅根据执行 AI 的口头说明判定 PASS。每轮只输出一个唯一下一任务，并维护项目正式进度、blocker、Git HEAD 与证据链。

---

# 16. 冻结原则

RAC-MIPP 的科研顺序固定为：

```text
真实上游复现
    ↓
评测合同冻结
    ↓
通信不确定性
    ↓
现代强 baseline
    ↓
RAC-MIPP 方法
    ↓
主实验
    ↓
OOD + Ablation + Statistics
    ↓
论文冻结
```

**任何时候都不允许把顺序改成：**

```text
先写“新算法”
→ 跑一个漂亮 reward 曲线
→ 再找 baseline
→ 再补实验
→ 最后猜论文故事
```

项目最终目标不是“代码很多”，而是形成一条审稿人能够检查、复现实验能够支撑、所有论文 claim 都能回到证据的研究链。

---

**END OF PROJECT CONTROL FILE**
