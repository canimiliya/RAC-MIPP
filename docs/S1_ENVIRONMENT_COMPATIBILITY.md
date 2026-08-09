# S1 Environment and Compatibility Record

## Outcome

An isolated Windows environment is executable on the RTX 5060 Ti. CUDA tensor kernels, backward propagation, the upstream environment, a bounded COMA optimizer update, checkpoint write/load, and an evaluation episode all passed. This is engineering smoke evidence only, not a reproduced paper result.

## Canonical upstream requirements

The upstream README requests `matplotlib 3.5.1`, `numpy 1.22.2`, `opencv-python 4.5.5.62`, `scipy 1.8.1`, and `torch 1.13.0+cu117`. It omits several imported packages. This 2022-era stack predates the current GPU.

## Actual reproduction environment

- Prefix: `D:\AgentData\RAC-MIPP\S1-R0\conda`
- Python 3.11.15
- torch 2.11.0+cu130; bundled CUDA runtime 13.0
- NumPy 1.26.4; SciPy 1.13.1; Matplotlib 3.8.4; OpenCV 4.10.0
- PyYAML 6.0.2; seaborn 0.13.2; scikit-learn 1.5.2; TensorBoard 2.18.0
- GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB, compute capability 12.0
- Driver: 581.29

Conda and pip caches, `TEMP/TMP`, Torch cache, Matplotlib cache, checkpoints, and TensorBoard logs are directed to `D:\AgentData\RAC-MIPP\S1-R0`. The user's base Conda environment was not modified.

## Why different

The RTX 5060 Ti is a compute-capability 12.0 GPU. A CUDA 11.7-era PyTorch binary is not an appropriate native GPU target. The current environment therefore uses an official modern CUDA 13.0 PyTorch wheel and a conservative Python 3.11 scientific stack. The old versions remain the canonical provenance reference, not the executable GPU stack.

## Compatibility changes

1. Project-side adapter adds both the upstream root and `marl_framework` to `sys.path`, because imports mix package-qualified and top-level forms.
2. Runtime paths are redirected away from the hard-coded `/home/penguin2/...` upstream constants.
3. The adapter repairs `BatchMemory.get("mask")` from nonexistent `transition.masks` to the actual `transition.mask` field in-process.
4. The smoke config reduces budget to one (two loop time steps), transition batch to eight, batch count to one, and data passes to one. It does not change action, observation, reward, actor, critic, or COMA loss definitions.
5. Upstream files are not edited. Python bytecode generation is disabled for final verification so `.deps/ipp-marl` remains clean.

## Expected scientific impact

The import/path and mask-typo changes have no intended scientific effect. Modern PyTorch/CUDA and library versions can change floating-point kernels, RNG sequences, and optimization trajectories; formal reproduction must therefore record versions and compare distributions, not expect bitwise equality. The shortened smoke configuration has no scientific interpretation.

## Preserved recovery notes

- The first CUDA-wheel installation command reached the 10-minute command limit after caching the 1.9 GB wheel on D. A second progress-bar-free invocation reused that cache and completed successfully.
- The first full upstream smoke completed environment, train, checkpoint, and eval work but failed while serializing a NumPy `bool_` Gate. The adapter now casts Gate values to built-in `bool`; the failure remains recorded in the final summary's `attempt_history`.

## Reproduction commands

Create the isolated environment on D, install the official CUDA wheel and [requirements-modern.txt](../configs/s1/r0/requirements-modern.txt), then run:

```powershell
$env:TEMP='D:\AgentData\RAC-MIPP\S1-R0\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='D:\AgentData\RAC-MIPP\S1-R0\cache\pip'
$env:PYTHONDONTWRITEBYTECODE='1'
D:\AgentData\RAC-MIPP\S1-R0\conda\python.exe scripts\s1\run_upstream_smoke.py
```

The committed summary is `artifacts/s1/r0/repro_smoke_summary.json`. The checkpoint and TensorBoard event files are local-only under `D:\AgentData`.

## Boundaries

- `SMOKE_ONLY=true`
- `NOT_PAPER_RESULT=true`
- `LONG_TRAINING_STARTED=false`
- `FORMAL_REPRO_TRAINING_STARTED=false`
- No S2 work and no new RAC-MIPP algorithm work occurred.
