# Environment Record

Recorded on 2026-08-08 during `S0-R0-REPO-BOOTSTRAP-UPSTREAM-PROVENANCE-R1`.
Values below are observations from the listed commands; unavailable values are
recorded as `UNKNOWN`.

| Item | Observed value | Detection command |
|---|---|---|
| OS | Microsoft Windows 11 专业中文版, 10.0.26100, build 26100, 64-bit | `Get-CimInstance Win32_OperatingSystem` |
| CPU | Intel(R) Core(TM) Ultra 7 270K Plus; 24 cores / 24 logical processors | `Get-CimInstance Win32_Processor` |
| GPU | NVIDIA GeForce RTX 5060 Ti | `Get-CimInstance Win32_VideoController`, `nvidia-smi` |
| RAM | 50,873,458,688 bytes (approximately 47.4 GiB) | `Get-CimInstance Win32_ComputerSystem` |
| Python | 3.13.9; `D:\anaconda\python.exe` | `python --version`, `python -c "import sys; print(sys.executable)"` |
| Conda | 26.1.1; prefix `D:\anaconda` | `conda --version`, `CONDA_PREFIX` |
| CUDA toolkit (`nvcc`) | UNKNOWN; `nvcc` was not found on PATH | `nvcc --version` |
| NVIDIA driver | 581.29 | `nvidia-smi` |
| CUDA runtime reported by driver | 13.0 | `nvidia-smi` |
| Git | 2.55.0.windows.2 | `git --version` |
| pytest | 8.4.2 | `pytest --version` |

No CUDA toolkit version is inferred from the driver-reported runtime field.
