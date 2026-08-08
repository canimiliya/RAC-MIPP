# Upstream Provenance

This document records the local, pinned dependency used by the S0 bootstrap.
It does not authorize algorithm work, training, or later project stages.

```text
UPSTREAM_REPO=https://github.com/dmar-bonn/ipp-marl.git
UPSTREAM_BRANCH=master
UPSTREAM_COMMIT=1e9bdc3ba90f707ce79797468f533f5733c65e4b
LOCAL_DEPENDENCY_PATH=.deps/ipp-marl
CLONE_DATE=2026-08-08T20:43:33+08:00
```

## License audit observation

At the pinned checkout, the upstream repository root was inspected. The root
contained `README.md` and `marl_framework/`; no independent root-level
`LICENSE` file was found. This is an observation, not a license conclusion.

## Redistribution policy

The checkout under `.deps/ipp-marl` is a local ignored dependency and is not
part of the RAC-MIPP Git index. Until the upstream licensing terms are
verified from authoritative project information, RAC-MIPP will not publicly
redistribute copied upstream source. Any future use must preserve this
provenance record and obtain the required authorization before redistribution.
