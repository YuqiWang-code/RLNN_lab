"""Arm names for the static-prior dose and direction controls."""

DOSES = ("0.02", "0.125", "0.25", "0.5", "1", "2")
ORACLE_DOSES = ("0.5", "1", "2")
ORACLE_NAMES = ("oracle-kamis", "oracle-b", "anti-oracle")
AMPLITUDES = tuple(f"amp@{dose}" for dose in DOSES)
RANDOM_CONTROLS = tuple(f"rand@{dose}" for dose in DOSES)
ORACLES = tuple(f"{name}@{dose}" for name in ORACLE_NAMES for dose in ORACLE_DOSES)
P0_ARMS = ("b0", "amp@0", "null-rand", "centered", "v1-node-only") + AMPLITUDES + RANDOM_CONTROLS + ORACLES
