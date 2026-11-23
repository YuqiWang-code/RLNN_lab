"""Adapt a working copy of the release to the outer JAX evaluator JIT."""
import argparse
from pathlib import Path
import re


LINEN_JIT = re.compile(r"(?m)^[ \t]*@(?:partial\(flax\.linen\.jit,[^\n]+|flax\.linen\.jit)[ \t]*\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    networks = args.source / "Networks"
    for path in networks.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        compatible = LINEN_JIT.sub("", original)
        if compatible != original:
            path.write_text(compatible, encoding="utf-8")


if __name__ == "__main__":
    main()
