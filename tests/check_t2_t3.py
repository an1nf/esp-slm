#!/usr/bin/env python3
"""T2 and T3 packings decode the same tokens."""
import subprocess
import sys


def run(binary: str, model: str) -> str:
    out = subprocess.run([binary, "run", model, "--ids", "1,2,3,4,5", "--gen", "40"],
                         capture_output=True, text=True, check=True).stdout
    return next(l for l in out.splitlines() if l.startswith("generated:"))


def main() -> int:
    binary, t2, t3 = sys.argv[1:4]
    a, b = run(binary, t2), run(binary, t3)
    ok = a == b
    print(f"T2 = T3 bit-exact (40 greedy tokens): {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  T2:", a, "\n  T3:", b)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
