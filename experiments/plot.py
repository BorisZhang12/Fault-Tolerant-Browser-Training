"""Regenerate every README chart from run folders.

Reads runs/<run-id>/metrics.jsonl and events.jsonl and writes PNGs to docs/.
No result goes in the README unless a run folder backs it.
"""
import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    args = p.parse_args()
    # TODO(milestone 6): accuracy vs. epoch per kill rate (E3), time to 95% vs. tabs (E2),
    # one run's accuracy with worker deaths marked, drop vs. down-weight (E4).
    print(f"plot: reading {args.runs}, writing {args.out} (not implemented yet)")


if __name__ == "__main__":
    main()
