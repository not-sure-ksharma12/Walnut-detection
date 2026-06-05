#!/usr/bin/env python3
"""
run_all_folds.py — Train all k-fold models in sequence (calls train_fold.py 0..9).

  python run_all_folds.py
  python run_all_folds.py --log training_log.txt --device auto
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all CV folds sequentially")
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Save console output (default: training_log.txt in repo root)",
    )
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds to run")
    from device_utils import add_device_argument

    add_device_argument(parser, default="auto")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else WORKSPACE / "training_log.txt"
    log_path = log_path.resolve()
    script = WORKSPACE / "train_fold.py"

    with log_path.open("w", encoding="utf-8") as log_file:

        def tee(line: str) -> None:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        for k in range(args.n_folds):
            if not (WORKSPACE / f"fold_{k}.json").exists():
                tee(f"Missing fold_{k}.json — stopping. Run build_fold_jsons.py first.\n")
                return 1
            tee(f"\n{'=' * 60}\nFold {k}/{args.n_folds - 1}\n{'=' * 60}\n")
            proc = subprocess.Popen(
                [sys.executable, str(script), str(k), "--device", args.device],
                cwd=str(WORKSPACE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                tee(line)
            ret = proc.wait()
            if ret != 0:
                tee(f"Fold {k} failed with exit code {ret}\n")
                return ret
        tee(f"\nAll {args.n_folds} folds completed.\n")

    print(f"Log saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
