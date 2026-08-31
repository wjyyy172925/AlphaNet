import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RES_ROOT = PROJECT_ROOT / "Res"
STAGES = [
    ("dataset", "\u6784\u5efa\u6570\u636e\u96c6.py", "\u6784\u5efa\u6570\u636e\u96c6"),
    ("train", "\u8bad\u7ec3\u6846\u67b6.py", "\u8bad\u7ec3\u6a21\u578b"),
    ("backtest", "\u56de\u6d4b\u6846\u67b6.py", "\u56de\u6d4b"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run dataset, train, and backtest in order.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Deprecated; outputs now use per-stage run folders under Res/.",
    )
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest step.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later steps if one step fails.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def should_skip(stage_name, args):
    return {
        "dataset": args.skip_dataset,
        "train": args.skip_train,
        "backtest": args.skip_backtest,
    }[stage_name]


def run_stage(script_name, display_name, python_executable, env=None, dry_run=False):
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    command = [python_executable, str(script_path)]
    print(f"\n[{display_name}]")
    print("Command:", " ".join(command))

    if dry_run:
        return 0, 0.0

    start_time = time.time()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    elapsed = time.time() - start_time
    return completed.returncode, elapsed


def main():
    args = parse_args()
    if not args.dry_run:
        RES_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ALPHANET_OUTPUT_ROOT"] = str(RES_ROOT)

    print(f"Output root: {RES_ROOT}")
    stage_results = []

    for stage_name, script_name, display_name in STAGES:
        if should_skip(stage_name, args):
            print(f"\n[{display_name}] skipped")
            stage_results.append((display_name, "skipped", 0.0))
            continue

        try:
            return_code, elapsed = run_stage(
                script_name,
                display_name,
                args.python,
                env=env,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as exc:
            print(f"\n[{display_name}] failed: {exc}")
            if not args.continue_on_error:
                return 1
            stage_results.append((display_name, "missing", 0.0))
            continue

        status = "success" if return_code == 0 else f"failed({return_code})"
        print(f"[{display_name}] done in {elapsed:.2f}s, status: {status}")
        stage_results.append((display_name, status, elapsed))

        if return_code != 0 and not args.continue_on_error:
            print("\nPipeline stopped at the failed step.")
            return return_code

    print("\nSummary:")
    for display_name, status, elapsed in stage_results:
        print(f"- {display_name}: {status}, {elapsed:.2f}s")

    if any(str(status).startswith("failed") for _, status, _ in stage_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
