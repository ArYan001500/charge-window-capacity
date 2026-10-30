"""Command-line entry point."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(prog="python -m chargewin")
    commands = parser.add_subparsers(dest="command", required=True)
    rep = commands.add_parser("reproduce")
    rep.add_argument("--tier", choices=["paper", "quick", "main", "full"], required=True)
    rep.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    rep.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    rep.add_argument("--offline", action="store_true", help="use files already on disk, no network request")
    args = parser.parse_args()
    try:
        from chargewin.reproduce import run
        run(args.tier, args.data_dir, args.out_dir, args.offline)
    except Exception as exc:
        print(f"FAIL: {args.tier}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
