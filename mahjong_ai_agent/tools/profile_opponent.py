"""CLI tool for profiling opponents from multiple paifu logs and generating exploit profiles."""

from __future__ import annotations
import argparse
import glob
import os
import sys

# Ensure mahjong_ai_agent root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.opponent_profiler import OpponentProfiler


def main():
    parser = argparse.ArgumentParser(description="Profile opponents from paifu logs and compile exploit database.")
    parser.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "output"),
                        help="Directory containing mjai JSON/JSONL paifu logs")
    parser.add_argument("--files", nargs="+", help="Specific paifu log files to process")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "config", "opponent_profiles.json"),
                        help="Target output path for opponent_profiles.json")
    args = parser.parse_args()

    files_to_process = []
    if args.files:
        files_to_process = args.files
    elif os.path.exists(args.dir):
        # Match mjai json files or jsonl
        files_to_process.extend(glob.glob(os.path.join(args.dir, "*_mjai.json")))
        files_to_process.extend(glob.glob(os.path.join(args.dir, "*.jsonl")))
        if not files_to_process:
            files_to_process.extend(glob.glob(os.path.join(args.dir, "*.json")))

    if not files_to_process:
        print(f"[ProfilerCLI] No paifu log files found in: {args.dir}")
        return

    print(f"[ProfilerCLI] Found {len(files_to_process)} paifu files. Analyzing opponent behaviors...")
    profiler = OpponentProfiler()
    profiler.analyze_files(files_to_process)
    profiler.save_profiles(args.output)

    # Display summary table
    print("\n" + "=" * 80)
    print("                    OPPONENT PROFILES & BEHAVIORAL ARCHETYPES                    ")
    print("=" * 80)
    print(f"{'Player Name':<16} | {'Archetype':<20} | {'Aggr':<6} | {'Open':<6} | {'Riichi':<6} | {'Deal-in':<7} | {'Rounds':<6}")
    print("-" * 80)

    for name, p in profiler.profiles.items():
        if p.total_rounds >= 1:
            print(
                f"{name:<16} | {p.archetype:<20} | {p.aggressiveness:>5.2f} | "
                f"{p.open_rate:>5.2f} | {p.riichi_rate:>5.2f} | {p.deal_in_rate:>6.2f} | {p.total_rounds:>6}"
            )

    print("=" * 80)
    print(f"[ProfilerCLI] Successfully compiled profiles to: {args.output}\n")


if __name__ == "__main__":
    main()
