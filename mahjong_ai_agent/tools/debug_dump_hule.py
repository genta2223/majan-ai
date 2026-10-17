"""Debug utility to dump and analyze RecordHule and discard action structures in paifu data."""

import os
import json
import glob
from typing import Dict, Any, List


def inspect_paifu_file(path: str):
    print("=" * 80)
    print(f"[FILE] {path}")
    print("=" * 80)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON ({path}): {e}")
        # Try inspecting truncated raw text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            print(f"File raw length: {len(content)} chars. Last 200 chars:\n{content[-200:]}")
        return

    if isinstance(data, list):
        print(f"Data is MJAI event list (Total events: {len(data)})")
        horas = [ev for ev in data if ev.get("type") == "hora"]
        riichis = [ev for ev in data if ev.get("type") == "reach"]
        discards = [ev for ev in data if ev.get("type") == "dahai"]
        print(f"Total hora: {len(horas)}, reach: {len(riichis)}, dahai: {len(discards)}")
        for h_idx, h in enumerate(horas):
            print(f"  * Hora[{h_idx}]: actor={h.get('actor')}, target={h.get('target')}, pai={h.get('pai')}, score={h.get('score')}, deltas={h.get('deltas')}")
        return

    # 1. Accounts
    head = data.get("head", {})
    accounts = head.get("accounts") or data.get("players") or []
    print("\n--- 1. Accounts ---")
    for acc in accounts:
        print(f"  Seat: {acc.get('seat')} | Nickname: {acc.get('nickname') or acc.get('name')} | ID: {acc.get('account_id')}")

    # 2. Scan rounds and actions
    rounds = data.get("rounds", [])
    print(f"\n--- 2. Rounds (Total: {len(rounds)}) ---")

    hule_count = 0
    riichi_count = 0
    discard_count = 0

    for r_idx, r in enumerate(rounds):
        round_info = r.get("round_info", {})
        scores_at_start = round_info.get("scores", [])
        actions = r.get("actions", [])
        print(f"\n[Round {r_idx}] Start Scores: {scores_at_start} | Total Actions: {len(actions)}")

        for a_idx, a in enumerate(actions):
            act_name = a.get("action_name", "")
            act_data = a.get("data", {})

            # Check Discard / Riichi
            if "Discard" in act_name:
                discard_count += 1
                if act_data.get("is_liqi") or act_data.get("is_wliqi"):
                    riichi_count += 1
                    print(f"  Action #{a_idx} [{act_name} RIICHI!]: seat={act_data.get('seat')}, tile={act_data.get('tile')}, is_liqi={act_data.get('is_liqi')}")

            # Check Hule / Hora
            if "Hule" in act_name or act_name == "hora":
                hule_count += 1
                print(f"\n  >>> Action #{a_idx} [{act_name}] FOUND! <<<")
                print(f"  Full Action Data: {json.dumps(act_data, ensure_ascii=False, indent=4)}")
                
                # Examine score / point sources
                hules_list = act_data.get("hules", [])
                delta_scores = act_data.get("delta_scores", [])
                print(f"  * delta_scores: {delta_scores}")
                for h_idx, h in enumerate(hules_list):
                    print(f"  * Hule[{h_idx}]: seat={h.get('seat')}, target={h.get('target')}, zimo={h.get('zimo')}, tile={h.get('hu_tile')}")
                    print(f"    - h.keys: {list(h.keys())}")
                    for score_key in ("score", "point_sum", "dadian", "point_rong", "point_zimo", "point"):
                        if score_key in h:
                            print(f"    - {score_key} = {h.get(score_key)}")

    print(f"\n[Summary for {path}]")
    print(f"  Total Hule Events: {hule_count}")
    print(f"  Total Riichi Discards: {riichi_count}")
    print(f"  Total Discards: {discard_count}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_paths = [
        os.path.join(root, "output", "cache", "*.json"),
        os.path.join(root, "output", "*_mjai.json"),
        os.path.join(root, "data", "paifu_cache", "*.json"),
    ]

    found_files = []
    for sp in search_paths:
        found_files.extend(glob.glob(sp))

    print(f"Found {len(found_files)} paifu JSON files.")
    for fp in found_files:
        inspect_paifu_file(fp)


if __name__ == "__main__":
    main()
