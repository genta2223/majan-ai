"""Test CDP connectivity to active Majsoul Chrome/Electron instance."""

import asyncio
import json
import sys
import time
import urllib.request
import websockets

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fetch_cdp_targets(port: int = 9222):
    """Fetch targets from http://127.0.0.1:{port}/json."""
    url = f"http://127.0.0.1:{port}/json"
    print(f"[Step 1] Connecting to CDP HTTP endpoint: {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "MahjongAI-Test"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = resp.read()
            targets = json.loads(data.decode("utf-8", errors="replace"))
            print(f"  --> HTTP connection successful! Found {len(targets)} targets.")
            return targets
    except Exception as e:
        print(f"  [ERROR] Failed to connect to {url}: {e}")
        raise


def find_majsoul_tab(targets):
    """Locate the Majsoul game page tab."""
    print("\n[Step 2] Scanning targets for Majsoul ('雀魂-じゃんたま-') tab...")
    selected = None
    match_reason = ""

    # Dump all targets for reference
    for idx, t in enumerate(targets, 1):
        t_type = t.get("type", "")
        t_title = t.get("title", "")
        t_url = t.get("url", "")
        print(f"  [{idx}] Type: {t_type:<10} | Title: {t_title[:35]:<35} | URL: {t_url[:40]}")

    # Priority 1: type == page and "雀魂" in title
    for t in targets:
        if t.get("type") == "page" and "雀魂" in t.get("title", ""):
            selected = t
            match_reason = "Priority 1 (type=='page' and title contains '雀魂')"
            break

    # Priority 2: type == page and url/title contains mahjongsoul
    if not selected:
        for t in targets:
            title_lower = t.get("title", "").lower()
            url_lower = t.get("url", "").lower()
            if t.get("type") == "page" and ("mahjongsoul" in url_lower or "maj-soul" in url_lower or "majsoul" in title_lower):
                selected = t
                match_reason = "Priority 2 (type=='page' and URL contains 'mahjongsoul')"
                break

    if not selected:
        raise RuntimeError("Could not find Majsoul game tab among discovered targets!")

    ws_url = selected.get("webSocketDebuggerUrl")
    if not ws_url:
        raise RuntimeError(f"Target found ({selected.get('title')}), but webSocketDebuggerUrl is missing!")

    print(f"\n  --> Majsoul Tab Selected ({match_reason})")
    print(f"      ├─ Title  : {selected.get('title')}")
    print(f"      ├─ URL    : {selected.get('url')}")
    print(f"      └─ WS URL : {ws_url}")
    return ws_url


async def test_ws_listen(ws_url: str, duration_sec: float = 5.0):
    """Connect to DevTools WebSocket, enable Network, and listen for frames."""
    print(f"\n[Step 3] Connecting to DevTools WebSocket for {duration_sec:.1f}s...")
    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        print("  --> WebSocket connected successfully!")

        # Enable Network domain
        enable_cmd = {"id": 1, "method": "Network.enable", "params": {}}
        await ws.send(json.dumps(enable_cmd))
        print("  --> Sent 'Network.enable' command. Waiting for ACK and frame events...")

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration_sec:
            remaining = duration_sec - (time.time() - start_time)
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 1.0))
                msg = json.loads(raw_msg)
                method = msg.get("method", "")
                if msg.get("id") == 1:
                    print(f"  [ACK] Network.enable confirmed: {msg.get('result', {})}")
                elif method == "Network.webSocketFrameReceived":
                    frame_count += 1
                    params = msg.get("params", {})
                    resp = params.get("response", {})
                    payload = resp.get("payloadData", "")
                    print(f"  [FRAME #{frame_count}] Received WS frame (length: {len(payload)} chars)")
                elif method.startswith("Network.webSocket"):
                    print(f"  [WS EVENT] {method}")
            except asyncio.TimeoutError:
                pass

        print(f"\n[Summary] Successfully maintained connection for {duration_sec:.1f}s.")
        print(f"          Total WebSocket game frames captured: {frame_count}")
        print("==================================================================")
        print(">>> ALL 3 CDP VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print("==================================================================")


async def main():
    targets = fetch_cdp_targets(port=9222)
    ws_url = find_majsoul_tab(targets)
    await test_ws_listen(ws_url, duration_sec=5.0)


if __name__ == "__main__":
    asyncio.run(main())
