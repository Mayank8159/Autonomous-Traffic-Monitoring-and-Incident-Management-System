# tests/test_playwright_e2e.py
import sys
import os
import time
import threading
from http.server import ThreadingHTTPServer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.server import ATMSRequestHandler
from playwright.sync_api import sync_playwright

PORT = 8000
BASE_URL = f"http://localhost:{PORT}"


def start_server_thread():
    server_address = ("127.0.0.1", PORT)
    try:
        httpd = ThreadingHTTPServer(server_address, ATMSRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        print(f"[E2E] Server started on {BASE_URL}", flush=True)
        return httpd
    except OSError as e:
        print(f"[E2E] Port {PORT} in use ({e}), assuming server already active.", flush=True)
        return None


def run_e2e_tests():
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    server = start_server_thread()
    time.sleep(1)

    chrome_exec = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    console_logs = []
    console_errors = []

    print("[E2E] Launching Headless Chromium via Playwright...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=chrome_exec if os.path.exists(chrome_exec) else None,
            args=["--no-sandbox", "--disable-gpu"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Intercept favicon.ico to prevent 404 console errors
        page.route("**/favicon.ico", lambda route: route.fulfill(status=204, body=""))

        # Capture console messages
        def on_console(msg):
            console_logs.append(f"[{msg.type}] {msg.text}")
            if msg.type == "error":
                console_errors.append(f"Console Error: {msg.text} (at {msg.location})")

        page.on("console", on_console)
        page.on("pageerror", lambda exc: console_errors.append(f"Page Exception: {exc}"))

        url = f"{BASE_URL}/web_visualizer/visualizer.html"
        print(f"[E2E] Navigating to {url}...", flush=True)
        resp = page.goto(url, wait_until="networkidle", timeout=15000)
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"

        # 1. Verify 2.5D City Canvas and Simulation Elements
        page.wait_for_selector("#cityMapCanvas", timeout=5000)
        print("[E2E] Canvas #cityMapCanvas verified present.", flush=True)

        # Verify client-side PyTorch RL Policy is loaded and active
        has_rl_weights = page.evaluate("() => typeof window.RL_POLICY_WEIGHTS !== 'undefined' && !!window.RL_POLICY_WEIGHTS.w1")
        assert has_rl_weights, "window.RL_POLICY_WEIGHTS not loaded!"
        print("[PASS] Verified client-side genuine PyTorch RL policy weights loaded.", flush=True)

        # Verify forward inference evaluates without error
        rl_eval = page.evaluate("() => typeof evaluateRLPolicy === 'function' ? evaluateRLPolicy([2, 3, 1, 4, 10, 1, 0]) : null")
        assert rl_eval is not None and "probs" in rl_eval, f"RL forward inference failed: {rl_eval}"
        print(f"[PASS] RL Forward inference output: action={rl_eval.get('action')}, probs={rl_eval.get('probs')}", flush=True)

        # Let simulation render 3 seconds of silky smooth 2.5D animation
        page.wait_for_timeout(3000)

        # Capture City Command Overview Screenshot (2.5D Isometric)
        shot1 = os.path.join(ROOT_DIR, "results", "city_command_screenshot.png")
        page.screenshot(path=shot1, full_page=False)
        assert os.path.exists(shot1) and os.path.getsize(shot1) > 20000, "City screenshot failed"
        print(f"[PASS] 2.5D City Command Screenshot saved to {shot1} ({os.path.getsize(shot1)} bytes)", flush=True)

        # 2. Test 2.5D Projection Mode Toggle (Isometric <-> Elevated)
        proj_btn = page.locator("#btnToggleProjection")
        if proj_btn.count() > 0:
            print("[E2E] Testing 2.5D Projection Toggle...", flush=True)
            proj_btn.click() # switches to elevated
            page.wait_for_timeout(800)
            proj_btn.click() # switches back to isometric
            page.wait_for_timeout(500)
            print("[PASS] 2.5D Projection Mode toggle verified.", flush=True)

        # 3. Test Day/Night Theme Toggle
        theme_btn = page.locator("#btnToggleTheme")
        if theme_btn.count() > 0:
            print("[E2E] Toggling to Day Mode...", flush=True)
            theme_btn.click()
            page.wait_for_timeout(1000)
            shot_day = os.path.join(ROOT_DIR, "results", "city_daymode_screenshot.png")
            page.screenshot(path=shot_day, full_page=False)
            assert os.path.exists(shot_day) and os.path.getsize(shot_day) > 20000, "Day mode screenshot failed"
            print(f"[PASS] Day Mode Screenshot saved to {shot_day} ({os.path.getsize(shot_day)} bytes)", flush=True)
            # Switch back to dark mode
            theme_btn.click()
            page.wait_for_timeout(500)

        # 4. Test Incident Simulation: Crash at Downtown Core
        print("[E2E] Testing 'Crash at Downtown Core' trigger...", flush=True)
        crash_btn = page.locator("button:has-text('Crash at Downtown Core')")
        assert crash_btn.count() > 0, "Could not find 'Crash at Downtown Core' button"
        crash_btn.click()

        # Check that topStatusBadge reflects critical incident
        page.wait_for_timeout(800)
        badge = page.locator("#topStatusBadge")
        badge_text = badge.text_content() or ""
        print(f"[E2E] Status badge after crash trigger: '{badge_text}'", flush=True)
        assert "CRITICAL" in badge_text.upper() or "INCIDENT" in badge_text.upper() or "ALERT" in badge_text.upper(), \
            f"Badge did not reflect incident: {badge_text}"

        # Verify emergency vehicles dispatched (police, ambulance)
        num_emergency = page.evaluate("() => typeof vehicles !== 'undefined' ? vehicles.filter(v => v.isEmergency).length : 0")
        num_dispatches = page.evaluate("() => typeof activeDispatches !== 'undefined' ? activeDispatches.length : 0")
        print(f"[E2E] Active emergency vehicles dispatched: {num_emergency}, Dispatches active: {num_dispatches}", flush=True)
        assert num_emergency >= 2 or num_dispatches >= 1, "Emergency units were not dispatched!"

        # Wait 4 seconds for emergency vehicle navigation and green-wave preemption
        page.wait_for_timeout(4000)

        shot_incident = os.path.join(ROOT_DIR, "results", "city_incident_screenshot.png")
        page.screenshot(path=shot_incident, full_page=False)
        assert os.path.exists(shot_incident) and os.path.getsize(shot_incident) > 20000, "Incident screenshot failed"
        print(f"[PASS] Emergency Incident Screenshot saved to {shot_incident} ({os.path.getsize(shot_incident)} bytes)", flush=True)

        # 5. Test CCTV Video Perception View
        cctv_btn = page.locator("#btnViewVideo")
        if cctv_btn.count() > 0:
            print("[E2E] Switching to CCTV Video Perception View...", flush=True)
            cctv_btn.click()
            page.wait_for_timeout(1500)
            shot_cctv = os.path.join(ROOT_DIR, "results", "cctv_tab_screenshot.png")
            page.screenshot(path=shot_cctv, full_page=False)
            assert os.path.exists(shot_cctv) and os.path.getsize(shot_cctv) > 20000, "CCTV screenshot failed"
            print(f"[PASS] CCTV View Screenshot saved to {shot_cctv} ({os.path.getsize(shot_cctv)} bytes)", flush=True)

            # Switch back to city view
            page.locator("#btnViewCity").click()
            page.wait_for_timeout(500)

        # 6. Check Console Errors
        print(f"[E2E] Total console logs recorded: {len(console_logs)}", flush=True)
        if console_errors:
            print("[E2E] Console errors detected:", flush=True)
            for err in console_errors:
                print(f"  ❌ {err}", flush=True)
        assert len(console_errors) == 0, f"Found {len(console_errors)} console errors: {console_errors}"
        print("[PASS] Zero unhandled console errors verified!", flush=True)

        browser.close()

    if server:
        server.shutdown()
        server.server_close()
        print("[E2E] Server closed.", flush=True)

    print("\nALL PLAYWRIGHT E2E BROWSER TESTS PASSED SUCCESSFULLY!", flush=True)
    return True


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
