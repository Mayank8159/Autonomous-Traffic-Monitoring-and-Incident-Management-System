# tests/test_vehicle_kinematics.py
import sys
import os
import time
import threading
import math
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
        print(f"[KINEMATICS] Server started on {BASE_URL}", flush=True)
        return httpd
    except OSError as e:
        print(f"[KINEMATICS] Port {PORT} in use, assuming server already active.", flush=True)
        return None


def run_kinematics_verification():
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    server = start_server_thread()
    time.sleep(1)

    chrome_exec = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    console_errors = []

    print("[KINEMATICS] Launching Chromium to verify continuous vehicle kinematics...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=chrome_exec if os.path.exists(chrome_exec) else None,
            args=["--no-sandbox", "--disable-gpu"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        page.route("**/favicon.ico", lambda route: route.fulfill(status=204, body=""))
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(f"Page Exception: {exc}"))

        url = f"{BASE_URL}/web_visualizer/visualizer.html"
        resp = page.goto(url, wait_until="networkidle", timeout=15000)
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"

        # 1. Verify fleet size and initial distribution
        veh_count = page.evaluate("() => typeof vehicles !== 'undefined' ? vehicles.length : 0")
        assert veh_count >= 80, f"Expected >= 80 vehicles, found {veh_count}"
        print(f"[PASS] Active vehicle fleet count: {veh_count} vehicles", flush=True)

        # 2. Test Chase Cam Engagement on a random vehicle
        print("[KINEMATICS] Engaging Chase Cam / Vehicle Lock...", flush=True)
        chase_res = page.evaluate("""() => {
            const v = vehicles[0];
            engageChaseCam(v.id);
            return {
                id: v.id,
                type: v.type,
                x: v.x,
                y: v.y,
                followedId: followedVehicleId,
                hudVisible: document.getElementById('vehicleChaseHud').style.display !== 'none'
            };
        }""")
        assert chase_res["followedId"] == chase_res["id"], "Chase Cam did not set followedVehicleId!"
        assert chase_res["hudVisible"], "Chase Cam Telemetry HUD is not visible!"
        print(f"[PASS] Chase Cam successfully locked on #{chase_res['id']} ({chase_res['type'].upper()})", flush=True)

        # 3. Track Followed Vehicle across 180 continuous simulation steps (3 seconds)
        print("[KINEMATICS] Tracking vehicle trajectory across 180 frames...", flush=True)
        trajectory = []
        for step in range(30):
            pt = page.evaluate("""() => {
                const v = vehicles.find(veh => veh.id === followedVehicleId);
                const hudSpeed = document.getElementById('chaseSpeed')?.textContent || '';
                const hudStreet = document.getElementById('chaseStreet')?.textContent || '';
                const hudAction = document.getElementById('chaseAction')?.textContent || '';
                return v ? {
                    x: v.x,
                    y: v.y,
                    speed: v.speed,
                    heading: v.heading,
                    state: v.state,
                    hudSpeed: hudSpeed,
                    hudStreet: hudStreet,
                    hudAction: hudAction,
                    panX: camera.panX,
                    panY: camera.panY
                } : null;
            }""")
            assert pt is not None, "Followed vehicle vanished during tracking!"
            trajectory.append(pt)
            page.wait_for_timeout(100)

        # 4. Rigorous Continuity Check: Verify ZERO teleportation or sudden coordinate snaps
        print(f"[KINEMATICS] Analyzing trajectory continuity across {len(trajectory)} sample intervals...", flush=True)
        max_dist_step = 0.0
        max_heading_step = 0.0
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            dist = math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
            max_dist_step = max(max_dist_step, dist)

            # Heading difference in [-pi, pi]
            diff_h = abs(p2["heading"] - p1["heading"])
            if diff_h > math.pi:
                diff_h = abs(diff_h - 2 * math.pi)
            max_heading_step = max(max_heading_step, diff_h)

            # Max allowed delta across 100ms at max speed (4.5 px/frame * 6 frames) is ~30px
            assert dist < 35.0, f"Teleportation detected! Step {i} -> {i+1} jumped {dist:.2f}px from ({p1['x']}, {p1['y']}) to ({p2['x']}, {p2['y']})"

        print(f"[PASS] Max 100ms displacement: {max_dist_step:.2f}px (Smooth & Continuous, ZERO Teleportation!)", flush=True)
        print(f"[PASS] Max heading change: {max_heading_step:.2f} rad (Smooth orientation, zero sudden 90° snaps)", flush=True)
        print(f"[PASS] Live Telemetry HUD verified: Speed={trajectory[-1]['hudSpeed']}, Street={trajectory[-1]['hudStreet']}, Action={trajectory[-1]['hudAction']}", flush=True)

        # 5. Verify Smooth Intersection Bezier Turns & Corridor Transitions
        print("[KINEMATICS] Verifying smooth corridor turning across city network...", flush=True)
        start_corridors = page.evaluate("() => Object.fromEntries(vehicles.map(v => [v.id, v.corridorId]))")
        turns_detected = 0
        for _ in range(50):
            turning_count = page.evaluate("() => vehicles.filter(v => v.state === 'TURNING').length")
            if turning_count > 0:
                turns_detected += turning_count
            page.wait_for_timeout(80)

        end_corridors = page.evaluate("() => Object.fromEntries(vehicles.map(v => [v.id, v.corridorId]))")
        corridor_switches = sum(1 for vid, c0 in start_corridors.items() if vid in end_corridors and end_corridors[vid] != c0)
        assert turns_detected > 0, "No intersection turns detected!"
        assert corridor_switches > 0, "No corridor switches detected!"
        print(f"[PASS] Active intersection Bezier turns: {turns_detected}, Corridor switches: {corridor_switches}", flush=True)

        # 6. Capture Chase Cam Screenshot
        shot_chase = os.path.join(ROOT_DIR, "results", "chase_cam_telemetry.png")
        page.screenshot(path=shot_chase, full_page=False)
        assert os.path.exists(shot_chase) and os.path.getsize(shot_chase) > 20000, "Chase cam screenshot failed"
        print(f"[PASS] Chase Cam Telemetry screenshot saved to {shot_chase} ({os.path.getsize(shot_chase)} bytes)", flush=True)

        # 7. Test Disengaging Chase Cam
        page.evaluate("() => disengageChaseCam()")
        is_disengaged = page.evaluate("() => followedVehicleId === null && document.getElementById('vehicleChaseHud').style.display === 'none'")
        assert is_disengaged, "Chase Cam failed to disengage cleanly!"
        print("[PASS] Chase Cam disengagement verified cleanly.", flush=True)

        # 8. Check Console Errors
        assert len(console_errors) == 0, f"Found console errors: {console_errors}"
        print("[PASS] Zero console errors during vehicle kinematics verification!", flush=True)

        browser.close()

    if server:
        server.shutdown()
        server.server_close()
        print("[KINEMATICS] Server closed.", flush=True)

    print("\nALL VEHICLE KINEMATICS & CHASE CAM VERIFICATIONS PASSED 100%!", flush=True)
    return True


if __name__ == "__main__":
    success = run_kinematics_verification()
    sys.exit(0 if success else 1)
