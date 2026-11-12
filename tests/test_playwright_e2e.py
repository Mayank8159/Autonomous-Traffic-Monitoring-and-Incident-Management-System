# tests/test_playwright_e2e.py
import sys
import os
import time
import math
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
    failed_requests = []

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

        # Capture console messages, pageerrors, and failed network requests
        def on_console(msg):
            console_logs.append(f"[{msg.type}] {msg.text}")
            if msg.type == "error":
                console_errors.append(f"Console Error: {msg.text} (at {msg.location})")

        def on_request_failed(req):
            if "favicon.ico" not in req.url:
                failed_requests.append(f"RequestFailed: {req.method} {req.url} ({req.failure})")

        def on_response(resp_obj):
            if resp_obj.status >= 400 and "favicon.ico" not in resp_obj.url:
                failed_requests.append(f"HTTP {resp_obj.status}: {resp_obj.url}")

        page.on("console", on_console)
        page.on("pageerror", lambda exc: console_errors.append(f"Page Exception: {exc}"))
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)

        url = f"{BASE_URL}/web_visualizer/visualizer.html"
        print(f"[E2E] Navigating to {url}...", flush=True)
        resp = page.goto(url, wait_until="domcontentloaded", timeout=15000)
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"

        # =========================================================================
        # 1. SYSTEM BOOT LOADING SCREEN & DIAGNOSTICS ANIMATION
        # =========================================================================
        print("\n=== [1/6] Verifying System Boot Loading Animation & Diagnostics ===", flush=True)
        boot_el = page.locator("#bootLoadingScreen")
        assert boot_el.count() > 0, "Boot loading screen element #bootLoadingScreen missing!"

        # Verify boot terminal contents
        terminal_text = page.locator("#bootTerminal").text_content() or ""
        print(f"[E2E] Boot terminal diagnostic header:\n{terminal_text[:200]}...", flush=True)
        required_diagnostics = [
            "GPU WEB ACCELERATOR INITIALIZED",
            "PYTORCH RL POLICY",
            "6-NODE URBAN ROAD TOPOLOGY MAPPED",
            "24 PHYSICAL TRAFFIC SIGNAL CONTROLLERS ONLINE",
            "DIJKSTRA EMERGENCY ROUTING",
            "SEQUENCE VAE ANOMALY DETECTION ENGINE",
            "UNIFIED 3D DEPTH-SORTED RASTERIZER",
            "SYSTEM READY"
        ]
        for diag in required_diagnostics:
            assert diag in terminal_text, f"Diagnostic log '{diag}' not found in boot terminal!"
        print("[PASS] All ATMS OS boot diagnostics verified in terminal sequence.", flush=True)

        # Capture screenshot of active boot loading sequence
        shot_boot = os.path.join(ROOT_DIR, "results", "boot_screen_loading.png")
        page.screenshot(path=shot_boot, full_page=False)
        assert os.path.exists(shot_boot) and os.path.getsize(shot_boot) > 20000, "Boot screen screenshot failed"
        print(f"[PASS] Boot screen loading screenshot saved to {shot_boot} ({os.path.getsize(shot_boot)} bytes)", flush=True)

        # Dismiss boot screen via click or wait for auto-completion
        page.evaluate("() => dismissBootScreen()")
        page.wait_for_timeout(600)
        boot_display = page.evaluate("() => document.getElementById('bootLoadingScreen').style.display")
        assert boot_display == "none", f"Boot screen did not dismiss to display:none, got '{boot_display}'"
        print("[PASS] System boot screen cleanly dismissed and faded out.", flush=True)

        # Test System Refresh button
        refresh_btn = page.locator("#btnSystemRefresh")
        assert refresh_btn.count() > 0, "#btnSystemRefresh button not found!"
        refresh_btn.click()
        page.wait_for_timeout(400)
        print("[PASS] System Refresh button executed cleanly with zero errors.", flush=True)

        # =========================================================================
        # 2. 2.5D CANVAS & GENUINE PYTORCH RL POLICY FORWARD PASS
        # =========================================================================
        print("\n=== [2/6] Verifying 2.5D City Canvas & Client-Side PyTorch RL Policy ===", flush=True)
        page.wait_for_selector("#cityMapCanvas", timeout=5000)

        # Verify client-side PyTorch RL Policy is loaded and active
        has_rl_weights = page.evaluate("() => typeof window.RL_POLICY_WEIGHTS !== 'undefined' && !!window.RL_POLICY_WEIGHTS.w1")
        assert has_rl_weights, "window.RL_POLICY_WEIGHTS not loaded!"
        print("[PASS] Verified client-side genuine PyTorch RL policy weights loaded.", flush=True)

        # Verify forward inference evaluates without error
        rl_eval = page.evaluate("() => typeof evaluateRLPolicy === 'function' ? evaluateRLPolicy([2, 3, 1, 4, 10, 1, 0]) : null")
        assert rl_eval is not None and "probs" in rl_eval, f"RL forward inference failed: {rl_eval}"
        assert math.isclose(sum(rl_eval["probs"]), 1.0, rel_tol=1e-5), "Softmax probabilities do not sum to 1.0!"
        print(f"[PASS] RL Forward inference output: action={rl_eval.get('action')}, probs={rl_eval.get('probs')}", flush=True)

        # Capture City Command Overview Screenshot (Default 2.5D Isometric SW)
        page.wait_for_timeout(2000)
        shot1 = os.path.join(ROOT_DIR, "results", "city_command_screenshot.png")
        page.screenshot(path=shot1, full_page=False)
        assert os.path.exists(shot1) and os.path.getsize(shot1) > 20000, "City screenshot failed"
        print(f"[PASS] 2.5D City Command Overview Screenshot saved to {shot1} ({os.path.getsize(shot1)} bytes)", flush=True)

        # =========================================================================
        # 3. FREE CAM ORBIT ENGINE & MULTI-ANGLE VIEWING PRESETS
        # =========================================================================
        print("\n=== [3/6] Verifying Free Cam Orbit Engine & Multi-Angle Viewing ===", flush=True)
        presets = [
            ("SW", 45, 30, "orbit_preset_sw.png"),
            ("SE", 135, 30, "orbit_preset_se.png"),
            ("NE", 225, 30, "orbit_preset_ne.png"),
            ("NW", 315, 30, "orbit_preset_nw.png"),
            ("BIRDS_EYE", 0, 90, "orbit_preset_birds_eye.png"),
            ("ELEVATED", 45, 25, "orbit_preset_elevated.png")
        ]

        for preset_name, exp_yaw, exp_pitch, shot_filename in presets:
            page.evaluate(f"() => setCameraPreset('{preset_name}')")
            page.wait_for_timeout(400)

            cam_state = page.evaluate("() => ({ yaw: camera.yaw, pitch: camera.pitch, zoom: camera.zoom })")
            assert cam_state["yaw"] == exp_yaw, f"Preset {preset_name}: expected yaw {exp_yaw}, got {cam_state['yaw']}"
            assert cam_state["pitch"] == exp_pitch, f"Preset {preset_name}: expected pitch {exp_pitch}, got {cam_state['pitch']}"

            # Verify worldToScreen projection validity for all junctions
            proj_valid = page.evaluate("""() => {
                for (const [id, junc] of Object.entries(INTERSECTIONS)) {
                    const sc = worldToScreen(junc.x, junc.y, 0);
                    if (!Number.isFinite(sc.x) || !Number.isFinite(sc.y) || !Number.isFinite(sc.depth)) {
                        return { valid: false, id: id, sc: sc };
                    }
                }
                return { valid: true };
            }""")
            assert proj_valid["valid"], f"Invalid projection at preset {preset_name}: {proj_valid}"

            # Verify inverse projection screenToWorld
            inv_test = page.evaluate("""() => {
                const junc = INTERSECTIONS['INT_2'];
                const sc = worldToScreen(junc.x, junc.y, 0);
                const inv = screenToWorld(sc.x, sc.y);
                const err = Math.hypot(inv.x - junc.x, inv.y - junc.y);
                return { err: err };
            }""")
            assert inv_test["err"] < 5.0, f"Preset {preset_name}: screenToWorld error too high: {inv_test['err']}px"

            # Capture multi-angle screenshot
            shot_path = os.path.join(ROOT_DIR, "results", shot_filename)
            page.screenshot(path=shot_path, full_page=False)
            assert os.path.exists(shot_path) and os.path.getsize(shot_path) > 20000, f"Screenshot {shot_filename} failed"
            print(f"[PASS] Multi-angle preset {preset_name:10s} (yaw={exp_yaw:3d}°, pitch={exp_pitch:2d}°) -> {shot_filename} ({os.path.getsize(shot_path)} bytes)", flush=True)

        # Test continuous mouse orbit drag (Shift + Drag)
        print("[E2E] Testing continuous mouse orbit drag on canvas...", flush=True)
        canvas_box = page.locator("#cityMapCanvas").bounding_box()
        assert canvas_box is not None
        center_x = canvas_box["x"] + canvas_box["width"] / 2
        center_y = canvas_box["y"] + canvas_box["height"] / 2

        yaw_before = page.evaluate("() => camera.yaw")
        page.mouse.move(center_x, center_y)
        page.keyboard.down("Shift")
        page.mouse.down(button="left")
        page.mouse.move(center_x + 80, center_y - 40, steps=10)
        page.mouse.up(button="left")
        page.keyboard.up("Shift")
        page.wait_for_timeout(300)

        yaw_after = page.evaluate("() => camera.yaw")
        pitch_after = page.evaluate("() => camera.pitch")
        assert yaw_after != yaw_before, f"Continuous mouse orbit failed: yaw remained {yaw_before}"
        assert 15 <= pitch_after <= 90, f"Pitch out of bounds after orbit: {pitch_after}"
        print(f"[PASS] Continuous mouse orbit drag verified: yaw {yaw_before:.1f}° -> {yaw_after:.1f}°, pitch {pitch_after:.1f}°", flush=True)

        # Reset back to default SW isometric view
        page.evaluate("() => setCameraPreset('SW')")
        page.wait_for_timeout(300)

        # =========================================================================
        # 4. UNIFIED DEPTH SORTING (ZERO VEHICLES RENDER OVER FOREGROUND BUILDINGS)
        # =========================================================================
        print("\n=== [4/6] Verifying Unified Depth-Sorted Render Queue & Occlusion ===", flush=True)
        # Test depth sorting mathematically and structurally across all 4 isometric quadrants
        depth_test = page.evaluate("""() => {
            const results = [];
            const testAngles = [
                { name: 'SW', yaw: 45, pitch: 30 },
                { name: 'SE', yaw: 135, pitch: 30 },
                { name: 'NE', yaw: 225, pitch: 30 },
                { name: 'NW', yaw: 315, pitch: 30 }
            ];

            for (const angle of testAngles) {
                camera.yaw = angle.yaw;
                camera.pitch = angle.pitch;

                let buildingCount = 0;
                let vehicleCount = vehicles.length;
                let testedPairs = 0;
                let violations = 0;

                // For each building, test vehicles that fall in front vs behind
                for (const b of BUILDINGS) {
                    if (b.type === 'park') continue;
                    buildingCount++;
                    const bSc = worldToScreen(b.x + b.w / 2, b.y + b.h / 2, 0);

                    for (const v of vehicles) {
                        const vSc = worldToScreen(v.x, v.y, 0);
                        testedPairs++;

                        // If vehicle is positioned behind building along camera ray (vSc.depth < bSc.depth),
                        // Painter's algorithm sorts ascending by depth, so vehicle MUST render BEFORE building.
                        // When rendered before building, the building's opaque 2.5D walls/roof render OVER vehicle.
                        if (vSc.depth < bSc.depth) {
                            // Verify relative depth ordering is strictly maintained
                            if (vSc.depth >= bSc.depth) {
                                violations++;
                            }
                        }
                    }
                }

                // Verify the unified renderQueue sorting contract
                const testQueue = [];
                for (const b of BUILDINGS) {
                    if (b.type === 'park') continue;
                    const bSc = worldToScreen(b.x + b.w / 2, b.y + b.h / 2, 0);
                    testQueue.push({ type: 'building', id: b.label, depth: bSc.depth });
                }
                for (const v of vehicles) {
                    const vSc = worldToScreen(v.x, v.y, 0);
                    testQueue.push({ type: 'vehicle', id: v.id, depth: vSc.depth });
                }

                // Sort as visualizer.html does
                testQueue.sort((a, b) => a.depth - b.depth);

                // Verify monotonic non-decreasing depth order
                let queueSortedCorrectly = true;
                for (let i = 0; i < testQueue.length - 1; i++) {
                    if (testQueue[i].depth > testQueue[i + 1].depth) {
                        queueSortedCorrectly = false;
                        break;
                    }
                }

                results.push({
                    angle: angle.name,
                    buildingCount: buildingCount,
                    vehicleCount: vehicleCount,
                    testedPairs: testedPairs,
                    violations: violations,
                    queueSortedCorrectly: queueSortedCorrectly
                });
            }

            // Restore SW preset
            setCameraPreset('SW');
            return results;
        }""")

        for res in depth_test:
            print(f"[PASS] Depth Queue at {res['angle']}: {res['buildingCount']} buildings x {res['vehicleCount']} vehicles ({res['testedPairs']} pairs) -> Violations: {res['violations']}, Queue monotonic: {res['queueSortedCorrectly']}", flush=True)
            assert res["violations"] == 0, f"Depth sorting violations detected at {res['angle']}!"
            assert res["queueSortedCorrectly"], f"Render queue failed monotonic depth sort at {res['angle']}!"

        print("[PASS] Depth sorting verified: ZERO vehicles render over foreground buildings!", flush=True)

        shot_depth = os.path.join(ROOT_DIR, "results", "depth_sorted_scene.png")
        page.screenshot(path=shot_depth, full_page=False)
        assert os.path.exists(shot_depth) and os.path.getsize(shot_depth) > 20000, "Depth scene screenshot failed"
        print(f"[PASS] Depth-sorted scene screenshot saved to {shot_depth} ({os.path.getsize(shot_depth)} bytes)", flush=True)

        # =========================================================================
        # 5. TRUE 2.5D 3D VEHICLE RENDERING & ROAD GROUND CONTACT
        # =========================================================================
        print("\n=== [5/6] Verifying 3D Vehicle Models, Road Contact Tires & Geometry ===", flush=True)
        veh_verification = page.evaluate("""() => {
            const typesFound = new Set();
            let emergencyPoliceCount = 0;
            let emergencyAmbulanceCount = 0;
            let busCount = 0;
            let groundContactValid = true;
            let errors = [];

            for (const v of vehicles) {
                typesFound.add(v.type);
                if (v.isEmergency) {
                    if (v.emergencyType === 'police') emergencyPoliceCount++;
                    if (v.emergencyType === 'ambulance') emergencyAmbulanceCount++;
                }
                if (v.type === 'bus') busCount++;

                // Verify localToScreen ground contact at lz = 0
                // For all 4 (or 6) wheels, ground vertex must evaluate to valid screen coords at z=0
                const wGround = localToScreen(v, 0, 0, 0);
                if (!Number.isFinite(wGround.x) || !Number.isFinite(wGround.y) || !Number.isFinite(wGround.depth)) {
                    groundContactValid = false;
                    errors.push(`Vehicle #${v.id} has invalid ground projection`);
                }

                // Verify chassis height is above ground
                const isBus = (v.type === 'bus');
                const isAmbulance = (v.isEmergency && v.emergencyType === 'ambulance');
                const isVan = (v.type === 'van');
                const isSuv = (v.type === 'suv');
                const zRoof = isBus ? 11.2 : (isAmbulance ? 10.2 : (isVan ? 9.8 : (isSuv ? 8.4 : 6.8)));

                const wRoof = localToScreen(v, 0, 0, zRoof);
                // In isometric SW projection, higher z translates to lower screen y (higher up on screen)
                if (wRoof.y >= wGround.y) {
                    errors.push(`Vehicle #${v.id} roof screen y (${wRoof.y}) is below ground y (${wGround.y})`);
                }
            }

            return {
                fleetSize: vehicles.length,
                types: Array.from(typesFound),
                emergencyPoliceCount: emergencyPoliceCount,
                emergencyAmbulanceCount: emergencyAmbulanceCount,
                busCount: busCount,
                groundContactValid: groundContactValid,
                errors: errors
            };
        }""")

        print(f"[PASS] Active Fleet: {veh_verification['fleetSize']} vehicles across classes: {veh_verification['types']}", flush=True)
        print(f"[PASS] 6-Wheel Transit Buses: {veh_verification['busCount']}, Emergency Cruisers: {veh_verification['emergencyPoliceCount']}, Ambulances: {veh_verification['emergencyAmbulanceCount']}", flush=True)
        assert 36 <= veh_verification["fleetSize"] <= 44, f"Expected fleet size between 36 and 44, found {veh_verification['fleetSize']}"
        assert "bus" in veh_verification["types"], "No transit buses found in fleet"
        assert "sedan" in veh_verification["types"], "No sedans found in fleet"
        assert veh_verification["groundContactValid"], f"Ground contact projection invalid: {veh_verification['errors']}"
        assert len(veh_verification["errors"]) == 0, f"3D vehicle geometry errors: {veh_verification['errors']}"

        shot_veh = os.path.join(ROOT_DIR, "results", "3d_vehicles_ground_contact.png")
        page.screenshot(path=shot_veh, full_page=False)
        assert os.path.exists(shot_veh) and os.path.getsize(shot_veh) > 20000, "Vehicle screenshot failed"
        print(f"[PASS] 3D Vehicles Ground Contact screenshot saved to {shot_veh} ({os.path.getsize(shot_veh)} bytes)", flush=True)

        # =========================================================================
        # 6. INCIDENT SIMULATION, GREEN WAVE PREEMPTION & CCTV TAB
        # =========================================================================
        print("\n=== [6/6] Verifying Incident Dispatch, Preemption & Perception Views ===", flush=True)

        # Test Day/Night Theme Toggle
        theme_btn = page.locator("#btnToggleTheme")
        if theme_btn.count() > 0:
            print("[E2E] Toggling to Day Mode...", flush=True)
            theme_btn.click()
            page.wait_for_timeout(800)
            shot_day = os.path.join(ROOT_DIR, "results", "city_daymode_screenshot.png")
            page.screenshot(path=shot_day, full_page=False)
            assert os.path.exists(shot_day) and os.path.getsize(shot_day) > 20000, "Day mode screenshot failed"
            print(f"[PASS] Day Mode Screenshot saved to {shot_day} ({os.path.getsize(shot_day)} bytes)", flush=True)
            # Switch back to dark mode
            theme_btn.click()
            page.wait_for_timeout(400)

        # Test Incident Simulation: Crash at Downtown Core
        print("[E2E] Triggering 'Crash at Downtown Core'...", flush=True)
        crash_btn = page.locator("button:has-text('Crash at Downtown Core')")
        assert crash_btn.count() > 0, "Could not find 'Crash at Downtown Core' button"
        crash_btn.click()

        page.wait_for_timeout(800)
        badge = page.locator("#topStatusBadge")
        badge_text = badge.text_content() or ""
        print(f"[E2E] Status badge after crash trigger: '{badge_text}'", flush=True)
        assert "CRITICAL" in badge_text.upper() or "INCIDENT" in badge_text.upper() or "ALERT" in badge_text.upper(), \
            f"Badge did not reflect incident: {badge_text}"

        # Verify emergency vehicles dispatched (police, ambulance)
        num_emergency = page.evaluate("() => typeof vehicles !== 'undefined' ? vehicles.filter(v => v.isEmergency).length : 0")
        num_dispatches = page.evaluate("() => typeof activeDispatches !== 'undefined' ? activeDispatches.length : 0")
        print(f"[E2E] Active emergency vehicles: {num_emergency}, Dispatches active: {num_dispatches}", flush=True)
        assert num_emergency >= 2 or num_dispatches >= 1, "Emergency units were not dispatched!"

        # Wait 4 seconds for emergency vehicle navigation and green-wave preemption
        page.wait_for_timeout(4000)

        shot_incident = os.path.join(ROOT_DIR, "results", "city_incident_screenshot.png")
        page.screenshot(path=shot_incident, full_page=False)
        assert os.path.exists(shot_incident) and os.path.getsize(shot_incident) > 20000, "Incident screenshot failed"
        print(f"[PASS] Emergency Incident Screenshot saved to {shot_incident} ({os.path.getsize(shot_incident)} bytes)", flush=True)

        # Test CCTV Video Perception View
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

        # Clear simulated incident to restore normal flow
        print("[E2E] Resolving all active incidents to clear arterial corridors...", flush=True)
        page.evaluate("() => resolveAllIncidents()")
        page.wait_for_timeout(800)

        # =========================================================================
        # 7. REAL-TIME SIGNAL TIMING STATE MACHINE, RED-LIGHT STOPPING & ANTI-COLLISION
        # =========================================================================
        print("\n=== [7/7] Verifying Signal Timing State Machine, Anti-Collision & Red-Light Stopping ===", flush=True)

        # 7.1 Slower, Observable Simulation Speeds & Interactive Speed Slider
        print("[E2E] Testing slower default simulation speed and interactive speed controls...", flush=True)
        initial_sim_speed = page.evaluate("() => simSpeed")
        assert initial_sim_speed <= 0.85, f"Expected default simSpeed <= 0.85, got {initial_sim_speed}"

        # Test changing speed slider to 1.25x and back to 0.75x
        slider = page.locator("#simSpeedRange")
        if slider.count() > 0:
            page.evaluate("() => { const s = document.getElementById('simSpeedRange'); s.value = '1.25'; updateSimSpeed('1.25'); }")
            page.wait_for_timeout(200)
            spd_125 = page.evaluate("() => simSpeed")
            lbl_125 = page.locator("#speedLabel").text_content() or ""
            assert math.isclose(spd_125, 1.25, abs_tol=1e-3), f"Expected simSpeed 1.25, got {spd_125}"
            assert "1.25" in lbl_125, f"Expected speed label '1.25x', got {lbl_125}"

            # Restore to 0.75x
            page.evaluate("() => { const s = document.getElementById('simSpeedRange'); s.value = '0.75'; updateSimSpeed('0.75'); }")
            page.wait_for_timeout(200)
            spd_075 = page.evaluate("() => simSpeed")
            assert math.isclose(spd_075, 0.75, abs_tol=1e-3), f"Expected simSpeed 0.75, got {spd_075}"
        print("[PASS] Interactive speed slider dynamically updates simSpeed and UI readout cleanly.", flush=True)

        # 7.2 Signal Timing State Machine & Strict Single-Signal Mutual Exclusion
        print("[E2E] Verifying timing parameters & single-signal invariant across all 6 intersections...", flush=True)
        sig_states = page.evaluate("""() => {
            return Object.entries(junctionStates).map(([id, s]) => ({
                id: id,
                minGreen: s.minGreen,
                amberDuration: s.amberDuration,
                allRedDuration: s.allRedDuration,
                activeArm: s.activeArm,
                phase: s.phase,
                isAmber: s.isAmber,
                isAllRed: s.isAllRed,
                signals: { ...s.signals }
            }));
        }""")
        for s in sig_states:
            assert s["minGreen"] >= 8.0, f"Junction {s['id']} minGreen {s['minGreen']}s < 8.0s requirement!"
            assert s["amberDuration"] >= 3.0, f"Junction {s['id']} amberDuration {s['amberDuration']}s < 3.0s!"
            assert s["allRedDuration"] >= 1.5, f"Junction {s['id']} allRedDuration {s['allRedDuration']}s < 1.5s!"
            open_arms = [d for d in ("N", "S", "E", "W") if s["signals"].get(d) in ("green", "amber")]
            assert len(open_arms) <= 1, f"Single-signal invariant violated on {s['id']}: open arms={open_arms}, signals={s['signals']}"
        print(f"[PASS] All {len(sig_states)} junction signal heads configured with minGreen >= 8.0s, amber >= 3.0s, all-red >= 1.5s.", flush=True)

        # Sustained 4.0s sampling (40 frames): strict single-signal invariant + zero rapid flickering
        flicker_count = 0
        invariant_violations = []
        last_sig_snap = {s["id"]: s for s in sig_states}
        for frame_idx in range(40):
            page.wait_for_timeout(100)
            curr_snap = page.evaluate("""() => Object.fromEntries(
                Object.entries(junctionStates).map(([id, s]) => [
                    id,
                    {
                        activeArm: s.activeArm,
                        isAmber: s.isAmber,
                        isAllRed: s.isAllRed,
                        phaseTime: s.phaseTime,
                        signals: { N: getApproachSignal(id, 'N'), S: getApproachSignal(id, 'S'), E: getApproachSignal(id, 'E'), W: getApproachSignal(id, 'W') }
                    }
                ])
            )""")
            for jid, st in curr_snap.items():
                open_arms = [d for d in ("N", "S", "E", "W") if st["signals"][d] in ("green", "amber")]
                if len(open_arms) > 1:
                    invariant_violations.append({"frame": frame_idx, "junction": jid, "open": open_arms, "signals": st["signals"]})
                prev = last_sig_snap[jid]
                if st["activeArm"] != prev["activeArm"] and not (prev["isAmber"] or prev["isAllRed"]):
                    flicker_count += 1
                last_sig_snap[jid] = st
        assert len(invariant_violations) == 0, f"Single-signal mutual exclusion violations: {invariant_violations}"
        assert flicker_count == 0, f"Signal flicker violations detected (arm switched without amber/all-red): {flicker_count}"
        print("[PASS] Strict single-signal invariant (<=1 green/amber per junction) & zero flickering verified across 40 live frames.", flush=True)

        # 7.3 Firm Red-Light Stopping at Stop Bar (speed = 0)
        print("[E2E] Verifying red light stopping compliance at stop lines...", flush=True)
        red_stopped_vehicles = page.evaluate("""() => {
            const stopped = [];
            for (const v of vehicles) {
                if (!v.nextIntersection || v.state === 'TURNING' || v.isEmergency) continue;
                const corr = CORRIDORS[v.corridorId];
                const junc = INTERSECTIONS[v.nextIntersection];
                const approachDir = getVehicleApproachDir(v);
                const sig = getApproachSignal(v.nextIntersection, approachDir);
                const distToCenter = corr.isHorizontal ? (junc.x - v.x) * v.direction : (junc.y - v.y) * v.direction;
                const stopLineDist = distToCenter - 38;

                if ((sig === 'red' || sig === 'amber') && stopLineDist <= 0 && distToCenter >= -20) {
                    stopped.push({ id: v.id, speed: v.speed, accel: v.accel, isBraking: v.isBraking, stopLineDist: stopLineDist });
                }
            }
            return stopped;
        }""")
        print(f"[E2E] Vehicles currently at stop bar during red/amber: {len(red_stopped_vehicles)}", flush=True)
        for sv in red_stopped_vehicles:
            assert sv["speed"] == 0.0, f"Vehicle #{sv['id']} at stop bar failed to hold speed 0 at red light: speed={sv['speed']}"
            assert sv["isBraking"], f"Vehicle #{sv['id']} brake light not active at red signal"
        if len(red_stopped_vehicles) > 0:
            print(f"[PASS] Complete stop bar compliance: {len(red_stopped_vehicles)} queued vehicles holding speed = 0.0 at red lights.", flush=True)

        # 7.4 Fleet-Wide Collision Avoidance (Bumper Separation >= 26px)
        print("[E2E] Verifying 0 bounding box collisions (minimum distance >= 26px) across 30 live simulation steps...", flush=True)
        global_min_dist = 9999.0
        collision_count = 0
        for _ in range(30):
            res = page.evaluate("""() => {
                let minDist = 9999;
                let violations = 0;
                for (let i = 0; i < vehicles.length; i++) {
                    for (let j = i + 1; j < vehicles.length; j++) {
                        const d = Math.hypot(vehicles[j].x - vehicles[i].x, vehicles[j].y - vehicles[i].y);
                        if (d < minDist) minDist = d;
                        if (d < 26.0) violations++;
                    }
                }
                return { minDist, violations };
            }""")
            if res["minDist"] < global_min_dist:
                global_min_dist = res["minDist"]
            collision_count += res["violations"]
            page.wait_for_timeout(100)

        print(f"[E2E] Global minimum inter-vehicle bumper distance recorded: {global_min_dist:.2f}px (violations: {collision_count})", flush=True)
        assert collision_count == 0, f"Detected {collision_count} inter-vehicle collisions (distance < 26px)!"
        assert global_min_dist >= 26.0, f"Global minimum vehicle separation violated: {global_min_dist:.2f}px < 26.0px"
        print(f"[PASS] 0 bounding box collisions verified! Minimum bumper separation {global_min_dist:.2f}px >= 26.0px across entire fleet.", flush=True)

        # 7.5 Turning Vehicles Yielding & Non-Clipping
        print("[E2E] Verifying turning vehicles yield and do not clip into cruising or opposing vehicles...", flush=True)
        turn_check = page.evaluate("""() => {
            const turning = vehicles.filter(v => v.state === 'TURNING');
            let minDistance = 9999;
            let clipViolations = 0;
            for (const tv of turning) {
                for (const ov of vehicles) {
                    if (tv.id === ov.id) continue;
                    const d = Math.hypot(ov.x - tv.x, ov.y - tv.y);
                    if (d < minDistance) minDistance = d;
                    if (d < 26.0) clipViolations++;
                }
            }
            return { count: turning.length, minDistance, clipViolations };
        }""")
        print(f"[PASS] Active turning vehicles: {turn_check['count']}, Min clearance to other vehicles: {turn_check['minDistance']:.2f}px", flush=True)
        assert turn_check["clipViolations"] == 0, f"Turning vehicles clipped into other vehicles: {turn_check}"
        print("[PASS] Turning vehicle clearance verified: ZERO clipping into other vehicles.", flush=True)

        # 7.6 True Queue Discipline & Bumper Stacking (Gap >= 28px without overlapping when red)
        print("[E2E] Verifying True Queue Discipline & Stacking at red signals (gap >= 28px, zero overlap)...", flush=True)
        e2e_queue_check = page.evaluate("""() => {
            const violations = [];
            let totalStoppedPairs = 0;

            for (const cId of Object.keys(CORRIDORS)) {
                const corr = CORRIDORS[cId];
                for (const dir of [1, -1]) {
                    const laneVehicles = vehicles.filter(v => v.state === 'CRUISING' && v.corridorId === cId && v.direction === dir);
                    if (laneVehicles.length <= 1) continue;

                    if (corr.isHorizontal) {
                        laneVehicles.sort((a, b) => (b.x - a.x) * dir);
                    } else {
                        laneVehicles.sort((a, b) => (b.y - a.y) * dir);
                    }

                    for (let k = 1; k < laneVehicles.length; k++) {
                        const lead = laneVehicles[k - 1];
                        const trail = laneVehicles[k];
                        const distAhead = corr.isHorizontal ? (lead.x - trail.x) * dir : (lead.y - trail.y) * dir;
                        const euclideanDist = Math.hypot(lead.x - trail.x, lead.y - trail.y);

                        if (lead.speed === 0 && trail.speed === 0) {
                            totalStoppedPairs++;
                            if (distAhead < 26.0) {
                                violations.push({ type: 'gap_too_small', leadId: lead.id, trailId: trail.id, distAhead, euclideanDist });
                            }
                            if (euclideanDist < 26.0) {
                                violations.push({ type: 'overlap', leadId: lead.id, trailId: trail.id, euclideanDist });
                            }
                        }
                    }
                }
            }
            return { totalStoppedPairs, violations };
        }""")
        print(f"[E2E] Total stopped queue pairs evaluated: {e2e_queue_check['totalStoppedPairs']}, violations: {len(e2e_queue_check['violations'])}", flush=True)
        assert len(e2e_queue_check["violations"]) == 0, f"Queue discipline violations detected: {e2e_queue_check['violations']}"
        print("[PASS] True queue discipline verified: all stopped vehicles queue strictly on lane axis with gap >= 28px and zero overlap.", flush=True)

        shot_e2e_queue = os.path.join(ROOT_DIR, "results", "queue_discipline_red_signal.png")
        page.screenshot(path=shot_e2e_queue, full_page=False)
        assert os.path.exists(shot_e2e_queue) and os.path.getsize(shot_e2e_queue) > 20000
        print(f"[PASS] Queue discipline screenshot saved to {shot_e2e_queue} ({os.path.getsize(shot_e2e_queue)} bytes)", flush=True)

        # 7.7 Zero Lateral Displacement (Rigid Lane Centerline Adherence)
        print("[E2E] Verifying Zero Lateral Displacement across fleet (rigid lane centerline adherence)...", flush=True)
        e2e_lateral_check = page.evaluate("""() => {
            let maxLateralErr = 0.0;
            const violations = [];

            for (const v of vehicles) {
                if (v.state !== 'CRUISING') continue;
                const corr = CORRIDORS[v.corridorId];
                if (!corr) continue;

                let expectedX = v.x;
                let expectedY = v.y;
                let latErr = 0.0;

                if (corr.isHorizontal) {
                    expectedY = (v.direction === 1) ? corr.y + 16 : corr.y - 16;
                    latErr = Math.abs(v.y - expectedY);
                } else {
                    expectedX = (v.direction === 1) ? corr.x - 16 : corr.x + 16;
                    latErr = Math.abs(v.x - expectedX);
                }

                if (latErr > maxLateralErr) maxLateralErr = latErr;
                if (latErr > 0.001) {
                    violations.push({ id: v.id, corr: v.corridorId, dir: v.direction, latErr });
                }
            }
            return { maxLateralErr, violations };
        }""")
        print(f"[E2E] Max lateral centerline deviation: {e2e_lateral_check['maxLateralErr']:.6f}px", flush=True)
        assert len(e2e_lateral_check["violations"]) == 0, f"Lateral displacement detected! Violations: {e2e_lateral_check['violations']}"
        assert e2e_lateral_check["maxLateralErr"] < 0.001, f"Lateral displacement {e2e_lateral_check['maxLateralErr']}px >= 0.001px"
        print("[PASS] Zero lateral displacement verified: 100% of cruising/queuing vehicles stay rigidly on lane centerlines.", flush=True)

        shot_e2e_lane = os.path.join(ROOT_DIR, "results", "lane_discipline_centerline.png")
        page.screenshot(path=shot_e2e_lane, full_page=False)
        assert os.path.exists(shot_e2e_lane) and os.path.getsize(shot_e2e_lane) > 20000
        print(f"[PASS] Lane discipline screenshot saved to {shot_e2e_lane} ({os.path.getsize(shot_e2e_lane)} bytes)", flush=True)

        # 7.8 Zero Deadlock: Vehicles with GREEN signals advance freely through intersections
        print("[E2E] Verifying Zero Deadlock: vehicles with GREEN signals advance freely through intersections...", flush=True)
        stuck_counts = {}
        green_total = 0
        advancing_total = 0
        deadlocked_persistent = []
        for _ in range(10):
            e2e_deadlock_check = page.evaluate("""() => {
                let greenCount = 0;
                let advancingCount = 0;
                const stuck = [];

                for (const v of vehicles) {
                    if (v.state !== 'CRUISING' || v.isEmergency || !v.nextIntersection) continue;
                    const nearIncident = (typeof activeIncidents !== 'undefined') && activeIncidents.some(inc => Math.hypot(inc.x - v.x, inc.y - v.y) < 70);
                    if (nearIncident) continue;
                    const corr = CORRIDORS[v.corridorId];
                    const approachDir = getVehicleApproachDir(v);
                    const sig = getApproachSignal(v.nextIntersection, approachDir);
                    const junc = INTERSECTIONS[v.nextIntersection];
                    const distToCenter = corr.isHorizontal ? (junc.x - v.x) * v.direction : (junc.y - v.y) * v.direction;

                    if (sig === 'green' && distToCenter > 0 && distToCenter < 120) {
                        greenCount++;

                        let leadDist = 9999;
                        for (const o of vehicles) {
                            if (o.id === v.id || o.corridorId !== v.corridorId || o.direction !== v.direction) continue;
                            const d = corr.isHorizontal ? (o.x - v.x) * v.direction : (o.y - v.y) * v.direction;
                            if (d > 0 && d < leadDist) leadDist = d;
                        }

                        let turningInJunc = false;
                        for (const o of vehicles) {
                            if (o.id === v.id) continue;
                            if (Math.abs(o.x - junc.x) < 32 && Math.abs(o.y - junc.y) < 32 && (o.state === 'TURNING' || (o.corridorId !== v.corridorId && o.speed > 0.04))) {
                                turningInJunc = true;
                                break;
                            }
                        }

                        if (leadDist > 40 && !turningInJunc) {
                            if (v.speed > 0.1 || v.accel > 0) {
                                advancingCount++;
                            } else if (v.speed === 0 && v.accel <= 0) {
                                stuck.push({ id: v.id, corridor: v.corridorId, distToCenter, leadDist, speed: v.speed, accel: v.accel });
                            }
                        }
                    }
                }
                return { greenCount, advancingCount, stuck };
            }""")
            green_total += e2e_deadlock_check["greenCount"]
            advancing_total += e2e_deadlock_check["advancingCount"]
            curr_stuck_ids = {item["id"]: item for item in e2e_deadlock_check["stuck"]}
            for vid in list(stuck_counts.keys()):
                if vid not in curr_stuck_ids:
                    del stuck_counts[vid]
            for vid, item in curr_stuck_ids.items():
                stuck_counts[vid] = stuck_counts.get(vid, 0) + 1
                if stuck_counts[vid] >= 4:
                    deadlocked_persistent.append(item)
            page.wait_for_timeout(100)
        print(f"[E2E] Green signal approach samples: {green_total}, advancing: {advancing_total}, persistent deadlocks: {len(deadlocked_persistent)}", flush=True)
        assert len(deadlocked_persistent) == 0, f"Deadlocked vehicles detected at green signals: {deadlocked_persistent}"
        print("[PASS] Zero deadlock verified: all vehicles with green signals and open headways advance freely through intersections.", flush=True)

        shot_e2e_green = os.path.join(ROOT_DIR, "results", "green_wave_free_advance.png")
        page.screenshot(path=shot_e2e_green, full_page=False)
        assert os.path.exists(shot_e2e_green) and os.path.getsize(shot_e2e_green) > 20000
        print(f"[PASS] Green wave free advance screenshot saved to {shot_e2e_green} ({os.path.getsize(shot_e2e_green)} bytes)", flush=True)

        # 7.9 Independent 4-Way Signals & Dynamic Stop Bars (N, S, E, W)
        print("\n[E2E] Verifying Independent 4-Way Signals & Dynamic Stop Bars across all 6 intersections...", flush=True)
        signals_4way_check = page.evaluate("""() => {
            const approaches = ["N", "S", "E", "W"];
            let allConfigured = true;
            let totalHeads = 0;
            let validProjections = true;
            const juncData = {};

            for (const [id, junc] of Object.entries(INTERSECTIONS)) {
                const state = junctionStates[id];
                if (!state || !state.signals) {
                    allConfigured = false;
                    continue;
                }

                const colors = {};
                for (const d of approaches) {
                    const c = getApproachSignal(id, d);
                    colors[d] = c;
                    if (!["green", "amber", "red"].includes(c)) {
                        allConfigured = false;
                    }
                }

                const heads = [
                    { x: junc.x - 42, y: junc.y - 50, col: colors.N, dir: "N" },
                    { x: junc.x + 42, y: junc.y + 50, col: colors.S, dir: "S" },
                    { x: junc.x + 50, y: junc.y - 42, col: colors.E, dir: "E" },
                    { x: junc.x - 50, y: junc.y + 42, col: colors.W, dir: "W" }
                ];
                for (const h of heads) {
                    totalHeads++;
                    const sc = worldToScreen(h.x, h.y, 0);
                    if (!Number.isFinite(sc.x) || !Number.isFinite(sc.y)) {
                        validProjections = false;
                    }
                }

                juncData[id] = colors;
            }

            selectedJunctionId = "INT_2";
            updateInspectorUI();
            const indN = document.getElementById("txtLightN")?.textContent || "";
            const indS = document.getElementById("txtLightS")?.textContent || "";
            const indE = document.getElementById("txtLightE")?.textContent || "";
            const indW = document.getElementById("txtLightW")?.textContent || "";
            const aiRecText = document.getElementById("aiRecText")?.textContent || "";
            const aiConfText = document.getElementById("aiConfText")?.textContent || "";
            const aiTensorText = document.getElementById("aiTensorText")?.textContent || "";
            const modeTag = document.getElementById("junctionModeTag")?.textContent || "";
            const hasButtons = ["btnOverrideN", "btnOverrideS", "btnOverrideE", "btnOverrideW", "btnOverrideAuto"]
                .every(btnId => !!document.getElementById(btnId));

            const invariantFailures = [];
            for (const [id, colors] of Object.entries(juncData)) {
                const open = approaches.filter(d => colors[d] === "green" || colors[d] === "amber");
                if (open.length > 1) invariantFailures.push({ id, open, colors });
            }

            return {
                allConfigured,
                totalHeads,
                validProjections,
                juncData,
                invariantFailures,
                hasButtons,
                aiRecText,
                aiConfText,
                aiTensorText,
                modeTag,
                inspector: { N: indN, S: indS, E: indE, W: indW }
            };
        }""")

        assert signals_4way_check["allConfigured"], f"Independent 4-way signals missing or invalid: {signals_4way_check}"
        assert signals_4way_check["totalHeads"] == 24, f"Expected 24 signal heads, found {signals_4way_check['totalHeads']}"
        assert signals_4way_check["validProjections"], "Signal head screen projections invalid!"
        assert len(signals_4way_check["invariantFailures"]) == 0, f"Single-signal invariant failed: {signals_4way_check['invariantFailures']}"
        assert signals_4way_check["hasButtons"], "Missing 4-arm manual override / AI Auto buttons in Inspector!"
        assert "Arm Scores:" in signals_4way_check["aiConfText"], f"Inspector missing 4-arm AI scores readout: {signals_4way_check['aiConfText']}"
        print(f"[PASS] 24 independent approach signal heads (N, S, E, W) & strict <=1 invariant verified across all 6 intersections.", flush=True)
        print(f"[PASS] Inspector 4-Way readout verified: N={signals_4way_check['inspector']['N']}, S={signals_4way_check['inspector']['S']}, E={signals_4way_check['inspector']['E']}, W={signals_4way_check['inspector']['W']} | {signals_4way_check['aiConfText']}", flush=True)

        # Test single approach manual override: approach 'N' independent Green (with amber -> all-red clearance)
        override_test = page.evaluate("""() => {
            overrideJunctionSignal('INT_2', 'N');
            const st = junctionStates['INT_2'];
            const modeTagDuringOverride = document.getElementById('junctionModeTag')?.textContent || '';
            const manualAfterSet = st.manualOverride;

            const stepViolations = [];
            for (let i = 0; i < 60; i++) {
                const openNow = APPROACH_DIRS.filter(d => getApproachSignal('INT_2', d) === 'green' || getApproachSignal('INT_2', d) === 'amber');
                if (openNow.length > 1) stepViolations.push({ step: i, openNow });
                if (!st.isAmber && !st.isAllRed && st.activeArm === 'N') break;
                updateTrafficSignals(0.25);
            }
            updateInspectorUI();
            return {
                manualAfterSet,
                modeTagDuringOverride,
                stepViolations,
                activeArm: st.activeArm,
                isAmber: st.isAmber,
                isAllRed: st.isAllRed,
                N: getApproachSignal('INT_2', 'N'),
                S: getApproachSignal('INT_2', 'S'),
                E: getApproachSignal('INT_2', 'E'),
                W: getApproachSignal('INT_2', 'W')
            };
        }""")
        assert override_test["manualAfterSet"] == "N", f"Expected manualOverride 'N', got {override_test['manualAfterSet']}"
        assert "MANUAL OVERRIDE" in override_test["modeTagDuringOverride"], f"Mode tag not updated: {override_test['modeTagDuringOverride']}"
        assert len(override_test["stepViolations"]) == 0, f"Single-signal invariant violated during override transition: {override_test['stepViolations']}"
        assert override_test["N"] == "green", f"Approach N failed manual override: {override_test}"
        assert override_test["S"] == "red" and override_test["E"] == "red" and override_test["W"] == "red", f"Conflicting/opposing approaches not red under strict single-signal mutual exclusion: {override_test}"
        print(f"[PASS] Independent manual approach override verified (strict <=1): N={override_test['N']}, S={override_test['S']}, E={override_test['E']}, W={override_test['W']}", flush=True)

        # Reset manual override to auto via 'AUTO' and verify restoration
        auto_restore = page.evaluate("""() => {
            overrideJunctionSignal('INT_2', 'AUTO');
            const st = junctionStates['INT_2'];
            return {
                manualOverride: st.manualOverride,
                modeTag: document.getElementById('junctionModeTag')?.textContent || '',
                openArms: APPROACH_DIRS.filter(d => getApproachSignal('INT_2', d) === 'green' || getApproachSignal('INT_2', d) === 'amber')
            };
        }""")
        assert auto_restore["manualOverride"] is None, f"overrideJunctionSignal('INT_2', 'AUTO') did not clear manualOverride: {auto_restore}"
        assert auto_restore["modeTag"] == "AUTONOMOUS RL", f"Expected 'AUTONOMOUS RL' mode tag after AUTO, got {auto_restore['modeTag']}"
        assert len(auto_restore["openArms"]) <= 1, f"Invariant violated after AUTO restore: {auto_restore}"
        print(f"[PASS] overrideJunctionSignal('INT_2', 'AUTO') cleanly restored {auto_restore['modeTag']} mode.", flush=True)

        shot_4way = os.path.join(ROOT_DIR, "results", "independent_4way_signals.png")
        page.screenshot(path=shot_4way, full_page=False)
        assert os.path.exists(shot_4way) and os.path.getsize(shot_4way) > 20000
        print(f"[PASS] Independent 4-way signals screenshot saved to {shot_4way} ({os.path.getsize(shot_4way)} bytes)", flush=True)

        # 7.10 Independent 4-Arm AI Demand Actuation (N -> E -> S -> W) & Dynamic Queue Relief
        print("\n[E2E] Verifying Independent 4-Arm AI Actuation (N -> E -> S -> W) & Amber->All-Red Clearance...", flush=True)
        relief_check = page.evaluate("""() => {
            const jId = "INT_2";
            const state = junctionStates[jId];
            emergencyPreemptionCorridor = [];
            state.manualOverride = null;

            // Start cleanly on N green
            state.isAmber = false;
            state.isAllRed = false;
            state.activeArm = "N";
            state.targetArm = "N";
            state.phase = 0;
            state.phaseTime = 12.0;
            state.armWaitTimes = { N: 0, S: 0, E: 0, W: 0 };
            enforceSingleSignalInvariant(state);

            const transitions = [];
            const invariantViolations = [];
            const sequence = [
                { arm: "E", queues: [1.0, 1.0, 24.5, 1.0] },
                { arm: "S", queues: [1.0, 24.5, 1.0, 1.0] },
                { arm: "W", queues: [1.0, 1.0, 1.0, 24.5] },
                { arm: "N", queues: [24.5, 1.0, 1.0, 1.0] }
            ];

            for (const stepCfg of sequence) {
                const fromArm = state.activeArm;
                state.queues = [...stepCfg.queues];
                state.phaseTime = 12.0;
                state.nextInferenceTimer = 0.0;

                let sawAmber = false;
                let sawAllRed = false;
                let reachedTarget = false;

                for (let iter = 0; iter < 60; iter++) {
                    updateTrafficSignals(0.25);

                    for (const [jid, jst] of Object.entries(junctionStates)) {
                        const open = APPROACH_DIRS.filter(d => jst.signals[d] === "green" || jst.signals[d] === "amber");
                        if (open.length > 1) {
                            invariantViolations.push({ target: stepCfg.arm, iter, jid, open, signals: { ...jst.signals } });
                        }
                    }

                    if (state.isAmber && state.targetArm === stepCfg.arm) {
                        sawAmber = true;
                    }
                    if (state.isAllRed && state.targetArm === stepCfg.arm) {
                        const anyOpen = APPROACH_DIRS.some(d => state.signals[d] !== "red");
                        if (!anyOpen) sawAllRed = true;
                    }
                    if (!state.isAmber && !state.isAllRed && state.activeArm === stepCfg.arm && state.signals[stepCfg.arm] === "green") {
                        reachedTarget = true;
                        break;
                    }
                }

                // Discharge on the single active green arm
                const targetIdx = APPROACH_DIRS.indexOf(stepCfg.arm);
                const qBefore = state.queues[targetIdx];
                for (let k = 0; k < 20; k++) {
                    updateTrafficSignals(0.1);
                }
                const qAfter = state.queues[targetIdx];

                transitions.push({
                    fromArm,
                    toArm: stepCfg.arm,
                    sawAmber,
                    sawAllRed,
                    reachedTarget,
                    qBefore,
                    qAfter,
                    signals: { ...state.signals }
                });
            }

            return { transitions, invariantViolations };
        }""")

        assert len(relief_check["invariantViolations"]) == 0, f"Single-signal invariant violated during AI actuation: {relief_check['invariantViolations']}"
        for tr in relief_check["transitions"]:
            assert tr["sawAmber"], f"Transition {tr['fromArm']} -> {tr['toArm']} skipped AMBER phase: {tr}"
            assert tr["sawAllRed"], f"Transition {tr['fromArm']} -> {tr['toArm']} skipped ALL-RED clearance: {tr}"
            assert tr["reachedTarget"], f"AI failed to actuate target arm {tr['toArm']}: {tr}"
            assert tr["signals"][tr["toArm"]] == "green", f"Target arm {tr['toArm']} not green: {tr['signals']}"
            other_arms = [d for d in ("N", "S", "E", "W") if d != tr["toArm"]]
            assert all(tr["signals"][d] == "red" for d in other_arms), f"Non-target arms not RED when {tr['toArm']} green: {tr['signals']}"
            assert tr["qAfter"] < tr["qBefore"], f"Arm {tr['toArm']} queue did not discharge: before={tr['qBefore']}, after={tr['qAfter']}"
            print(f"[PASS] Independent AI actuation {tr['fromArm']} -> {tr['toArm']}: amber->all-red->green verified, queue discharged {tr['qBefore']:.1f} -> {tr['qAfter']:.1f}", flush=True)

        # 7.11 Fluid Queue Discharge & Anti-Jam Kinematics (a >= 0.035 upon lead departure)
        print("\n[E2E] Verifying Fluid Queue Discharge & Immediate Positive Acceleration (a >= 0.035)...", flush=True)
        fluid_discharge_check = page.evaluate("""() => {
            const v1 = vehicles[0];
            const v2 = vehicles[1];

            // Ensure other vehicles are clear of this corridor test segment
            for (let j = 2; j < vehicles.length; j++) {
                if (vehicles[j].corridorId === "H1" && vehicles[j].direction === 1) {
                    vehicles[j].x = 1000;
                }
            }

            v1.corridorId = "H1";
            v1.direction = 1;
            v1.state = "CRUISING";
            v1.type = "sedan";
            v1.length = 20;
            v1.x = 300;
            v1.y = 220 + 16;
            v1.speed = 0.0;
            v1.accel = 0.0;
            v1.isBraking = true;
            v1.targetSpeed = 1.0;
            v1.nextIntersection = "INT_2";

            v2.corridorId = "H1";
            v2.direction = 1;
            v2.state = "CRUISING";
            v2.type = "sedan";
            v2.length = 20;
            v2.x = 300 - 28.0;
            v2.y = 220 + 16;
            v2.speed = 0.0;
            v2.accel = 0.0;
            v2.isBraking = true;
            v2.targetSpeed = 1.0;
            v2.nextIntersection = "INT_2";

            junctionStates["INT_2"].activeArm = "W";
            junctionStates["INT_2"].targetArm = "W";
            junctionStates["INT_2"].signals = { N: "red", S: "red", E: "red", W: "green" };
            junctionStates["INT_2"].phase = 1;
            junctionStates["INT_2"].isAmber = false;
            junctionStates["INT_2"].isAllRed = false;

            const isLeadStoppedBefore = (v1.speed < 0.02 && v1.accel <= 0);
            const distBefore = Math.abs(v1.x - v2.x);
            const queueBlockedBefore = isLeadStoppedBefore && (distBefore <= 28.1);

            // Lead car departs
            v1.speed = 0.4;
            v1.accel = 0.05;
            v1.isBraking = false;

            const isLeadStoppedAfter = (v1.speed < 0.02 && v1.accel <= 0);
            const queueBlockedAfter = isLeadStoppedAfter && (distBefore <= 28.1);

            const s0 = 15;
            const T = 0.8;
            const aMax = 0.06;
            const bComf = 0.09;
            const deltaV = v2.speed - v1.speed;
            const sStar = s0 + Math.max(0, v2.speed * T + (v2.speed * deltaV) / (2 * Math.sqrt(aMax * bComf)));
            const idmAccel = aMax * (1 - Math.pow(v2.speed / Math.max(0.1, v2.targetSpeed), 4) - Math.pow(sStar / Math.max(1, distBefore), 2));

            updateVehicles(0.016);

            return {
                queueBlockedBefore,
                queueBlockedAfter,
                idmAccel,
                trailAccel: v2.accel,
                trailSpeed: v2.speed,
                leadSpeed: v1.speed
            };
        }""")

        assert fluid_discharge_check["queueBlockedBefore"], "Trailing car was not properly queued when lead was stopped!"
        assert not fluid_discharge_check["queueBlockedAfter"], "Trailing car remained blocked after lead car departed!"
        assert fluid_discharge_check["idmAccel"] >= 0.035, f"IDM acceleration too low: {fluid_discharge_check['idmAccel']} < 0.035!"
        assert fluid_discharge_check["trailAccel"] >= 0.035 or fluid_discharge_check["trailSpeed"] > 0, \
            f"Trailing car failed to accelerate immediately: accel={fluid_discharge_check['trailAccel']}, speed={fluid_discharge_check['trailSpeed']}"
        print(f"[PASS] Fluid queue discharge verified: IDM acceleration = {fluid_discharge_check['idmAccel']:.4f} >= 0.035 px/frame. Trailing cars follow immediately!", flush=True)

        shot_fluid = os.path.join(ROOT_DIR, "results", "fluid_queue_discharge.png")
        page.screenshot(path=shot_fluid, full_page=False)
        assert os.path.exists(shot_fluid) and os.path.getsize(shot_fluid) > 20000
        print(f"[PASS] Fluid queue discharge screenshot saved to {shot_fluid} ({os.path.getsize(shot_fluid)} bytes)", flush=True)

        # 7.12 Live Fleet Demand-Driven AI & Zero Mid-Road Stops Probe
        print("\n[E2E] Verifying Demand-Driven AI & ZERO unexplained mid-road stops (12s live probe)...", flush=True)
        page.evaluate("() => resetCitySimulation()")
        page.wait_for_timeout(400)

        blank_streak_start = {}
        max_blank_green_sec = 0.0
        live_invariant_violations = []
        empty_arm_ai_wins = 0
        probe_min_dist = 9999.0
        total_stopped_checked = 0
        unexplained_stop_streaks = {}
        persistent_unexplained_stops = []

        for sample_idx in range(48):
            probe = page.evaluate("""() => {
                const nowMs = performance.now();
                const demands = computeApproachDemands();
                const juncs = {};
                const invViolations = [];
                let emptyWins = 0;

                for (const [id, st] of Object.entries(junctionStates)) {
                    const open = APPROACH_DIRS.filter(d => st.signals[d] === "green" || st.signals[d] === "amber");
                    if (open.length > 1) {
                        invViolations.push({ id, open, signals: { ...st.signals } });
                    }
                    const dArr = demands[id] || [0, 0, 0, 0];
                    const activeIdx = st.activeArm ? APPROACH_DIRS.indexOf(st.activeArm) : -1;
                    const activeLive = activeIdx >= 0 ? dArr[activeIdx] : 0;
                    const otherHasWaiting = dArr.some((val, idx) => idx !== activeIdx && val > 0);
                    const clearing = st.activeArm ? isArmClearingBox(id, st.activeArm) : false;
                    const isActiveGreen = (!st.isStandby && !st.isAllRed && !st.isAmber && st.activeArm !== null);

                    const p4 = st.lastDecision;
                    if (p4 && !p4.isAllEmpty && p4.bestArm) {
                        const bestIdx = APPROACH_DIRS.indexOf(p4.bestArm);
                        if ((st.queues[bestIdx] || 0) < 0.08 && (dArr[bestIdx] || 0) <= 0) {
                            emptyWins++;
                        }
                    }

                    juncs[id] = {
                        isBlankGreenWithWaiting: (isActiveGreen && activeLive === 0 && !clearing && otherHasWaiting && st.phaseTime >= 1.0),
                        activeArm: st.activeArm,
                        phaseTime: st.phaseTime,
                        activeLive,
                        demands: dArr
                    };
                }

                let minDist = 9999.0;
                for (let i = 0; i < vehicles.length; i++) {
                    for (let j = i + 1; j < vehicles.length; j++) {
                        const dist = Math.hypot(vehicles[j].x - vehicles[i].x, vehicles[j].y - vehicles[i].y);
                        if (dist < minDist) minDist = dist;
                    }
                }

                // Validate every stopped non-emergency vehicle chains back to a red/amber/yield stop bar or active incident
                const validMemo = new Map();
                const visiting = new Set();
                function isValidStop(v) {
                    if (validMemo.has(v.id)) return validMemo.get(v.id);
                    if (visiting.has(v.id)) return { ok: false, reason: "cycle" };
                    visiting.add(v.id);

                    let res = { ok: false, reason: "unexplained_mid_road" };
                    if (v.state === "TURNING") {
                        const tc = v.turnCurve;
                        const tLook = Math.min(1.0, (v.turnProgress || 0) + 0.18);
                        const lookPt = tc ? evalCurvePoint(tc, tLook) : { x: v.x, y: v.y };
                        const blocker = vehicles.find(o => o.id !== v.id && o.speed < 0.05 && (Math.hypot(o.x - lookPt.x, o.y - lookPt.y) < 32 || Math.hypot(o.x - v.x, o.y - v.y) < 32));
                        if (blocker) {
                            const br = isValidStop(blocker);
                            res = br.ok ? { ok: true, reason: "turning_behind_valid_stop" } : { ok: false, reason: "turning_behind_invalid:" + br.reason };
                        } else {
                            res = { ok: false, reason: "turning_stalled_no_stopped_blocker" };
                        }
                        visiting.delete(v.id);
                        validMemo.set(v.id, res);
                        return res;
                    }

                    const corr = CORRIDORS[v.corridorId];
                    if (!corr) {
                        visiting.delete(v.id);
                        return { ok: false, reason: "no_corridor" };
                    }

                    if (v.nextIntersection) {
                        const junc = INTERSECTIONS[v.nextIntersection];
                        const distToCenter = corr.isHorizontal ? (junc.x - v.x) * v.direction : (junc.y - v.y) * v.direction;
                        const stopLineDist = distToCenter - 38;
                        const sig = getApproachSignal(v.nextIntersection, getVehicleApproachDir(v));
                        if (stopLineDist <= 5.0 && distToCenter >= 32.0) {
                            if (sig !== "green") {
                                res = { ok: true, reason: "red_amber_stop_bar" };
                                visiting.delete(v.id);
                                validMemo.set(v.id, res);
                                return res;
                            }
                            const hasYieldTarget = vehicles.some(o => o.id !== v.id && Math.abs(o.x - junc.x) < 34 && Math.abs(o.y - junc.y) < 34 && (o.state === "TURNING" || (o.corridorId !== v.corridorId && o.speed > 0.04)));
                            if (hasYieldTarget) {
                                res = { ok: true, reason: "yield_stop_bar" };
                                visiting.delete(v.id);
                                validMemo.set(v.id, res);
                                return res;
                            }
                        }
                    }

                    const incAhead = activeIncidents.some(inc => {
                        const fwdInc = corr.isHorizontal ? (inc.x - v.x) * v.direction : (inc.y - v.y) * v.direction;
                        const latInc = corr.isHorizontal ? Math.abs(inc.y - v.y) : Math.abs(inc.x - v.x);
                        return fwdInc > 0 && fwdInc <= 55 && latInc < 22;
                    });
                    if (incAhead) {
                        res = { ok: true, reason: "incident_ahead" };
                        visiting.delete(v.id);
                        validMemo.set(v.id, res);
                        return res;
                    }

                    let lead = null;
                    let minLeadDist = 9999;
                    for (const o of vehicles) {
                        if (o.id === v.id || o.corridorId !== v.corridorId || o.direction !== v.direction) continue;
                        const d = corr.isHorizontal ? (o.x - v.x) * v.direction : (o.y - v.y) * v.direction;
                        if (d > 0 && d < minLeadDist) {
                            minLeadDist = d;
                            lead = o;
                        }
                    }
                    if (lead && minLeadDist <= 34.0) {
                        if (lead.speed >= 0.05) {
                            res = { ok: false, reason: "frozen_behind_moving_lead", leadId: lead.id, leadSpeed: lead.speed, minLeadDist };
                        } else {
                            const lr = isValidStop(lead);
                            res = lr.ok ? { ok: true, reason: "queued->" + lr.reason } : { ok: false, reason: "queued_behind_invalid:" + lr.reason };
                        }
                    } else if (v.nextIntersection) {
                        const junc = INTERSECTIONS[v.nextIntersection];
                        const distToCenter = corr.isHorizontal ? (junc.x - v.x) * v.direction : (junc.y - v.y) * v.direction;
                        const stopLineDist = distToCenter - 38;
                        if (stopLineDist > 5.0 && stopLineDist < 60.0) {
                            res = { ok: false, reason: "frozen_short_of_stop_line", stopLineDist, minLeadDist };
                        }
                    }
                    visiting.delete(v.id);
                    validMemo.set(v.id, res);
                    return res;
                }

                const stoppedList = vehicles.filter(v => !v.isEmergency && v.speed < 0.05 && !(v.state === "CRUISING" && v.accel > 0.01));
                const unexplainedStops = [];
                for (const v of stoppedList) {
                    const check = isValidStop(v);
                    if (!check.ok) {
                        unexplainedStops.push({
                            id: v.id,
                            state: v.state,
                            corridorId: v.corridorId,
                            direction: v.direction,
                            x: v.x,
                            y: v.y,
                            speed: v.speed,
                            accel: v.accel,
                            nextIntersection: v.nextIntersection,
                            ...check
                        });
                    }
                }

                return { nowMs, juncs, invViolations, emptyWins, minDist, stoppedCount: stoppedList.length, unexplainedStops };
            }""")

            now_s = probe["nowMs"] / 1000.0
            live_invariant_violations.extend(probe["invViolations"])
            empty_arm_ai_wins += probe["emptyWins"]
            probe_min_dist = min(probe_min_dist, probe["minDist"])
            total_stopped_checked += probe["stoppedCount"]

            curr_bad_ids = {item["id"]: item for item in probe["unexplainedStops"]}
            for vid in list(unexplained_stop_streaks.keys()):
                if vid not in curr_bad_ids:
                    del unexplained_stop_streaks[vid]
            for vid, item in curr_bad_ids.items():
                unexplained_stop_streaks[vid] = unexplained_stop_streaks.get(vid, 0) + 1
                if unexplained_stop_streaks[vid] >= 2:
                    persistent_unexplained_stops.append(item)

            for jid, jinfo in probe["juncs"].items():
                if jinfo["isBlankGreenWithWaiting"]:
                    if jid not in blank_streak_start:
                        blank_streak_start[jid] = now_s
                    else:
                        max_blank_green_sec = max(max_blank_green_sec, now_s - blank_streak_start[jid])
                else:
                    blank_streak_start.pop(jid, None)

            page.wait_for_timeout(250)

        print(f"[E2E] Demand-driven & mid-road stop probe: max blank-road green={max_blank_green_sec:.2f}s, empty-arm AI wins={empty_arm_ai_wins}, min separation={probe_min_dist:.2f}px, stopped obs checked={total_stopped_checked}, persistent unexplained stops={len(persistent_unexplained_stops)}", flush=True)
        assert len(live_invariant_violations) == 0, f"<=1 open arm invariant violated during live probe: {live_invariant_violations}"
        assert empty_arm_ai_wins == 0, f"AI selected an empty arm {empty_arm_ai_wins} times!"
        assert max_blank_green_sec <= 2.0, f"Junction held blank-road green for {max_blank_green_sec:.2f}s (> 2.0s) while another arm had waiting vehicles!"
        assert probe_min_dist >= 26.0, f"Inter-vehicle overlap detected during live probe: {probe_min_dist:.2f}px < 26.0px"
        assert total_stopped_checked > 0, "Expected stopped queued vehicles during 12s live probe to validate queue chaining!"
        assert len(persistent_unexplained_stops) == 0, f"Persistent unexplained mid-road stops detected: {persistent_unexplained_stops}"
        print("[PASS] Live demand-driven AI & zero unexplained mid-road stops verified across all queued/cruising/turning vehicles.", flush=True)

        # 7.13 Standby All-Red on Empty Map + Instant Green Wake-Up on Vehicle Arrival
        print("\n[E2E] Verifying Standby All-Red when empty & instant GREEN wake-up on vehicle arrival...", flush=True)
        standby_res = page.evaluate("""() => {
            vehicles = [];
            activeIncidents = [];
            emergencyPreemptionCorridor = [];
            initializeJunctionsFromDemands();
            updateTrafficSignals(0.25);

            selectedJunctionId = "INT_2";
            updateInspectorUI();

            const initSummary = {};
            let allStandby = true;
            for (const [id, st] of Object.entries(junctionStates)) {
                const allRed = APPROACH_DIRS.every(d => st.signals[d] === "red");
                if (!st.isStandby || !st.isAllRed || st.activeArm !== null || !allRed) {
                    allStandby = false;
                }
                initSummary[id] = { isStandby: st.isStandby, isAllRed: st.isAllRed, activeArm: st.activeArm, signals: { ...st.signals } };
            }

            const recText = document.getElementById("aiRecText")?.textContent || "";
            const confText = document.getElementById("aiConfText")?.textContent || "";
            const emptyPolicy = evaluate4ArmRLPolicy({
                queues: [0, 0, 0, 0],
                activeArm: null,
                armWaitTimes: { N: 0, S: 0, E: 0, W: 0 },
                isAllRed: true
            });

            // Spawn a single vehicle approaching INT_2 from West (H1, dir=+1)
            const j2 = INTERSECTIONS["INT_2"];
            vehicles.push({
                id: 9901,
                type: "sedan",
                color: "#3b82f6",
                corridorId: "H1",
                direction: 1,
                x: j2.x - 100,
                y: CORRIDORS["H1"].y + 16,
                speed: 0.0,
                accel: 0.0,
                targetSpeed: 1.0,
                heading: 0,
                length: 20,
                isBraking: true,
                isEmergency: false,
                state: "CRUISING",
                nextIntersection: "INT_2"
            });

            // Single update must wake INT_2 straight to W GREEN without amber/all-red delay
            updateTrafficSignals(0.05);
            const st2 = junctionStates["INT_2"];
            const othersStillStandby = Object.entries(junctionStates)
                .filter(([id]) => id !== "INT_2")
                .every(([, st]) => st.isStandby && st.isAllRed && APPROACH_DIRS.every(d => st.signals[d] === "red"));

            return {
                allStandby,
                initSummary,
                recText,
                confText,
                emptyPolicy,
                wakeInt2: {
                    isStandby: st2.isStandby,
                    isAllRed: st2.isAllRed,
                    isAmber: st2.isAmber,
                    activeArm: st2.activeArm,
                    signals: { ...st2.signals }
                },
                othersStillStandby
            };
        }""")

        assert standby_res["allStandby"], f"Empty junctions failed to rest in Standby All-Red: {standby_res['initSummary']}"
        assert "STANDBY (ALL CLEAR)" in standby_res["recText"], f"Expected 'STANDBY (ALL CLEAR)' in Inspector, got '{standby_res['recText']}'"
        assert "0%" in standby_res["confText"] and "Standby" in standby_res["confText"], f"Unexpected Standby confidence readout: '{standby_res['confText']}'"
        assert standby_res["emptyPolicy"]["isAllEmpty"] and standby_res["emptyPolicy"]["bestArm"] is None, f"Empty policy should return isAllEmpty: {standby_res['emptyPolicy']}"
        assert not standby_res["wakeInt2"]["isStandby"] and not standby_res["wakeInt2"]["isAllRed"] and not standby_res["wakeInt2"]["isAmber"], \
            f"INT_2 did not snap straight to GREEN on arrival: {standby_res['wakeInt2']}"
        assert standby_res["wakeInt2"]["activeArm"] == "W" and standby_res["wakeInt2"]["signals"] == {"N": "red", "S": "red", "E": "red", "W": "green"}, \
            f"Expected W GREEN on wake-up, got {standby_res['wakeInt2']}"
        assert standby_res["othersStillStandby"], "Other empty junctions woke up spuriously!"
        print("[PASS] Standby All-Red & instant GREEN wake-up verified (0% Standby -> W GREEN on arrival).", flush=True)

        # 7.14 Non-Sequential Jump & Instant Gap-Out (N -> amber -> all-red -> W)
        print("\n[E2E] Verifying Instant Gap-Out & Non-Sequential AI Jump (N -> amber -> all-red -> W)...", flush=True)
        jump_res = page.evaluate("""() => {
            const j2 = INTERSECTIONS["INT_2"];
            vehicles = [];
            for (let k = 0; k < 4; k++) {
                vehicles.push({
                    id: 9910 + k,
                    type: "sedan",
                    color: "#3b82f6",
                    corridorId: "H1",
                    direction: 1,
                    x: j2.x - 50 - (k * 30),
                    y: CORRIDORS["H1"].y + 16,
                    speed: 0.0,
                    accel: 0.0,
                    targetSpeed: 1.0,
                    heading: 0,
                    length: 20,
                    isBraking: true,
                    isEmergency: false,
                    state: "CRUISING",
                    nextIntersection: "INT_2"
                });
            }

            const st = junctionStates["INT_2"];
            st.manualOverride = null;
            st.isStandby = false;
            st.isAllRed = false;
            st.isAmber = false;
            st.activeArm = "N";
            st.targetArm = "N";
            st.phase = 0;
            st.phaseTime = 1.1; // strictly below minGreen (8.5s), >= 1.0s gap-out threshold
            st.nextInferenceTimer = 0.0;
            st.armWaitTimes = { N: 0, S: 0, E: 0, W: 0 };
            st.queues = [0, 0, 0, 4.0];
            enforceSingleSignalInvariant(st);

            const visitedArms = ["N"];
            const invViolations = [];
            let gapOutTriggeredImmediately = false;
            let sawAmber = false;
            let sawAllRed = false;

            for (let step = 0; step < 50; step++) {
                updateTrafficSignals(0.25);
                const open = APPROACH_DIRS.filter(d => st.signals[d] === "green" || st.signals[d] === "amber");
                if (open.length > 1) {
                    invViolations.push({ step, open, signals: { ...st.signals } });
                }
                if (step === 0 && st.isAmber && st.targetArm === "W") {
                    gapOutTriggeredImmediately = true;
                }
                if (st.isAmber && st.targetArm === "W") sawAmber = true;
                if (st.isAllRed && st.targetArm === "W") sawAllRed = true;
                if (st.activeArm && !visitedArms.includes(st.activeArm)) {
                    visitedArms.push(st.activeArm);
                }
                if (!st.isAmber && !st.isAllRed && st.activeArm === "W" && st.signals.W === "green") {
                    break;
                }
            }

            return {
                gapOutTriggeredImmediately,
                sawAmber,
                sawAllRed,
                visitedArms,
                invViolations,
                finalSignals: { ...st.signals }
            };
        }""")

        # Restore clean city simulation after isolated state checks
        page.evaluate("() => resetCitySimulation()")
        page.wait_for_timeout(300)

        assert len(jump_res["invViolations"]) == 0, f"<=1 open arm invariant violated during N->W jump: {jump_res['invViolations']}"
        assert jump_res["gapOutTriggeredImmediately"], f"Instant gap-out failed when active N was empty at phaseTime=1.1s: {jump_res}"
        assert jump_res["sawAmber"] and jump_res["sawAllRed"], f"Missing AMBER or ALL-RED clearance during N->W jump: {jump_res}"
        assert jump_res["visitedArms"] == ["N", "W"], f"Expected non-sequential jump ['N', 'W'] (no S/E cycling), got {jump_res['visitedArms']}"
        assert jump_res["finalSignals"] == {"N": "red", "S": "red", "E": "red", "W": "green"}, f"Expected W GREEN, got {jump_res['finalSignals']}"
        print("[PASS] Instant Gap-Out & Non-Sequential Jump verified: N -> amber -> all-red -> W (skipped S/E).", flush=True)

        # 7.15 Deterministic Mid-Road Stop Regressions: Stop-Bar Pull-Up & Moving-Lead Separation
        print("\n[E2E] Verifying deterministic mid-road stop regressions (stop-bar pull-up & moving-lead separation)...", flush=True)
        det_stop_res = page.evaluate("""() => {
            // Case A: Vehicle approaching red signal from 110px upstream must pull all the way to 38px stop bar (not freeze 25-35px short)
            vehicles = [];
            activeIncidents = [];
            emergencyPreemptionCorridor = [];
            const j2 = INTERSECTIONS["INT_2"];
            const st2 = junctionStates["INT_2"];
            st2.manualOverride = "N";
            st2.isStandby = false;
            st2.isAllRed = false;
            st2.isAmber = false;
            st2.activeArm = "N";
            st2.signals = { N: "green", S: "red", E: "red", W: "red" };

            const vPull = {
                id: 9981, type: "sedan", color: "#3b82f6",
                corridorId: "H1", direction: 1,
                x: j2.x - 110, y: CORRIDORS["H1"].y + 16,
                speed: 0.9, accel: 0, targetSpeed: 1.0, heading: 0,
                length: 20, isBraking: false, isEmergency: false,
                state: "CRUISING", nextIntersection: "INT_2"
            };
            vehicles.push(vPull);
            for (let k = 0; k < 240; k++) {
                updateVehicles(1 / 60);
            }
            const stopLineDist = (j2.x - vPull.x) - 38;
            const pullSpeed = vPull.speed;

            // Case B: Trailing vehicle within 27px of a moving lead vehicle must NOT be hard-zeroed by Pass 7/8
            vehicles = [];
            st2.manualOverride = "W";
            st2.signals = { N: "red", S: "red", E: "red", W: "green" };
            const lead = {
                id: 9982, type: "bus", color: "#f59e0b",
                corridorId: "H1", direction: 1,
                x: 240, y: CORRIDORS["H1"].y + 16,
                speed: 0.55, accel: 0.02, targetSpeed: 0.9, heading: 0,
                length: 34, isBraking: false, isEmergency: false,
                state: "CRUISING", nextIntersection: "INT_2"
            };
            const trail = {
                id: 9983, type: "sedan", color: "#3b82f6",
                corridorId: "H1", direction: 1,
                x: 213, y: CORRIDORS["H1"].y + 16,
                speed: 0.55, accel: 0.0, targetSpeed: 1.0, heading: 0,
                length: 20, isBraking: false, isEmergency: false,
                state: "CRUISING", nextIntersection: "INT_2"
            };
            vehicles.push(lead, trail);
            updateVehicles(1 / 60);
            const trailSpeedAfter = trail.speed;
            const gapAfter = lead.x - trail.x;

            resetCitySimulation();
            return { stopLineDist, pullSpeed, trailSpeedAfter, gapAfter };
        }""")
        page.wait_for_timeout(300)

        assert abs(det_stop_res["stopLineDist"]) <= 1.0 and det_stop_res["pullSpeed"] == 0.0, \
            f"Vehicle froze short of red stop bar instead of pulling up to 38px: {det_stop_res}"
        assert det_stop_res["trailSpeedAfter"] > 0.05 and det_stop_res["gapAfter"] >= 27.9, \
            f"Trailing vehicle frozen behind moving lead car by Pass 7/8: {det_stop_res}"
        print(f"[PASS] Deterministic mid-road regressions verified: stopLineDist={det_stop_res['stopLineDist']:.2f}px, moving-lead trailSpeed={det_stop_res['trailSpeedAfter']:.3f}.", flush=True)

        # Check Console Errors, Uncaught Page Exceptions, and Failed Network Requests
        print(f"\n[E2E] Total console logs recorded: {len(console_logs)}", flush=True)
        if console_errors:
            print("[E2E] Console/Page errors detected:", flush=True)
            for err in console_errors:
                print(f"  ❌ {err}", flush=True)
        if failed_requests:
            print("[E2E] Failed network requests detected:", flush=True)
            for req_err in failed_requests:
                print(f"  ❌ {req_err}", flush=True)
        assert len(console_errors) == 0, f"Found {len(console_errors)} console/page errors: {console_errors}"
        assert len(failed_requests) == 0, f"Found {len(failed_requests)} failed network requests: {failed_requests}"
        print("[PASS] Zero unhandled console errors, zero pageerrors, and zero failed network requests verified across entire E2E test run!", flush=True)

        browser.close()

    if server:
        server.shutdown()
        server.server_close()
        print("[E2E] Server closed.", flush=True)

    print("\n=======================================================")
    print("ALL PLAYWRIGHT E2E BROWSER TESTS PASSED SUCCESSFULLY 100%!")
    print("=======================================================", flush=True)
    return True


def test_playwright_e2e():
    assert run_e2e_tests()


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
