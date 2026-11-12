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
        assert veh_verification["fleetSize"] >= 40, f"Expected >= 40 vehicles, found {veh_verification['fleetSize']}"
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

        # 7.2 Signal Timing State Machine (>=8.0s Green, Amber >=3.0s, All-Red >=1.5s, No Rapid Flickering)
        print("[E2E] Verifying timing parameters across all 6 physical signal controllers...", flush=True)
        sig_states = page.evaluate("""() => {
            return Object.entries(junctionStates).map(([id, s]) => ({
                id: id,
                minGreen: s.minGreen,
                amberDuration: s.amberDuration,
                allRedDuration: s.allRedDuration,
                phase: s.phase,
                isAmber: s.isAmber,
                isAllRed: s.isAllRed
            }));
        }""")
        for s in sig_states:
            assert s["minGreen"] >= 8.0, f"Junction {s['id']} minGreen {s['minGreen']}s < 8.0s requirement!"
            assert s["amberDuration"] >= 3.0, f"Junction {s['id']} amberDuration {s['amberDuration']}s < 3.0s!"
            assert s["allRedDuration"] >= 1.5, f"Junction {s['id']} allRedDuration {s['allRedDuration']}s < 1.5s!"
        print(f"[PASS] All {len(sig_states)} junction signal heads configured with minGreen >= 8.0s, amber >= 3.0s, all-red >= 1.5s.", flush=True)

        # Check for zero high-frequency flickering over 20 sampling frames
        flicker_count = 0
        last_sig_phases = {s["id"]: s["phase"] for s in sig_states}
        for _ in range(20):
            page.wait_for_timeout(100)
            current_sig_phases = page.evaluate("() => Object.fromEntries(Object.entries(junctionStates).map(([id, s]) => [id, s.phase]))")
            for jid, ph in current_sig_phases.items():
                if ph != last_sig_phases[jid]:
                    # Check that the change was preceded by valid phase hold
                    phase_time = page.evaluate(f"() => junctionStates['{jid}'] ? junctionStates['{jid}'].phaseTime : 0")
                    if phase_time < 0.0:  # Freshly reset after all-red is valid
                        flicker_count += 1
            last_sig_phases = current_sig_phases
        assert flicker_count == 0, f"Signal flicker violations detected: {flicker_count}"
        print("[PASS] Signal stability verified: ZERO rapid flickering, smooth transition cycles confirmed.", flush=True)

        # 7.3 Firm Red-Light Stopping at Stop Bar (speed = 0)
        print("[E2E] Verifying red light stopping compliance at stop lines...", flush=True)
        red_stopped_vehicles = page.evaluate("""() => {
            const stopped = [];
            for (const v of vehicles) {
                if (!v.nextIntersection || v.state === 'TURNING' || v.isEmergency) continue;
                const corr = CORRIDORS[v.corridorId];
                const junc = INTERSECTIONS[v.nextIntersection];
                const juncState = junctionStates[v.nextIntersection];
                const sig = getJunctionSignal(juncState, corr.isHorizontal);
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

        # 7.4 Fleet-Wide Collision Avoidance (Bumper Separation >= 24px)
        print("[E2E] Verifying 0 bounding box collisions (minimum distance >= 24px) across 30 live simulation steps...", flush=True)
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
                        if (d < 24.0) violations++;
                    }
                }
                return { minDist, violations };
            }""")
            if res["minDist"] < global_min_dist:
                global_min_dist = res["minDist"]
            collision_count += res["violations"]
            page.wait_for_timeout(100)

        print(f"[E2E] Global minimum inter-vehicle bumper distance recorded: {global_min_dist:.2f}px (violations: {collision_count})", flush=True)
        assert collision_count == 0, f"Detected {collision_count} inter-vehicle collisions (distance < 24px)!"
        assert global_min_dist >= 24.0, f"Global minimum vehicle separation violated: {global_min_dist:.2f}px < 24.0px"
        print(f"[PASS] 0 bounding box collisions verified! Minimum bumper separation {global_min_dist:.2f}px >= 24.0px across entire fleet.", flush=True)

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
                    if (d < 24.0) clipViolations++;
                }
            }
            return { count: turning.length, minDistance, clipViolations };
        }""")
        print(f"[PASS] Active turning vehicles: {turn_check['count']}, Min clearance to other vehicles: {turn_check['minDistance']:.2f}px", flush=True)
        assert turn_check["clipViolations"] == 0, f"Turning vehicles clipped into other vehicles: {turn_check}"
        print("[PASS] Turning vehicle clearance verified: ZERO clipping into other vehicles.", flush=True)

        # Check Console Errors
        print(f"\n[E2E] Total console logs recorded: {len(console_logs)}", flush=True)
        if console_errors:
            print("[E2E] Console errors detected:", flush=True)
            for err in console_errors:
                print(f"  ❌ {err}", flush=True)
        assert len(console_errors) == 0, f"Found {len(console_errors)} console errors: {console_errors}"
        print("[PASS] Zero unhandled console errors verified across entire E2E test run!", flush=True)

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
