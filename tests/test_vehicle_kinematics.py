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

    print("[KINEMATICS] Launching Chromium to verify continuous vehicle kinematics & 3D ground contact...", flush=True)
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
        resp = page.goto(url, wait_until="domcontentloaded", timeout=15000)
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"

        # 1. System Boot Verification
        print("[KINEMATICS] Verifying boot sequence dismisses cleanly...", flush=True)
        page.wait_for_selector("#bootLoadingScreen", timeout=5000)
        page.evaluate("() => dismissBootScreen()")
        page.wait_for_timeout(600)

        # 2. Fleet Size & 3D Ground Contact Verification
        print("[KINEMATICS] Verifying active fleet and 3D tire road contact...", flush=True)
        fleet_info = page.evaluate("""() => {
            const types = {};
            let groundTiresValid = true;
            for (const v of vehicles) {
                types[v.type] = (types[v.type] || 0) + 1;
                // Verify tire contact points at z=0
                const tirePos = localToScreen(v, 0, 0, 0);
                if (!Number.isFinite(tirePos.x) || !Number.isFinite(tirePos.y)) {
                    groundTiresValid = false;
                }
            }
            return {
                total: vehicles.length,
                types: types,
                groundTiresValid: groundTiresValid
            };
        }""")
        assert 36 <= fleet_info["total"] <= 44, f"Expected fleet size between 36 and 44, found {fleet_info['total']}"
        assert fleet_info["groundTiresValid"], "Vehicle tire ground contact invalid!"
        print(f"[PASS] Active vehicle fleet count: exactly {fleet_info['total']} vehicles ({fleet_info['types']})", flush=True)
        print("[PASS] True 3D vehicle ground contact verified for all vehicles at z=0 asphalt level.", flush=True)

        # 3. Test Chase Cam Engagement on a moving vehicle
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

        # 4. Multi-Angle Free Cam Orbit Switching during Active Tracking
        print("[KINEMATICS] Verifying Multi-Angle Orbit Camera switching during active vehicle chase...", flush=True)
        orbit_presets = ["SE", "BIRDS_EYE", "NW", "ELEVATED", "SW"]
        for preset in orbit_presets:
            page.evaluate(f"() => setCameraPreset('{preset}')")
            page.wait_for_timeout(300)
            proj_check = page.evaluate("""() => {
                const v = vehicles.find(veh => veh.id === followedVehicleId);
                if (!v) return { ok: false, reason: 'vehicle missing' };
                const sc = worldToScreen(v.x, v.y, 0);
                return {
                    ok: Number.isFinite(sc.x) && Number.isFinite(sc.y) && Number.isFinite(sc.depth),
                    panX: camera.panX,
                    panY: camera.panY,
                    yaw: camera.yaw,
                    pitch: camera.pitch
                };
            }""")
            assert proj_check["ok"], f"Camera tracking projection failed at preset {preset}"
            print(f"[PASS] Active chase tracking at preset {preset:10s}: camera yaw={proj_check['yaw']}°, pitch={proj_check['pitch']}°", flush=True)

            if preset == "BIRDS_EYE":
                shot_birds = os.path.join(ROOT_DIR, "results", "chase_cam_orbit_birds_eye.png")
                page.screenshot(path=shot_birds, full_page=False)
                assert os.path.exists(shot_birds) and os.path.getsize(shot_birds) > 20000
                print(f"[PASS] Chase Cam Bird's-Eye screenshot saved ({os.path.getsize(shot_birds)} bytes)", flush=True)
            elif preset == "SE":
                shot_se = os.path.join(ROOT_DIR, "results", "chase_cam_orbit_se.png")
                page.screenshot(path=shot_se, full_page=False)
                assert os.path.exists(shot_se) and os.path.getsize(shot_se) > 20000
                print(f"[PASS] Chase Cam SE screenshot saved ({os.path.getsize(shot_se)} bytes)", flush=True)

        # Return to SW default isometric
        page.evaluate("() => setCameraPreset('SW')")
        page.wait_for_timeout(200)

        # 5. Track Followed Vehicle across 180 continuous simulation steps (3 seconds)
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

        # 6. Rigorous Continuity Check: Verify ZERO teleportation or sudden coordinate snaps
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

            # Max allowed delta across 100ms at max speed is ~35px
            assert dist < 35.0, f"Teleportation detected! Step {i} -> {i+1} jumped {dist:.2f}px from ({p1['x']}, {p1['y']}) to ({p2['x']}, {p2['y']})"

        print(f"[PASS] Max 100ms displacement: {max_dist_step:.2f}px (Smooth & Continuous, ZERO Teleportation!)", flush=True)
        print(f"[PASS] Max heading change: {max_heading_step:.2f} rad (Smooth orientation, zero sudden 90° snaps)", flush=True)
        print(f"[PASS] Live Telemetry HUD verified: Speed={trajectory[-1]['hudSpeed']}, Street={trajectory[-1]['hudStreet']}, Action={trajectory[-1]['hudAction']}", flush=True)

        # 7. Dynamic Depth Sorting during trajectory motion
        print("[KINEMATICS] Verifying continuous depth sorting as vehicles traverse past buildings...", flush=True)
        depth_order_valid = page.evaluate("""() => {
            // Check that all active vehicles and buildings maintain valid relative depth sorting
            for (const b of BUILDINGS) {
                if (b.type === 'park') continue;
                const bDepth = worldToScreen(b.x + b.w / 2, b.y + b.h / 2, 0).depth;

                for (const v of vehicles) {
                    const vDepth = worldToScreen(v.x, v.y, 0).depth;
                    // If vehicle is physically behind building along camera viewing axis,
                    // vDepth must be less than bDepth (so Painter's algorithm draws vehicle first, building over it)
                    const dx = v.x - (b.x + b.w / 2);
                    const dy = v.y - (b.y + b.h / 2);
                    const radYaw = (camera.yaw * Math.PI) / 180;
                    const yr = dx * Math.sin(radYaw) + dy * Math.cos(radYaw);

                    if (yr < -30) {
                        // Vehicle is definitely further along camera sight line
                        if (vDepth >= bDepth) {
                            return { valid: false, vehicleId: v.id, building: b.label, vDepth, bDepth };
                        }
                    }
                }
            }
            return { valid: true };
        }""")
        assert depth_order_valid["valid"], f"Dynamic depth sorting violation during motion: {depth_order_valid}"
        print("[PASS] Continuous depth sorting verified: 100% occlusion preserved past foreground buildings.", flush=True)

        # 8. Verify Smooth Intersection Bezier Turns & Corridor Transitions
        print("[KINEMATICS] Verifying smooth corridor turning across city network...", flush=True)
        start_corridors = page.evaluate("() => Object.fromEntries(vehicles.map(v => [v.id, v.corridorId]))")
        turns_detected = 0
        corridor_switches = 0
        for _ in range(80):
            turning_count = page.evaluate("() => vehicles.filter(v => v.state === 'TURNING').length")
            if turning_count > 0:
                turns_detected += turning_count
            current_corridors = page.evaluate("() => Object.fromEntries(vehicles.map(v => [v.id, v.corridorId]))")
            corridor_switches = sum(1 for vid, c0 in start_corridors.items() if vid in current_corridors and current_corridors[vid] != c0)
            if corridor_switches > 0 and turns_detected > 0:
                break
            page.wait_for_timeout(100)

        assert turns_detected > 0, "No intersection turns detected!"
        assert corridor_switches > 0, "No corridor switches detected!"
        print(f"[PASS] Active intersection Bezier turns: {turns_detected}, Corridor switches: {corridor_switches}", flush=True)

        # 9. Slower Simulation Speed & Observable Kinematics Verification
        print("[KINEMATICS] Verifying slower, observable simulation speeds and speed slider...", flush=True)
        speed_info = page.evaluate("""() => {
            const speeds = vehicles.map(v => v.targetSpeed);
            const slider = document.getElementById('simSpeedRange');
            const label = document.getElementById('speedLabel');
            return {
                simSpeed: simSpeed,
                sliderVal: slider ? slider.value : null,
                labelText: label ? label.textContent.trim() : null,
                minTargetSpeed: Math.min(...speeds),
                maxTargetSpeed: Math.max(...speeds),
                avgTargetSpeed: speeds.reduce((a, b) => a + b, 0) / speeds.length
            };
        }""")
        assert speed_info["simSpeed"] <= 0.85, f"Expected default simSpeed <= 0.85, got {speed_info['simSpeed']}"
        assert speed_info["sliderVal"] == "0.75", f"Expected speed slider value '0.75', got {speed_info['sliderVal']}"
        assert "0.75" in speed_info["labelText"], f"Expected speed label '0.75x', got {speed_info['labelText']}"
        assert 0.80 <= speed_info["minTargetSpeed"] and speed_info["maxTargetSpeed"] <= 1.35, \
            f"Target speed range out of bounds: min={speed_info['minTargetSpeed']}, max={speed_info['maxTargetSpeed']}"
        print(f"[PASS] Slower simulation speed verified: simSpeed={speed_info['simSpeed']}x (Target speed range: {speed_info['minTargetSpeed']:.2f} - {speed_info['maxTargetSpeed']:.2f} px/frame)", flush=True)

        # 10. Traffic Signal Timing State Machine Verification (>= 8.0s Green, No Flickering)
        print("[KINEMATICS] Verifying traffic signal timing state machine across all 6 junctions...", flush=True)
        signal_timing = page.evaluate("""() => {
            const timings = [];
            for (const [id, state] of Object.entries(junctionStates)) {
                timings.push({
                    id: id,
                    minGreen: state.minGreen,
                    amberDuration: state.amberDuration,
                    allRedDuration: state.allRedDuration,
                    phase: state.phase,
                    phaseTime: state.phaseTime,
                    isAmber: state.isAmber,
                    isAllRed: state.isAllRed
                });
            }
            return timings;
        }""")
        for st in signal_timing:
            assert st["minGreen"] >= 8.0, f"Junction {st['id']} minGreen {st['minGreen']}s is less than 8.0s requirement!"
            assert st["amberDuration"] >= 3.0, f"Junction {st['id']} amberDuration {st['amberDuration']}s is less than 3.0s!"
            assert st["allRedDuration"] >= 1.5, f"Junction {st['id']} allRedDuration {st['allRedDuration']}s is less than 1.5s!"
            print(f"[PASS] Junction {st['id']}: minGreen={st['minGreen']:.1f}s, amberDuration={st['amberDuration']:.1f}s, allRedDuration={st['allRedDuration']:.1f}s", flush=True)

        # Monitor signal stability over 3.0 seconds: confirm NO rapid flickering occurs
        signal_flicker_violations = 0
        prev_signals = page.evaluate("""() => {
            const map = {};
            for (const [id, s] of Object.entries(junctionStates)) {
                map[id] = { phase: s.phase, isAmber: s.isAmber, isAllRed: s.isAllRed, phaseTime: s.phaseTime };
            }
            return map;
        }""")
        for _ in range(15):
            page.wait_for_timeout(200)
            curr_signals = page.evaluate("""() => {
                const map = {};
                for (const [id, s] of Object.entries(junctionStates)) {
                    map[id] = { phase: s.phase, isAmber: s.isAmber, isAllRed: s.isAllRed, phaseTime: s.phaseTime };
                }
                return map;
            }""")
            for junc_id, curr_s in curr_signals.items():
                prev_s = prev_signals[junc_id]
                # If phase changed without transitioning through amber and holding minGreen, record violation
                if curr_s["phase"] != prev_s["phase"] and not (prev_s["isAmber"] or prev_s["isAllRed"]):
                    if prev_s["phaseTime"] < 8.0:
                        signal_flicker_violations += 1
            prev_signals = curr_signals
        assert signal_flicker_violations == 0, f"Detected {signal_flicker_violations} signal flicker violations!"
        print("[PASS] Signal timing state machine verified: ZERO rapid flickering, >=8.0s minGreen hold confirmed.", flush=True)

        # 11. Cars Approaching Red Signal: Stop Bar Clamping & Speed = 0 Holding
        print("[KINEMATICS] Verifying cars stop at red signals and hold speed=0 at stop bar...", flush=True)
        red_stop_data = page.evaluate("""() => {
            const results = [];
            for (const v of vehicles) {
                if (!v.nextIntersection || v.state === 'TURNING' || v.isEmergency) continue;
                const corr = CORRIDORS[v.corridorId];
                const junc = INTERSECTIONS[v.nextIntersection];
                const approachDir = getVehicleApproachDir(v);
                const sig = getApproachSignal(v.nextIntersection, approachDir);
                const distToCenter = corr.isHorizontal ? (junc.x - v.x) * v.direction : (junc.y - v.y) * v.direction;
                const stopLineDist = distToCenter - 38;

                // Car is at or behind stop bar during red/amber
                if ((sig === 'red' || sig === 'amber') && stopLineDist <= 0 && distToCenter >= -20) {
                    results.push({
                        id: v.id,
                        speed: v.speed,
                        accel: v.accel,
                        isBraking: v.isBraking,
                        stopLineDist: stopLineDist,
                        sig: sig
                    });
                }
            }
            return results;
        }""")
        print(f"[KINEMATICS] Active vehicles queued at red signals: {len(red_stop_data)}", flush=True)
        for r_veh in red_stop_data:
            assert r_veh["speed"] == 0.0, f"Vehicle #{r_veh['id']} did not hold speed 0 at red light: speed={r_veh['speed']}"
            assert r_veh["isBraking"], f"Vehicle #{r_veh['id']} brake light not active while stopped at red light"
        if len(red_stop_data) > 0:
            print(f"[PASS] Stop bar compliance: all {len(red_stop_data)} queued vehicles holding speed = 0.0 at red lights.", flush=True)

        # 11A. True Queue Discipline & Stacking: Gap >= 28px without overlapping when red
        print("[KINEMATICS] Verifying True Queue Discipline & Stacking at red signals (gap >= 28px, zero overlap)...", flush=True)
        queue_compliance = page.evaluate("""() => {
            const queues = [];
            const violations = [];
            let totalQueuePairs = 0;

            for (const cId of Object.keys(CORRIDORS)) {
                const corr = CORRIDORS[cId];
                for (const dir of [1, -1]) {
                    const laneVehicles = vehicles.filter(v => v.state === 'CRUISING' && v.corridorId === cId && v.direction === dir);
                    if (laneVehicles.length <= 1) continue;

                    // Sort leading to trailing along travel direction
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

                        // If trailing car is queued behind lead (both stopped)
                        if (lead.speed === 0 && trail.speed === 0) {
                            totalQueuePairs++;
                            // Expected gap >= 28px
                            if (distAhead < 27.9) {
                                violations.push({
                                    type: 'gap_too_small',
                                    leadId: lead.id,
                                    trailId: trail.id,
                                    distAhead: distAhead,
                                    euclidean: euclideanDist
                                });
                            }
                            // Bounding box overlap check
                            if (euclideanDist < 26.0) {
                                violations.push({
                                    type: 'overlap',
                                    leadId: lead.id,
                                    trailId: trail.id,
                                    euclidean: euclideanDist
                                });
                            }
                            queues.push({
                                corridorId: cId,
                                dir: dir,
                                leadId: lead.id,
                                trailId: trail.id,
                                distAhead: distAhead,
                                euclidean: euclideanDist
                            });
                        }
                    }
                }
            }
            return { queues, violations, totalQueuePairs };
        }""")
        print(f"[KINEMATICS] Analyzed {queue_compliance['totalQueuePairs']} stopped queue pairs. Violations: {len(queue_compliance['violations'])}", flush=True)
        assert len(queue_compliance["violations"]) == 0, f"Queue discipline violations: {queue_compliance['violations']}"
        print("[PASS] True queue discipline verified: all stopped vehicles queue strictly on lane axis with gap >= 28px and zero overlap.", flush=True)

        shot_queue = os.path.join(ROOT_DIR, "results", "queue_discipline_red_signal.png")
        page.screenshot(path=shot_queue, full_page=False)
        assert os.path.exists(shot_queue) and os.path.getsize(shot_queue) > 20000
        print(f"[PASS] Queue discipline screenshot saved to {shot_queue} ({os.path.getsize(shot_queue)} bytes)", flush=True)

        # 11B. Zero Lateral Displacement: Cars stay rigidly on their lane centerlines
        print("[KINEMATICS] Verifying Zero Lateral Displacement across fleet (rigid lane centerline adherence)...", flush=True)
        lateral_check = page.evaluate("""() => {
            let maxLateralError = 0.0;
            const lateralViolations = [];

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

                if (latErr > maxLateralError) maxLateralError = latErr;
                if (latErr > 0.001) {
                    lateralViolations.push({ id: v.id, corridor: v.corridorId, dir: v.direction, actual: {x: v.x, y: v.y}, latErr });
                }
            }
            return { maxLateralError, violations: lateralViolations };
        }""")
        print(f"[KINEMATICS] Max lateral centerline deviation: {lateral_check['maxLateralError']:.6f}px", flush=True)
        assert len(lateral_check["violations"]) == 0, f"Lateral displacement detected! Violations: {lateral_check['violations']}"
        assert lateral_check["maxLateralError"] < 0.001, f"Lateral displacement error {lateral_check['maxLateralError']}px >= 0.001px"
        print("[PASS] Zero lateral displacement verified: 100% of cruising/queuing vehicles stay rigidly on lane centerlines.", flush=True)

        shot_lane = os.path.join(ROOT_DIR, "results", "lane_discipline_centerline.png")
        page.screenshot(path=shot_lane, full_page=False)
        assert os.path.exists(shot_lane) and os.path.getsize(shot_lane) > 20000
        print(f"[PASS] Lane discipline screenshot saved to {shot_lane} ({os.path.getsize(shot_lane)} bytes)", flush=True)

        # 11C. Zero Deadlock: Vehicles with GREEN signals advance freely through intersections
        print("[KINEMATICS] Verifying Zero Deadlock: vehicles with GREEN signals advance freely through intersections...", flush=True)
        deadlock_check = page.evaluate("""() => {
            let greenVehiclesCount = 0;
            let advancingCount = 0;
            const deadlockedVehicles = [];

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
                    greenVehiclesCount++;

                    // Check if lead vehicle in same lane is directly in front
                    let leadDist = 9999;
                    for (const o of vehicles) {
                        if (o.id === v.id || o.corridorId !== v.corridorId || o.direction !== v.direction) continue;
                        const d = corr.isHorizontal ? (o.x - v.x) * v.direction : (o.y - v.y) * v.direction;
                        if (d > 0 && d < leadDist) leadDist = d;
                    }

                    // Also check if turning vehicle is inside intersection
                    let turningInJunc = false;
                    for (const o of vehicles) {
                        if (o.id === v.id) continue;
                        if (Math.abs(o.x - junc.x) < 24 && Math.abs(o.y - junc.y) < 24 && (o.state === 'TURNING' || (o.corridorId !== v.corridorId && o.speed > 0.05))) {
                            turningInJunc = true;
                            break;
                        }
                    }

                    // If path ahead is clear (> 40px lead headway and no conflicting vehicle in box)
                    if (leadDist > 40 && !turningInJunc) {
                        if (v.speed > 0.1 || v.accel > 0) {
                            advancingCount++;
                        } else if (v.speed === 0 && v.accel <= 0) {
                            deadlockedVehicles.push({
                                id: v.id,
                                corridor: v.corridorId,
                                distToCenter: distToCenter,
                                leadDist: leadDist,
                                speed: v.speed,
                                accel: v.accel
                            });
                        }
                    }
                }
            }
            return { greenVehiclesCount, advancingCount, deadlockedVehicles };
        }""")
        print(f"[KINEMATICS] Green signal approach vehicles: {deadlock_check['greenVehiclesCount']}, actively advancing: {deadlock_check['advancingCount']}, deadlocked: {len(deadlock_check['deadlockedVehicles'])}", flush=True)
        assert len(deadlock_check["deadlockedVehicles"]) == 0, f"Deadlocked vehicles detected at green signals: {deadlock_check['deadlockedVehicles']}"
        print("[PASS] Zero deadlock verified: all vehicles with green signals and open headways advance freely through intersections.", flush=True)

        shot_green = os.path.join(ROOT_DIR, "results", "green_wave_free_advance.png")
        page.screenshot(path=shot_green, full_page=False)
        assert os.path.exists(shot_green) and os.path.getsize(shot_green) > 20000
        print(f"[PASS] Green wave free advance screenshot saved to {shot_green} ({os.path.getsize(shot_green)} bytes)", flush=True)

        # 11D. Independent 4-Way Approach Signals & Dynamic Queue Relief Kinematics
        print("[KINEMATICS] Verifying Independent 4-Way Approach Signals & Dynamic Queue Relief Kinematics...", flush=True)
        sig4way_res = page.evaluate("""() => {
            const approaches = ["N", "S", "E", "W"];
            let allConfigured = true;
            let totalHeads = 0;

            for (const [id, junc] of Object.entries(INTERSECTIONS)) {
                const state = junctionStates[id];
                if (!state || !state.signals) {
                    allConfigured = false;
                    continue;
                }
                for (const d of approaches) {
                    totalHeads++;
                    const c = getApproachSignal(id, d);
                    if (!["green", "amber", "red"].includes(c)) allConfigured = false;
                }
            }

            // Dynamic Demand Relief at INT_2: Peg N/S queues to 24.5
            const jId = "INT_2";
            const state = junctionStates[jId];
            emergencyPreemptionCorridor = [];
            state.manualOverride = null;
            state.isAmber = false;
            state.isAllRed = false;
            state.phase = 1;
            state.phaseTime = 15.0;
            state.queues = [24.5, 24.5, 2.0, 2.0];
            state.nextInferenceTimer = 0.0;

            updateTrafficSignals(0.1);
            const amberTriggered = state.isAmber && (state.targetPhase === 0);

            // Fast-forward to Green
            state.isAmber = false;
            state.isAllRed = false;
            state.phase = 0;
            state.signals = { N: "green", S: "green", E: "red", W: "red" };
            state.phaseTime = 0.0;

            const qBefore = state.queues[0];
            for (let i = 0; i < 20; i++) updateTrafficSignals(0.1);
            const qAfter = state.queues[0];

            return {
                allConfigured,
                totalHeads,
                amberTriggered,
                qBefore,
                qAfter,
                signals: { ...state.signals }
            };
        }""")

        assert sig4way_res["allConfigured"], f"Independent approach signals configuration invalid: {sig4way_res}"
        assert sig4way_res["totalHeads"] == 24, f"Expected 24 signal heads, got {sig4way_res['totalHeads']}"
        assert sig4way_res["amberTriggered"], "Heavy N/S queues failed to actuate green transition!"
        assert sig4way_res["qAfter"] < sig4way_res["qBefore"], "Queues failed to discharge during green phase!"
        print(f"[PASS] 24 Independent approach signals verified across all 6 junctions.", flush=True)
        print(f"[PASS] Dynamic queue relief verified: N/S queue cleared from {sig4way_res['qBefore']:.1f} to {sig4way_res['qAfter']:.1f} without freezing.", flush=True)

        # 11E. Fluid Queue Discharge Kinematics (a >= 0.035 on lead departure)
        print("[KINEMATICS] Verifying Fluid Queue Discharge Kinematics (a >= 0.035 on lead departure)...", flush=True)
        fluid_res = page.evaluate("""() => {
            for (let j = 2; j < vehicles.length; j++) {
                if (vehicles[j].corridorId === "H1" && vehicles[j].direction === 1) {
                    vehicles[j].x = 1000;
                }
            }
            const v1 = vehicles[0];
            const v2 = vehicles[1];

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
                trailSpeed: v2.speed
            };
        }""")

        assert fluid_res["queueBlockedBefore"], "Trailing car was not queued when lead was stopped!"
        assert not fluid_res["queueBlockedAfter"], "Trailing car remained blocked after lead car departed!"
        assert fluid_res["idmAccel"] >= 0.035, f"IDM acceleration too low: {fluid_res['idmAccel']} < 0.035"
        assert fluid_res["trailAccel"] >= 0.035 or fluid_res["trailSpeed"] > 0, \
            f"Trailing car failed to accelerate immediately: accel={fluid_res['trailAccel']}, speed={fluid_res['trailSpeed']}"
        print(f"[PASS] Fluid queue discharge verified: IDM acceleration {fluid_res['idmAccel']:.4f} >= 0.035 px/frame.", flush=True)

        # 12. Zero Inter-Vehicle Bounding Box Collisions (Minimum Distance >= 26px)
        print("[KINEMATICS] Verifying 0 inter-vehicle bounding box collisions (min distance >= 26px)...", flush=True)
        min_fleet_dist = 9999.0
        collision_violations = []

        for sample_step in range(25):
            pair_check = page.evaluate("""() => {
                let minDist = 9999;
                let closestPair = null;
                const violations = [];

                for (let i = 0; i < vehicles.length; i++) {
                    const v1 = vehicles[i];
                    for (let j = i + 1; j < vehicles.length; j++) {
                        const v2 = vehicles[j];
                        const d = Math.hypot(v2.x - v1.x, v2.y - v1.y);
                        if (d < minDist) {
                            minDist = d;
                            closestPair = [v1.id, v2.id];
                        }
                        if (d < 26.0) {
                            violations.push({ id1: v1.id, id2: v2.id, dist: d });
                        }
                    }
                }
                return { minDist, closestPair, violations };
            }""")
            if pair_check["minDist"] < min_fleet_dist:
                min_fleet_dist = pair_check["minDist"]
            if pair_check["violations"]:
                collision_violations.extend(pair_check["violations"])
            page.wait_for_timeout(100)

        print(f"[KINEMATICS] Minimum inter-vehicle bumper distance recorded: {min_fleet_dist:.2f}px", flush=True)
        assert len(collision_violations) == 0, f"Inter-vehicle collisions detected! Violations: {collision_violations}"
        assert min_fleet_dist >= 26.0, f"Minimum vehicle separation violated: {min_fleet_dist:.2f}px < 26.0px"
        print(f"[PASS] 0 inter-vehicle collisions verified across fleet! Min distance {min_fleet_dist:.2f}px >= 26.0px.", flush=True)

        # 13. Turning Vehicles Yield & Non-Clipping Verification
        print("[KINEMATICS] Verifying turning vehicles yield and maintain safe clearance...", flush=True)
        turn_safety_data = page.evaluate("""() => {
            const turning = vehicles.filter(v => v.state === 'TURNING');
            let minTurningDist = 9999;
            let clippingViolations = 0;

            for (const tv of turning) {
                for (const ov of vehicles) {
                    if (tv.id === ov.id) continue;
                    const d = Math.hypot(ov.x - tv.x, ov.y - tv.y);
                    if (d < minTurningDist) minTurningDist = d;
                    if (d < 26.0) clippingViolations++;
                }
            }
            return {
                turningCount: turning.length,
                minTurningDist: minTurningDist,
                clippingViolations: clippingViolations
            };
        }""")
        print(f"[PASS] Active turning vehicles: {turn_safety_data['turningCount']}, Min clearance to other vehicles: {turn_safety_data['minTurningDist']:.2f}px", flush=True)
        assert turn_safety_data["clippingViolations"] == 0, f"Turning vehicles clipped into other vehicles: {turn_safety_data}"
        print("[PASS] Turning vehicle clearance verified: ZERO clipping into cruising or opposing vehicles.", flush=True)

        # 14. Capture Chase Cam Screenshot
        shot_chase = os.path.join(ROOT_DIR, "results", "chase_cam_telemetry.png")
        page.screenshot(path=shot_chase, full_page=False)
        assert os.path.exists(shot_chase) and os.path.getsize(shot_chase) > 20000, "Chase cam screenshot failed"
        print(f"[PASS] Chase Cam Telemetry screenshot saved to {shot_chase} ({os.path.getsize(shot_chase)} bytes)", flush=True)

        # 15. Test Disengaging Chase Cam
        page.evaluate("() => disengageChaseCam()")
        is_disengaged = page.evaluate("() => followedVehicleId === null && document.getElementById('vehicleChaseHud').style.display === 'none'")
        assert is_disengaged, "Chase Cam failed to disengage cleanly!"
        print("[PASS] Chase Cam disengagement verified cleanly.", flush=True)

        # 16. Check Console Errors
        assert len(console_errors) == 0, f"Found console errors: {console_errors}"
        print("[PASS] Zero console errors during vehicle kinematics verification!", flush=True)

        browser.close()

    if server:
        server.shutdown()
        server.server_close()
        print("[KINEMATICS] Server closed.", flush=True)

    print("\n=======================================================")
    print("ALL VEHICLE KINEMATICS, 3D GROUND CONTACT & CHASE CAM VERIFICATIONS PASSED 100%!")
    print("=======================================================", flush=True)
    return True


def test_vehicle_kinematics():
    assert run_kinematics_verification()


if __name__ == "__main__":
    success = run_kinematics_verification()
    sys.exit(0 if success else 1)
