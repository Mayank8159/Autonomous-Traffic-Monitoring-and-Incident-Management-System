# tests/test_api_endpoints.py
import sys
import os
import json
import time
import threading
import urllib.request
import urllib.error

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.server import ATMSRequestHandler
from http.server import ThreadingHTTPServer

TEST_PORT = 8012
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def start_test_server():
    server_address = ("127.0.0.1", TEST_PORT)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(server_address, ATMSRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def http_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))


def http_post(path, data=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Connection": "close"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def run_all_api_tests():
    print(f"Starting test server on port {TEST_PORT}...", flush=True)
    server = start_test_server()
    time.sleep(0.5)

    try:
        # 1. /api/status
        code, data = http_get("/api/status")
        assert code == 200, f"Expected 200, got {code}"
        assert data.get("status") == "ready", f"Unexpected status: {data}"
        assert "models" in data, "Missing models in status"
        assert all(data["models"].values()), f"Some models missing: {data['models']}"
        print("[PASS] GET /api/status - All 4 models verified present", flush=True)

        # 2. /api/city/topology
        code, topo = http_get("/api/city/topology")
        assert code == 200, f"Expected 200, got {code}"
        assert "intersections" in topo, "Missing intersections in topology"
        assert len(topo["intersections"]) == 6, f"Expected 6 intersections, got {len(topo['intersections'])}"
        assert len(topo["facilities"]) >= 3, f"Expected facilities, got {topo['facilities']}"
        assert len(topo["road_segments"]) >= 7, f"Expected road_segments, got {topo['road_segments']}"
        print(f"[PASS] GET /api/city/topology ({len(topo['intersections'])} intersections, {len(topo['road_segments'])} corridors, {len(topo['facilities'])} facilities)", flush=True)

        # 3. /api/city/state (including 4-arm telemetry & strict <=1 invariant)
        code, state = http_get("/api/city/state")
        assert code == 200, f"Expected 200, got {code}"
        assert "intersections" in state, "Missing intersections in state"
        for nid, inter in state["intersections"].items():
            lights = inter.get("lights", {})
            open_arms = [k for k, v in lights.items() if v in ("GREEN", "AMBER")]
            assert len(open_arms) <= 1, f"{nid} violated <=1 invariant in /api/city/state: {lights}"
            assert "arm_scores" in inter and "wait_timers" in inter and "is_all_red" in inter
        print(f"[PASS] GET /api/city/state (step: {state.get('step')}, <=1 invariant & 4-arm telemetry verified)", flush=True)

        # 4. /api/city/step (including explicit 4-arm action)
        code, step_resp = http_post("/api/city/step", {"actions": {"INT_2": "N"}})
        assert code == 200, f"Expected 200, got {code}"
        assert "state" in step_resp, "Missing state in step response"
        assert "reward" in step_resp, "Missing reward in step response"
        for nid, inter in step_resp["state"].items():
            lights = inter.get("lights", {})
            open_arms = [k for k, v in lights.items() if v in ("GREEN", "AMBER")]
            assert len(open_arms) <= 1, f"{nid} violated <=1 invariant in /api/city/step: {lights}"
        print(f"[PASS] POST /api/city/step (reward: {step_resp.get('reward'):.2f})", flush=True)

        # 5. /api/city/incident (Trigger crash at INT_2)
        code, inc_resp = http_post("/api/city/incident", {
            "node_id": "INT_2",
            "type": "collision",
            "x": 600,
            "y": 280
        })
        assert code == 200, f"Expected 200, got {code}"
        assert inc_resp.get("status") == "dispatched", f"Unexpected status: {inc_resp}"
        incident = inc_resp.get("incident", {})
        assert incident.get("node_id") == "INT_2" or incident.get("location_node") == "INT_2", f"Wrong location: {incident}"
        print(f"[PASS] POST /api/city/incident (incident id: {incident.get('id')}, police: {incident.get('dispatch', {}).get('police', {}).get('unit')}, ambulance: {incident.get('dispatch', {}).get('ambulance', {}).get('unit')})", flush=True)

        # Verify state reflects incident
        code, state_after_inc = http_get("/api/city/state")
        assert len(state_after_inc.get("incidents", [])) >= 1, "Expected active incident in state"
        assert len(state_after_inc.get("dispatches", [])) >= 1, "Expected active dispatches in state"
        print(f"[PASS] State reflects active incident and dispatches", flush=True)

        # Step while incident is active
        code, step_inc = http_post("/api/city/step", {})
        assert code == 200
        print(f"[PASS] Step during active incident", flush=True)

        # 6. /api/city/resolve_incident
        inc_id = incident.get("id")
        code, res_resp = http_post("/api/city/resolve_incident", {"id": inc_id})
        assert code == 200, f"Expected 200, got {code}"
        assert res_resp.get("status") == "resolved"
        print(f"[PASS] POST /api/city/resolve_incident", flush=True)

        # Verify resolved
        code, state_after_res = http_get("/api/city/state")
        assert len(state_after_res.get("incidents", [])) == 0, "Incident not resolved"
        print(f"[PASS] Incident resolved in state", flush=True)

        # 7. /api/city/reset
        code, reset_resp = http_post("/api/city/reset", {})
        assert code == 200
        assert reset_resp.get("status") == "reset"
        print(f"[PASS] POST /api/city/reset", flush=True)

        # 8. Thread safety check: 10 concurrent requests
        print("Testing thread safety with 10 concurrent requests...", flush=True)
        errors = []
        def worker(idx):
            try:
                if idx % 2 == 0:
                    c, s = http_get("/api/city/state")
                else:
                    c, s = http_post("/api/city/step", {})
                if c != 200:
                    errors.append(f"Request {idx} returned {c}")
            except Exception as e:
                errors.append(f"Request {idx} exception: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent request errors: {errors}"
        print(f"[PASS] Thread safety verified (10 concurrent requests, 0 errors)", flush=True)

        # 9. 4-arm /api/policy/predict & HTTP 400 input validation
        code, pred_ok = http_post("/api/policy/predict", {
            "state": [10.0, 2.0, 1.0, 1.0, 14.0, 1.0, 0.0],
            "wait_timers": [1, 5, 2, 0]
        })
        assert code == 200 and pred_ok.get("recommended_arm") in ("N", "S", "E", "W"), f"Unexpected predict response: {code} {pred_ok}"
        assert len(pred_ok.get("arm_scores", [])) == 4

        code_bad_pred, _ = http_post("/api/policy/predict", {"state": [1, 2]})
        assert code_bad_pred == 400, f"Expected 400 for invalid /api/policy/predict state, got {code_bad_pred}"

        code_bad_step, _ = http_post("/api/city/step", {"actions": {"INT_2": "INVALID"}})
        assert code_bad_step == 400, f"Expected 400 for invalid /api/city/step action, got {code_bad_step}"

        code_bad_inc, _ = http_post("/api/city/incident", {"node_id": "INT_2", "x": "not_a_number"})
        assert code_bad_inc == 400, f"Expected 400 for non-finite incident coordinate, got {code_bad_inc}"

        # Unknown node_id should deterministically fall back to INT_2 without 500 KeyError
        code_fb_inc, fb_inc = http_post("/api/city/incident", {"node_id": "INT_999"})
        assert code_fb_inc == 200 and fb_inc.get("incident", {}).get("node_id") == "INT_2", f"Unexpected fallback response: {code_fb_inc} {fb_inc}"
        http_post("/api/city/resolve_incident", {"id": fb_inc["incident"]["id"]})
        print("[PASS] 4-arm /api/policy/predict & HTTP 400 validation checks verified", flush=True)

        print("\nALL REST API ENDPOINT TESTS PASSED SUCCESSFULLY!", flush=True)
        return True

    finally:
        server.shutdown()
        server.server_close()
        print("Test server shutdown.", flush=True)


def test_rest_api_endpoints():
    """Pytest test case for ATMS REST API endpoints."""
    success = run_all_api_tests()
    assert success is True


if __name__ == "__main__":
    success = run_all_api_tests()
    sys.exit(0 if success else 1)
