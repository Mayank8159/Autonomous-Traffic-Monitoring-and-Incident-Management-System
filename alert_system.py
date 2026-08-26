"""
Async alert & dispatch module.

Responsibilities:
  • Export annotated frame snapshots to disk.
  • Log incident metadata to structured JSON-lines files.
  • Dispatch mock SMS payloads and HTTP webhooks asynchronously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import cv2
import numpy as np

from analytics import Incident
from config import AlertConfig, CFG, IncidentType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot exporter
# ---------------------------------------------------------------------------

class SnapshotExporter:
    """Saves frame snapshots to ``snapshot_dir`` with incident metadata in filename."""

    def __init__(self, cfg: AlertConfig) -> None:
        self._dir = Path(cfg.snapshot_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._quality = cfg.snapshot_quality

    def export(self, frame: np.ndarray, incident: Incident) -> Optional[str]:
        ts = datetime.fromtimestamp(incident.timestamp, tz=timezone.utc)
        fname = (
            f"{ts.strftime('%Y%m%d_%H%M%S')}_{incident.incident_type.value}"
            f"_track{incident.track_id}.jpg"
        )
        path = self._dir / fname
        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            return str(path)
        except Exception as exc:
            logger.error("Snapshot export failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Structured incident logger
# ---------------------------------------------------------------------------

class IncidentLogger:
    """Appends one JSON object per incident to a daily log file."""

    def __init__(self, cfg: AlertConfig) -> None:
        self._dir = Path(cfg.log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def log(self, incident: Incident, snapshot_path: Optional[str] = None) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self._dir / f"incidents_{today}.jsonl"
        entry: Dict[str, Any] = {
            "timestamp": incident.timestamp,
            "timestamp_utc": datetime.fromtimestamp(
                incident.timestamp, tz=timezone.utc
            ).isoformat(),
            "type": incident.incident_type.value,
            "track_id": incident.track_id,
            "location_id": incident.location_id,
            "metadata": incident.metadata,
        }
        if snapshot_path:
            entry["snapshot"] = snapshot_path
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return str(log_file)


# ---------------------------------------------------------------------------
# Webhook / SMS dispatcher
# ---------------------------------------------------------------------------

class _MockSMSDispatcher:
    """Logs SMS payloads to console instead of hitting a real gateway."""

    @staticmethod
    async def send(phone: str, body: str) -> bool:
        logger.info("[SMS MOCK] To=%s  Body=%s", phone, body)
        return True


class _WebhookDispatcher:
    """POST JSON payload to configured webhook URLs concurrently."""

    def __init__(self, urls: List[str], max_concurrent: int = 10) -> None:
        self._urls = urls
        self._sem = asyncio.Semaphore(max_concurrent)

    async def dispatch(self, payload: Dict[str, Any],
                       session: aiohttp.ClientSession) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        tasks = [self._post(url, payload, session) for url in self._urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for url, resp in zip(self._urls, responses):
            if isinstance(resp, Exception):
                logger.warning("Webhook %s failed: %s", url, resp)
                results[url] = False
            else:
                results[url] = resp  # type: ignore[assignment]
        return results

    async def _post(self, url: str, payload: Dict[str, Any],
                    session: aiohttp.ClientSession) -> bool:
        async with self._sem:
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    ok = resp.status < 400
                    if not ok:
                        logger.warning("Webhook %s returned %d", url, resp.status)
                    return ok
            except Exception as exc:
                logger.warning("Webhook POST to %s failed: %s", url, exc)
                return False


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class AlertSystem:
    """High-level alert coordinator used by the main processing loop."""

    def __init__(self, cfg: Optional[AlertConfig] = None) -> None:
        self._cfg = cfg or CFG.alert
        self._snapshotter = SnapshotExporter(self._cfg)
        self._logger = IncidentLogger(self._cfg)
        self._sms = _MockSMSDispatcher() if self._cfg.enable_sms_mock else None
        self._webhooks: Optional[_WebhookDispatcher] = (
            _WebhookDispatcher(list(self._cfg.webhook_urls), self._cfg.max_concurrent_dispatches)
            if self._cfg.webhook_urls
            else None
        )
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._session: Optional[aiohttp.ClientSession] = None
        self.recent_alerts: List[Dict[str, Any]] = []

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def process_incidents(self, incidents: List[Incident],
                                frame: np.ndarray) -> None:
        for inc in incidents:
            await self._handle(inc, frame)

    async def _handle(self, incident: Incident, frame: np.ndarray) -> None:
        # 1 — snapshot
        snap_path = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._snapshotter.export, frame, incident,
        )
        incident.frame_snapshot = None  # don't serialise numpy array

        # 2 — structured log
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self._logger.log, incident, snap_path,
        )

        # 3 — build payload
        payload: Dict[str, Any] = {
            "timestamp": incident.timestamp,
            "type": incident.incident_type.value,
            "track_id": incident.track_id,
            "location_id": incident.location_id,
            "metadata": incident.metadata,
            "snapshot_path": snap_path,
        }

        # 4 — mock SMS for high-severity
        if self._sms and incident.incident_type in (
            IncidentType.COLLISION, IncidentType.WRONG_WAY,
        ):
            sms_body = (
                f"ALERT: {incident.incident_type.value} detected "
                f"(track {incident.track_id}) at {incident.location_id}"
            )
            await self._sms.send("+0000000000", sms_body)

        # 5 — webhooks
        if self._webhooks:
            session = await self._ensure_session()
            results = await self._webhooks.dispatch(payload, session)
            payload["webhook_results"] = results

        self.recent_alerts.append(payload)
        if len(self.recent_alerts) > 200:
            self.recent_alerts = self.recent_alerts[-200:]

    async def shutdown(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._executor.shutdown(wait=False)
