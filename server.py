#!/usr/bin/env python
"""
Standalone server launcher for ATMS.
Usage:
    python server.py [--port 8000]
"""

import argparse
import os
from src.server import start_server

if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", 8000))
    default_host = os.environ.get("HOST", "0.0.0.0")
    parser = argparse.ArgumentParser(description="ATMS Web Server & Surveillance Console")
    parser.add_argument("--port", type=int, default=default_port, help=f"HTTP server port (default: $PORT or {default_port})")
    parser.add_argument("--host", type=str, default=default_host, help=f"HTTP server host (default: $HOST or {default_host})")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)
