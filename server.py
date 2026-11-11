#!/usr/bin/env python
"""
Standalone server launcher for ATMS.
Usage:
    python server.py [--port 8000]
"""

import argparse
from src.server import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATMS Web Server & Surveillance Console")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server host (default: 0.0.0.0)")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)
