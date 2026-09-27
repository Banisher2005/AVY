#!/usr/bin/env python3
"""Script to start the AVY Web Server & SSE Streaming Dashboard."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

from avy.config import AVYConfig
from avy.ui.server import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start AVY Streaming Live RAG Web Server")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port to bind (default: 8001)")
    parser.add_argument("--host", "-H", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    args, unknown = parser.parse_known_args()

    # Check for legacy positional port argument (e.g. `python scripts/demo_server.py 8001`)
    if args.port is None and unknown:
        for arg in unknown:
            if arg.isdigit():
                args.port = int(arg)
                break

    config = AVYConfig.load()
    port = args.port or config.server_port or 8001
    host = args.host or config.server_host or "127.0.0.1"

    config.server_port = port
    config.server_host = host

    print(f"Starting AVY Web Dashboard & Streaming API on http://{host}:{port}")
    app = create_app(config=config)
    uvicorn.run(app, host=host, port=port)
