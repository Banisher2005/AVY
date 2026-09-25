#!/usr/bin/env python3
"""Script to start the AVY Web Server & SSE Streaming Dashboard."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

from avy.config import AVYConfig
from avy.ui.server import create_app

if __name__ == "__main__":
    config = AVYConfig.load()
    print(f"Starting AVY Web Dashboard & Streaming API on http://{config.server_host}:{config.server_port}")
    app = create_app(config=config)
    uvicorn.run(app, host=config.server_host, port=config.server_port)
