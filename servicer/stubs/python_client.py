"""Minimal HTTP client stub for the HanWAM inference service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request as urllib_request


DEFAULT_URL = "http://127.0.0.1:8080/v1/plan"
DEFAULT_PAYLOAD = Path(__file__).resolve().parent / "sample_plan_request.json"


def post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Call HanWAM /v1/plan with a JSON payload.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    response = post_json(args.url, payload, timeout=args.timeout)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
