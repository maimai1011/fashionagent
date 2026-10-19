"""Provide example port settings to isolated unit tests."""

import os
from pathlib import Path

_example_env = Path(__file__).resolve().parents[1] / "storage/.env.example"
for _line in _example_env.read_text().splitlines():
    _name, _separator, _value = _line.partition("=")
    if _separator and _name.endswith("_PORT"):
        os.environ.setdefault(_name, _value)
