#!/bin/bash
set -euo pipefail
POLICY_ENV=${1:?pass the existing FLUX standalone virtual environment path}
uv pip install --python "$POLICY_ENV/bin/python" 'websockets>=14,<16' msgpack msgpack-numpy 'pydantic>=2.5'
