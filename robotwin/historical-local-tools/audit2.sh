set -eu
P=/workspace/hdd
B=$P/codex_remote_ops/flux_piperx_20261008
{
df -h /inspire/ssd /tmp "$P" 2>/dev/null || true
ls -la /workspace/ssd 2>/dev/null | head -20
find "$P/codex_remote_ops/anon_robotwin_protocol_20260926_r3" -maxdepth 3 -type d | head -60
find "$P" -maxdepth 5 \( -path '*/bin/python' -o -path '*/bin/python3' -o -name '*piper*' -o -name '*robotwin*' \) 2>/dev/null | head -100
ls /root/.local/share/uv/python 2>/dev/null || true
python3 -c 'import sys; print(sys.version)'
} > "$B/status/assets-audit.txt" 2>&1
