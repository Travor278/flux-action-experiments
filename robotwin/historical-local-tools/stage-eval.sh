set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
tar -xzf "$B/assets/robotwin-code.tar.gz" -C "$B"
if [ ! -e "$B/env_cfg" ]; then ln -s "$B/RoboTwin/env_cfg" "$B/env_cfg"; fi
if [ ! -e "$B/RoboTwin/XPolicyLab" ]; then ln -s "$B/XPolicyLab" "$B/RoboTwin/XPolicyLab"; fi
nohup bash -c 'timeout 45 find /workspace/ssd/shared-assets /workspace/ssd /workspace/hdd -maxdepth 6 -type d \( -name embodiments -o -name 050_bell -o -name curobo \) 2>/dev/null' > "$B/status/eval-asset-locations.txt" 2>&1 < /dev/null &
python3 -m py_compile "$B/XPolicyLab/policy/FLUX_Action/model.py" "$B/XPolicyLab/policy/FLUX_Action/deploy.py"
