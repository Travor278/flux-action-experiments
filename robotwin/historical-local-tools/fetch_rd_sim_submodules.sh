set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export GIT_LFS_SKIP_SMUDGE=1
nohup bash -c 'set -e; R="$1"; git -C "$R/RoboDojo" submodule update --init --depth 1 third_party/IsaacLab third_party/curobo; test "$(git -C "$R/RoboDojo/third_party/IsaacLab" rev-parse HEAD)" = afca7b09d60d8beb9c1cb28b43066499940b969b; test "$(git -C "$R/RoboDojo/third_party/curobo" rev-parse HEAD)" = d17b54ce32cba095c0b000c4c58777075d11de0e; printf PINNED_SIM_SOURCE_PASS > "$R/status/sim-source-verified.txt"' bash "$R" > "$R/status/sim-submodules.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/sim-submodules.pid"