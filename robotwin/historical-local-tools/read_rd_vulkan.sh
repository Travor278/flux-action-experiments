R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
find /etc/vulkan/icd.d /usr/share/vulkan/icd.d -maxdepth 1 -name '*.json' -print 2>/dev/null > "$R/status/vulkan-icd-files.txt"