R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
find "$B/system-libs" -name 'libGLU*' -o -name 'libOpenGL*' > "$R/status/existing-glu-libs.txt"