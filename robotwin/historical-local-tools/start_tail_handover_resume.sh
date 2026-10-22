B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/launch_tail_handover_resume.sh"
bash -n "$B/scripts/launch_tail_handover_resume.sh"
nohup bash "$B/scripts/launch_tail_handover_resume.sh" > "$B/status/flux-tail-handover-resume-launch.log" 2>&1 < /dev/null & echo $! > "$B/status/flux-tail-test.pid"
