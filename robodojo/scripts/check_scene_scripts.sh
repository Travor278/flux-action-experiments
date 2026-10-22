R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
("$R/policy-venv/bin/python" -m py_compile "$R/scripts/probe_scene_control.py" && bash -n "$R/scripts/launch_scene_control.sh" && echo SCENE_SCRIPTS_SYNTAX_PASS) > "$R/status/scene-control-syntax.log" 2>&1
ps -p 2162685 -o pid,etime,args > "$R/status/selected-download-process.txt"
