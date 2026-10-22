R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
ps -eo pid,etime,args | grep -E '(test_training_contract|prepare_formal_drafts)' | grep -v grep > "$R/status/formal-draft-process.txt"
