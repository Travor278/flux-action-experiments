"""Produce isolated integration scripts; never alter active training files."""
from pathlib import Path
import ast,hashlib,json
N=Path(__file__).parent
s=(N/'probe_conveyor_motion.py').read_text()
a=s.index('    class ResetOnlyClient:');b=s.index('    base=Path(ENV_CONFIG_PATH)',a)
s=s[:a]+s[b:]
s=s.replace("'port':1,'host':'localhost'","'port':int(os.environ['FLUX_WS_PORT']),'host':'127.0.0.1'")
s=s.replace("'runtime_control_smoke_not_score'","'real_ws_smoke_not_score'")
a=s.index('    class HoldMeasuredState:');b=s.index("    env.close();phase('complete')",a)
s=s[:a]+'''    info=get_robot_action_dim_info('arx_x5')
    receipts=[]
    for tick in range(53):
        assert not env.is_episode_end(), 'Diagnostic episode ended before interface coverage; not a benchmark result'
        raw=env.get_obs()
        state=np.asarray(pack_robot_state(raw,'joint',info),dtype=np.float32)
        assert state.shape==(14,) and np.isfinite(state).all()
        images={aliases[0]:np.ascontiguousarray(adapter.image_chw(raw,aliases).transpose(1,2,0)) for aliases in adapter.CAMERAS.values()}
        assert all(im.shape==(480,640,3) and im.dtype==np.uint8 and float(im.std())>1 for im in images.values())
        obs={'state':state,'images':images,'instruction':raw['instruction']}
        env.model_client.call(func_name='update_obs',obs=obs)
        actions=env.model_client.call(func_name='get_action')
        assert len(actions)==1
        flat=np.asarray(pack_robot_state({'state':actions[0]},'joint',info),dtype=np.float32)
        assert flat.shape==(14,) and np.isfinite(flat).all()
        env.take_action(actions[0])
        assert env.take_action_cnt==[tick+1]
        receipts.append({'tick':tick,'action_sha256':hashlib.sha256(flat.tobytes()).hexdigest(),'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'rgb_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()}})
        phase('real_ws_control',tick=tick,executed=env.take_action_cnt[0])
    env.model_client.call(func_name='reset')
    record={'time':time.time(),'verified':True,'task':task,'control_ticks':53,'layout_id':0,'receipts':receipts,'scope':'Real diagnostic export4 policy over official WebSocket, 53 official control ticks, no benchmark score or trained policy claim'}
    (out/'ws-control-verified.json').write_text(json.dumps(record,indent=2))
'''+s[b:]
s=s.replace("    restore_loaded_conveyor_graph(omni.usd.get_context().get_stage(),out/'graph-compat.jsonl')", "    if task=='match_and_pick_from_conveyor':restore_loaded_conveyor_graph(omni.usd.get_context().get_stage(),out/'graph-compat.jsonl')")
s=s.replace('Two-task runtime/observation/control smoke; no learned policy or score.','Real WebSocket diagnostic export integration; no benchmark score.')
ast.parse(s);(N/'probe_ws_control.py').write_text(s,newline='\n')
base=(N/'launch_conveyor_motion.sh').read_text()
base=base.replace('conveyor-motion-jobs','ws-control-jobs').replace('conveyor-motion-current.json','ws-control-current.json')
base=base.replace('probe_conveyor_motion.py','probe_ws_control.py').replace('render_passed','ws_passed').replace('render_failed','ws_failed')
base=base.replace("trap 'kill \"$telemetry_pid\" 2>/dev/null || true' EXIT", "server_pid=''\ntrap 'kill \"$telemetry_pid\" ${server_pid:-} 2>/dev/null || true' EXIT")
pos=base.index('set +e\n')
base=base[:pos]+'''export FLUX_WS_PORT=48691 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONPATH="$R/ws-deps:$R/scripts:$R/flux-action/src:$R/XPolicyLab:$R"
"$R/policy-venv/bin/python" "$R/scripts/ws_probe_server.py" > "$RUN/server.log" 2>&1 &
server_pid=$!
for i in $(seq 1 180); do
 if [ -f "$RUN/server-ready.json" ]; then break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then exit 91; fi
 sleep 2
done
test -f "$RUN/server-ready.json"
'''+base[pos:]
base=base.replace('for task in match_and_pick_from_conveyor; do','for task in cover_blocks match_and_pick_from_conveyor; do')
base=base.replace('scene-control-verified.json','ws-control-verified.json')
base=base.replace("p=out/'match_and_pick_from_conveyor/ws-control-verified.json'", "p=out/'match_and_pick_from_conveyor/ws-control-verified.json'")
base=base.replace("for task in ('match_and_pick_from_conveyor',)","for task in ('cover_blocks','match_and_pick_from_conveyor')")
base=base.replace("assert all(v['verified'] for v in r['tasks'].values())", "assert all(v['verified'] and v['control_ticks']==53 for v in r['tasks'].values())\n events=[json.loads(line) for line in (out/'server-events.jsonl').read_text().splitlines()]\n actions=[e for e in events if e['event']=='action'];assert len(actions)==106\n assert [e['tick'] for e in actions]==list(range(53))*2\n assert sum(e['tick']==52 and e['anchors']==[0,25,50] for e in actions)==2\n assert events[-1]['event']=='reset'\n r['server_history_audit']=True")
(N/'launch_ws_control.sh').write_text(base,newline='\n')
files={p:hashlib.sha256((N/p).read_bytes()).hexdigest() for p in ['ws_probe_server.py','probe_ws_control.py','launch_ws_control.sh','conveyor_graph_compat.py']}
gate={'ready':True,'scope':'Isolated bounded real WS/simulator functional integration, not benchmark scoring','files':files}
(N/'ws-control-ready.json').write_text(json.dumps(gate,indent=2),newline='\n')
