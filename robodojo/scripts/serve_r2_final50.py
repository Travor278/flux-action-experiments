"""Final10k server with the same verified CPU-first/text-offload WS deployment."""
import asyncio, json, os, time, traceback, sys
from pathlib import Path
import torch
from flux_action.policy import FluxActionPolicy
from XPolicyLab.policy.FLUX_RoboDojo_History.model import Model
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from client_server.ws.model_server import PolicyServer, PolicyServerConfig

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
out=Path(os.environ['FLUX_SCENE_RUN']);task=sys.argv[1]
assert task in ('cover_blocks','plug_in_charger')
def record(event,**kw):
    with (out/'server-events.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),'event':event,**kw})+'\n')

class AuditedModel(Model):
    def __init__(self):
        self.device=torch.device('cuda');self.info=get_robot_action_dim_info('arx_x5')
        export=Path(str(R).replace('/hdd/','/ssd/'))/'runs'/f'{task}-10k-official-history-r2-20261021'/'export-10000-ema'
        # Same CPU-first/text-offload loading already verified on this export.
        # Preserve exported dtype/config, do not cast scientific parameters.
        self.policy=FluxActionPolicy.from_pretrained(export,device='cpu')
        self.policy.set_text_encoder_offload(True)
        self.policy.to(self.device).eval()
        c=self.policy.config
        assert c.action_dim==14 and c.fps==25 and c.n_action_steps==16
        assert not hasattr(c,'prefix_memory_spec') and c.n_obs_steps==8 and c.history_snapshots==2
        record('loaded',export=str(export),dtype=c.torch_dtype,history=c.n_obs_steps,history_snapshots=c.history_snapshots)
        self.reset()
    def reset(self):
        super().reset()
        assert not self.policy._action_queue
        record('reset',official_history=True)
    def update_obs(self,obs):
        assert set(obs)=={'state','images','instruction'},set(obs)
        super().update_obs(obs)
    def get_action(self):
        actions=super().get_action()
        record('action',official_history=True,
               queue=len(self.policy._action_queue),cuda_peak=torch.cuda.max_memory_allocated())
        return actions

async def main():
    torch.set_num_threads(4)
    model=AuditedModel()
    server=PolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=int(os.environ['FLUX_WS_PORT'])))
    await server.start()
    (out/'server-ready.json').write_text(json.dumps({'time':time.time(),'url':server.url,'pid':os.getpid()}))
    await server.serve_forever()
try:
    asyncio.run(main())
except BaseException:
    (out/'server-failure.txt').write_text(traceback.format_exc())
    raise
