from pathlib import Path
import hashlib,json,shutil,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');B=R.parent/'flux_piperx_20261008';S=Path(str(R).replace('/hdd/','/ssd/'))
assert json.loads((R/'status/real-vae-independent-audit.json').read_text())['verified']
assert not (R/'configs/full-preflight-v1.json').exists()
S.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(S).free>350_000_000_000
base=json.loads((B/'configs/click_bell-10k-tail-v1.json').read_text())
stats=json.loads((R/'status/numeric-data-verified.json').read_text())['tasks']['cover_blocks']
base.update(source_root=str(R/'official-source'),index_dir=str(R/'assets/history-index-preflight-v1/cover_blocks'),output_dir=str(S/'runs/full-preflight-v1'),dataset='robodojo_history_dataset:build_dataset',steps=2,frozen_steps=0,trunk_warmup_steps=1,heads_warmup_steps=1,cooldown_start=100,cooldown_steps=100,windows_per_rank=2,grad_accumulation=8,prefetch_factor=2,frame_hw=[480,640],checkpoint_every=2,log_every=1,resume='none')
assert not Path(base['output_dir']).exists()
base['policy'].update(inference_profile='history',n_obs_steps=8,history_snapshots=2,condition_on_past_actions=True,prefix_memory_spec={'version':'causal_prefix_rgb_v1','ticks':[0,25,50],'fps':25,'unavailable_rule':'current_observation','reset':'every_episode'},fps=25.,action_modality='action_robodojo',caption_dropout=0.,augment=False,camera_dropout={},action_normalization={'q01':stats['action_q01'],'q99':stats['action_q99']},state_normalization={'q01':stats['state_q01'],'q99':stats['state_q99']})
(R/'configs').mkdir(exist_ok=True)
path=R/'configs/full-preflight-v1.json';path.write_text(json.dumps(base,indent=2))
(R/'status/full-preflight-prepared.json').write_text(json.dumps({'time':time.time(),'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Diagnostic two updates then exact resume3/export, no formal training budget freeze'},indent=2))
print('FULL_PREFLIGHT_CONFIG_PREPARED')
