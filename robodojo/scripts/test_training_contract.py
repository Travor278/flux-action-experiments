import json,shutil,tempfile,types
from pathlib import Path
from flux_action.config import PolicyConfig
from robodojo_training_contract import RoboDojoTrainer,custom_data_contract

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
def test_actual_resume_rejects_rows_and_episode_manifest_changes():
    original=R/'assets/history-index-preflight-v1/cover_blocks'
    export=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1/export-4-ema'
    cfg=json.loads((export/'config.json').read_text())
    with tempfile.TemporaryDirectory(dir=R/'status',prefix='contract-test-') as temp:
        root=Path(temp);index=root/'index';shutil.copytree(original,index)
        settings=types.SimpleNamespace(dataset='robodojo_history_dataset:build_dataset',index_dir=str(index),visits_per_epoch=32,to_dict=lambda:{})
        fake=object.__new__(RoboDojoTrainer);fake.config=settings;fake.policy=types.SimpleNamespace(config=PolicyConfig(**cfg))
        (root/'config.json').write_text(json.dumps(cfg))
        initial=fake._data_contract()
        (root/'state.json').write_text(json.dumps({'extra':{'data_contract':initial}}))
        fake._validate_resume_contract(root)
        # Moving/reformatting semantic manifest should not reject a resume.
        manifest=json.loads((index/'manifest.json').read_text())
        (index/'manifest.json').write_text(json.dumps(manifest,indent=4))
        assert fake._data_contract()==initial
        fake._validate_resume_contract(root)
        rows=(index/'rows.npy').read_bytes()
        (index/'rows.npy').write_bytes(rows[:-1]+bytes([rows[-1]^1]))
        try:fake._validate_resume_contract(root)
        except ValueError as exc:assert 'dataset/index' in str(exc)
        else:raise AssertionError('Changed numerical rows accepted')
        (index/'rows.npy').write_bytes(rows)
        manifest['episodes'][0]['n_frames']+=1
        (index/'manifest.json').write_text(json.dumps(manifest))
        try:fake._validate_resume_contract(root)
        except ValueError as exc:assert 'dataset/index' in str(exc)
        else:raise AssertionError('Changed episode manifest accepted')
