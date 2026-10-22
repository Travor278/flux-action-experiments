"""Bind custom dataset inputs to the native exact-resume checkpoint contract."""
import hashlib,json
from pathlib import Path
from flux_action.training.trainer import Trainer

def custom_data_contract(config):
    if config.dataset!='robodojo_history_dataset:build_dataset':
        raise ValueError('This run requires the audited RoboDojo dataset plug')
    index=Path(config.index_dir)
    manifest=json.loads((index/'manifest.json').read_text())
    with (index/'rows.npy').open('rb') as stream:
        rows_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    value={'version':'robodojo_selected_v1','dataset':config.dataset,
           'manifest':manifest,'rows_sha256':rows_sha256,
           'visits_per_epoch':config.visits_per_epoch}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class RoboDojoTrainer(Trainer):
    def _data_contract(self):
        # Compact rows are only a few MB. Recheck every save/resume so even a
        # mid-run replacement cannot silently receive the old contract.
        return custom_data_contract(self.config)
