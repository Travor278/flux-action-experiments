R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/RoboDojo"
PYTHONPATH="$R/RoboDojo:$R:$R/scripts" "$R/sim-venv/bin/python" - <<'PY' > "$R/status/scene-import-cpu.log" 2>&1
import numpy,torch
from PIL import Image
from omegaconf import OmegaConf
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config,process_randomization
from env.global_configs import ENV_CONFIG_PATH
print('PRE_APP_CPU_IMPORTS_PASS',flush=True)
PY
