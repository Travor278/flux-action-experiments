# FLUX_Action evaluation adapter

Evaluation-only adapter for [FLUX Action](https://github.com/black-forest-labs/flux-action), commit830f1ce178d7aea52f18fde31d8e8e36ce6bae21. Supports `aloha_agilex` and `joint` only. Training remains in the parent experiment scripts.

Pass an absolute exported checkpoint directory as `ckpt_name` with config.json, model.safetensors and manifest.json. Shared encoder paths must remain reachable. Uses saved absolute action normalization with no gripper flips. Three decoded RGB cameras map to head, left wrist and right wrist. Reference inference is used for the custom action namespace. Executes16 of32 predicted actions before replanning, matching the saved protocol.

Run `install.sh /absolute/flux-action/.venv` for RPC dependencies. Server launcher arg8 / eval.sh arg9 takes an absolute virtual environment directory. Environment-client arg10 uses the upstream conda convention.

```sh
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin click_bell /absolute/export-3000-ema aloha_agilex joint 42 0 0 /absolute/flux-action/.venv simulation_env
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboTwin click_bell /absolute/export-3000-ema aloha_agilex joint 42 0 0 /absolute/flux-action/.venv simulation_env
```

These full server-loop commands have NOT passed: checkpoint/export and simulation environment are pending. Contract tests do not demonstrate successful policy inference or robot performance.

The experiment's external LeRobot v2.1 exporter uses the official `decode_image_bit` and lossless video roundtrip checks. Head camera feature cam_head differs from generic converter cam_high; the FLUX index maps explicitly. pi05 must apply an explicit camera rename to the same source data. This adapter does no data conversion.
