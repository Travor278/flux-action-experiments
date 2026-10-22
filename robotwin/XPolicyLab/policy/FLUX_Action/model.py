"""FLUX Action reference inference for native RoboTwin joint commands."""
from pathlib import Path
import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

CAMERAS = {'images.head': 'cam_head', 'images.left_wrist': 'cam_left_wrist', 'images.right_wrist': 'cam_right_wrist'}
STATE_KEYS = ['left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state']

def observation_arrays(observations, dimensions):
    states, prompts = [], []
    images = {key: [] for key in CAMERAS}
    for obs in observations:
        parts = [np.asarray(obs['state'][k], dtype=np.float32).reshape(-1) for k in STATE_KEYS]
        if [len(x) for x in parts] != dimensions:
            raise ValueError('Observation joint dimensions do not match robot metadata')
        state = np.concatenate(parts)
        if not np.isfinite(state).all():
            raise ValueError('Nonfinite robot state')
        states.append(state)
        prompt = obs.get('instruction', obs.get('instructions'))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Expected a nonempty runtime instruction string')
        prompts.append(prompt)
        for key, camera in CAMERAS.items():
            rgb = np.asarray(obs['vision'][camera]['color'])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError('Policy server must supply decoded HWC uint8 RGB')
            images[key].append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return {'state': np.stack(states), 'task': prompts, **{key: np.stack(value) for key, value in images.items()}}

def action_dicts(actions, dimensions, execute_steps):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != sum(dimensions) or not np.isfinite(actions).all():
        raise ValueError('Invalid predicted native action chunk')
    if not 1 <= execute_steps <= actions.shape[1]:
        raise ValueError('Invalid executed action horizon')
    edges = np.cumsum([0] + dimensions)
    return [[{key: row[edges[i]:edges[i+1]].copy() for i, key in enumerate(STATE_KEYS)}
             for row in sequence[:execute_steps]] for sequence in actions]

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        import torch
        from flux_action.policy import FluxActionPolicy
        super().__init__()
        if model_cfg['action_type'] != 'joint' or model_cfg['env_cfg_type'] != 'aloha_agilex':
            raise ValueError('This adapter supports aloha_agilex joint actions only')
        dims = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self.dimensions = [dims['arm_dim'][0], dims['ee_dim'][0], dims['arm_dim'][1], dims['ee_dim'][1]]
        checkpoint = resolve_checkpoint_root(model_cfg, Path(__file__).parent/'checkpoints')
        self.device = model_cfg.get('device', 'cuda')
        self.model = FluxActionPolicy.from_pretrained(str(checkpoint), device=self.device)
        cfg = self.model.config
        if tuple(cfg.camera_keys) != tuple(CAMERAS) or cfg.action_dim != sum(self.dimensions):
            raise ValueError('Checkpoint camera/action contract differs from this adapter')
        if cfg.inference_profile != 'default' or cfg.action_parameterization != 'absolute' or cfg.gripper_flip_dims:
            raise ValueError('Expected single-frame absolute native actions with no gripper flips')
        cfg.validate_inference()
        self.execute_steps = int(model_cfg.get('execute_steps', cfg.n_action_steps))
        if self.execute_steps != cfg.n_action_steps:
            raise ValueError('Replanning horizon must match the saved protocol')
        self.observations = []
        self.torch = torch

    def update_obs(self, obs):
        self.observations = [obs]

    def update_obs_batch(self, obs_list):
        self.observations = list(obs_list)

    def get_action_batch(self, env_idx_list=None):
        if not self.observations:
            raise RuntimeError('No observation supplied after reset')
        if env_idx_list is not None and len(env_idx_list) != len(self.observations):
            raise ValueError('Active environment indices and observations differ')
        arrays = observation_arrays(self.observations, self.dimensions)
        batch = {key: value if key == 'task' else self.torch.from_numpy(value).to(self.device) for key, value in arrays.items()}
        with self.torch.inference_mode():
            predictions = self.model.predict_action_chunk(batch).float().cpu().numpy()
        return action_dicts(predictions, self.dimensions, self.execute_steps)

    def get_action(self):
        if len(self.observations) != 1:
            raise ValueError('get_action requires exactly one observation')
        return self.get_action_batch()[0]

    def reset(self):
        self.observations = []
        self.model.reset()
