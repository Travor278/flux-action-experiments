"""Same observation whitelist and per-tick packing verified by the WS smoke."""
import numpy as np
from XPolicyLab.policy.FLUX_RoboDojo_History.model import CAMERAS,image_chw
from XPolicyLab.utils.process_data import pack_robot_state,get_robot_action_dim_info
def eval_one_episode(TASK_ENV,model_client):
    model_client.call(func_name='begin_episode',eval_seed=int(TASK_ENV.eval_seed),layout_id=int(TASK_ENV.env_seeds[0]))
    info=get_robot_action_dim_info('arx_x5')
    while not TASK_ENV.is_episode_end():
        raw=TASK_ENV.get_obs()
        state=np.asarray(pack_robot_state(raw,'joint',info),dtype=np.float32)
        assert state.shape==(14,) and np.isfinite(state).all()
        images={aliases[0]:np.ascontiguousarray(image_chw(raw,aliases).transpose(1,2,0)) for aliases in CAMERAS.values()}
        model_client.call(func_name='update_obs',obs={'state':state,'images':images,'instruction':raw['instruction']})
        actions=model_client.call(func_name='get_action');assert len(actions)==1
        TASK_ENV.take_action(actions[0])
