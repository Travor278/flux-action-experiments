import numpy as np
import pytest
from XPolicyLab.policy.FLUX_Action.model import observation_arrays, action_dicts

def test_observation_mapping_preserves_rgb_and_joint_order():
    obs={'state':{'left_arm_joint_state':np.arange(6),'left_ee_joint_state':[.2],
                 'right_arm_joint_state':np.arange(6)+10,'right_ee_joint_state':[.8]},
         'instruction':'touch the bell','vision':{c:{'color':np.full((8,12,3),v,dtype=np.uint8)}
         for c,v in [('cam_head',11),('cam_left_wrist',22),('cam_right_wrist',33)]}}
    result=observation_arrays([obs],[6,1,6,1])
    np.testing.assert_allclose(result['state'][0],[0,1,2,3,4,5,.2,10,11,12,13,14,15,.8])
    assert [result[c].mean() for c in ['images.head','images.left_wrist','images.right_wrist']]==[11,22,33]
    assert result['task']==['touch the bell']

def test_action_mapping_preserves_native_units_and_replan_horizon():
    data=np.arange(2*32*14,dtype=np.float32).reshape(2,32,14)
    result=action_dicts(data,[6,1,6,1],16)
    assert [len(x) for x in result]==[16,16]
    np.testing.assert_array_equal(result[1][15]['right_arm_joint_state'],data[1,15,7:13])
    np.testing.assert_array_equal(result[0][0]['left_ee_joint_state'],[6])

@pytest.mark.parametrize('bad', [np.full((1,32,14),np.nan),np.zeros((1,32,13))])
def test_invalid_prediction_rejected(bad):
    with pytest.raises(ValueError):action_dicts(bad,[6,1,6,1],16)
