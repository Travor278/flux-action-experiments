import unittest
import numpy as np
from robodojo_history_reader import CAMERAS, RoboDojoHistoryReader, window_plan


def episode(length=5):
    ep = {'episode_index': 324, 'length': length, 'tasks': ['source instruction']}
    for camera in CAMERAS.values():
        k = 'videos/'+camera
        ep.update({k+'/from_timestamp': 10., k+'/to_timestamp': 10.+length/25, k+'/chunk_index': 0, k+'/file_index': 1})
    return ep


class HistoryReaderTests(unittest.TestCase):
    def test_reset_pads_only_past_and_measured_state(self):
        ep = episode()
        states = np.arange(70).reshape(5,14)
        actions = states+1000
        calls = []
        def decode(path, times):
            calls.append(times)
            return np.broadcast_to(np.uint8(2), (len(times),3,480,640))
        reader = RoboDojoHistoryReader(ep, 'cover_blocks', states, actions, '.', 'data', decode)
        item = reader.window_at(0, history=8, chunk=3)
        np.testing.assert_array_equal(item['state'], np.tile(states[0], (8,1)))
        np.testing.assert_array_equal(item['command_history'], np.tile(states[0], (8,1)))
        np.testing.assert_array_equal(item['action'], actions[:3])
        self.assertEqual(calls[0][:8], [10.]*8)
        self.assertEqual(item['observation_valid_mask'].tolist(), [False]*7+[True])

    def test_missing_camera_never_crosses_episode_and_actions_remain_real(self):
        ep = episode()
        camera = 'observation.images.cam_high'
        ep['videos/'+camera+'/to_timestamp'] = 10.16  # Four images, five real actions.
        p = window_plan(ep, 4, history=3, chunk=4)
        self.assertEqual(p['action_time_mask'], [True,False,False,False])
        self.assertEqual(p['command_indices'], [1,2,3])
        self.assertEqual(p['cameras']['images.head']['source_indices'], [2,3,3,3,3,3,3])
        self.assertTrue(all(t < 10.16 for t in p['cameras']['images.head']['timestamps']))
        self.assertEqual(p['video_valid_frames'], [True,True,False,False,False,False,False])

    def test_all_anchors_and_long_history_preserved(self):
        ep = episode(5)
        for t in range(5):
            p = window_plan(ep, t, history=100, chunk=32)
            self.assertEqual(p['state_indices'][-1], t)
            self.assertEqual(sum(p['action_time_mask']), 5-t)
            self.assertTrue(all(x is None or x < t for x in p['command_indices']))

    def test_other_task_rejected(self):
        ep = episode()
        ep['episode_index'] = 400
        with self.assertRaises(ValueError):
            RoboDojoHistoryReader(ep, 'cover_blocks', np.zeros((5,14)), np.zeros((5,14)), '.', 'data', None)


if __name__ == '__main__':
    unittest.main()
