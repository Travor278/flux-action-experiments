"""Episode-scoped CPU history reader, pending FLUX trainer/loss integration.

All real action rows are retained. Missing images are held causally within the
same episode and marked invalid as video targets, never borrowed from neighbours.
History length is explicit: this module does not choose a Memory task horizon.
"""
from pathlib import Path
import numpy as np
from video_boundary import camera_timestamp

CAMERAS = {
    'images.head': 'observation.images.cam_high',
    'images.left_wrist': 'observation.images.cam_left_wrist',
    'images.right_wrist': 'observation.images.cam_right_wrist',
}
SELECTED = {'cover_blocks': range(300, 400), 'match_and_pick_from_conveyor': range(1600, 1700)}


def window_plan(episode, start, history, chunk, fps=25):
    for value in (start, history, chunk):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('window arguments must be integers')
    length = episode['length']
    if not 0 <= start < length or history < 1 or chunk < 1:
        raise ValueError('invalid window')
    obs = list(range(start-history+1, start+1))
    image_indices = obs + list(range(start+1, start+chunk+1))
    plan = {
        'state_indices': [max(0, t) for t in obs],
        'observation_valid_mask': [t >= 0 for t in obs],
        'command_indices': [t-1 if t > 0 else None for t in obs],
        'action_indices': [min(t, length-1) for t in range(start, start+chunk)],
        'action_time_mask': [t < length for t in range(start, start+chunk)],
        'cameras': {},
    }
    for key, camera in CAMERAS.items():
        prefix = 'videos/' + camera
        # camera_timestamp validates integral half-open source bounds first.
        assert camera_timestamp(episode, camera, 0, fps) is not None
        available = min(length, round((episode[prefix+'/to_timestamp']-episode[prefix+'/from_timestamp'])*fps))
        source_indices = [min(max(t, 0), available-1) for t in image_indices]
        timestamps = [camera_timestamp(episode, camera, t, fps) for t in source_indices]
        assert all(t is not None and t < episode[prefix+'/to_timestamp'] for t in timestamps)
        plan['cameras'][key] = {
            'source_indices': source_indices, 'timestamps': timestamps,
            'valid_mask': [0 <= t < available for t in image_indices],
            'source_camera': camera,
        }
    # Conservative whole-frame target validity for packed multi-camera video.
    plan['video_valid_frames'] = [all(p['valid_mask'][i] for p in plan['cameras'].values()) for i in range(history+chunk)]
    return plan


class RoboDojoHistoryReader:
    """Read one selected episode; decoder(path, timestamps) returns uint8 TCHW.

    This reader emits masks; an unmodified FLUX trainer does NOT consume them.
    No formal training may use it until masked loss integration is verified.
    """
    def __init__(self, episode, task, states, actions, source_root, prefix, decoder):
        if task not in SELECTED or episode['episode_index'] not in SELECTED[task]:
            raise ValueError('episode outside selected training task')
        self.episode, self.task, self.decoder = episode, task, decoder
        self.root, self.prefix = Path(source_root).resolve(), prefix
        self.states, self.actions = np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)
        if self.states.shape != (episode['length'], 14) or self.actions.shape != self.states.shape:
            raise ValueError('numeric row shape mismatch')
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError('nonfinite numeric row')
        if not episode['tasks'] or not all(isinstance(x, str) and x for x in episode['tasks']):
            raise ValueError('missing source instructions')

    def window_at(self, start, *, history, chunk, caption_index=0, prefix_ticks=None):
        ep = self.episode
        p = window_plan(ep, start, history, chunk)
        # First observation/reset command matches official ObservationHistory:
        # measured initial state, never action[0] or the previous episode action.
        commands = [self.states[0] if t is None else self.actions[t] for t in p['command_indices']]
        item = {
            'state': self.states[p['state_indices']].copy(),
            'command_history': np.stack(commands).copy(),
            'action': self.actions[p['action_indices']].copy(),
            'action_prev': (self.states[0] if start == 0 else self.actions[start-1]).copy(),
            'action_time_mask': np.asarray(p['action_time_mask'], dtype=bool),
            'video_valid_frames': np.asarray(p['video_valid_frames'], dtype=bool),
            'observation_valid_mask': np.asarray(p['observation_valid_mask'], dtype=bool),
            'task': ep['tasks'][caption_index], 'episode_index': ep['episode_index'], 'start': start,
        }
        prefix_indices = None
        if prefix_ticks is not None:
            from prefix_memory import PrefixMemorySpec
            prefix_indices = PrefixMemorySpec(tuple(prefix_ticks)).indices(start)
            item['prefix_tick_indices'] = np.asarray(prefix_indices,dtype=np.int64)
            item['episode_tick'] = start
        for key, camera_plan in p['cameras'].items():
            k = 'videos/' + camera_plan['source_camera']
            relative = f"{self.prefix}/videos/{camera_plan['source_camera']}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('video source outside dataset root')
            frames = self.decoder(path, camera_plan['timestamps'])
            if frames.dtype != np.uint8 or frames.shape != (history+chunk, 3, 480, 640):
                raise ValueError('decoder must return native uint8 RGB TCHW')
            item[key] = frames
            item[key+'_valid_mask'] = np.asarray(camera_plan['valid_mask'], dtype=bool)
            if prefix_indices is not None:
                camera = camera_plan['source_camera']
                available = min(ep['length'],round((ep[k+'/to_timestamp']-ep[k+'/from_timestamp'])*25))
                timestamps = [camera_timestamp(ep,camera,min(t,available-1)) for t in prefix_indices]
                prefix_frames = self.decoder(path,timestamps)
                if prefix_frames.dtype != np.uint8 or prefix_frames.shape != (len(prefix_indices),3,480,640):
                    raise ValueError('invalid prefix RGB shape')
                item['prefix.'+key] = prefix_frames
                item['prefix.'+key+'_valid_mask'] = np.asarray([t < available for t in prefix_indices],dtype=bool)
        return item
