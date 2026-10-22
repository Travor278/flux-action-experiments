"""Strict v3 video episode bounds. Never silently read a neighbouring episode."""
import math


def camera_timestamp(episode, camera, frame_index, fps=25):
    """Return a source timestamp, or None when this camera lacks the observation.

    None must be handled explicitly by the dataset's causal padding/validity mask;
    it must never be passed through as a timestamp or replaced with the next clip.
    This helper does not implement or approve a training padding policy.
    """
    if not isinstance(frame_index, int) or not 0 <= frame_index < episode['length']:
        raise IndexError('frame index outside episode')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('invalid fps')
    key = 'videos/' + camera
    start, end = episode[key + '/from_timestamp'], episode[key + '/to_timestamp']
    span = (end - start) * fps
    if not math.isfinite(span) or span <= 0 or abs(span - round(span)) > 0.01:
        raise ValueError('non-integral or invalid video span')
    # Round only metadata floating-point error, never stretch a shorter video.
    valid_frames = min(episode['length'], round(span))
    return start + frame_index / fps if frame_index < valid_frames else None
