import unittest
from video_boundary import camera_timestamp


class VideoBoundaryTests(unittest.TestCase):
    def episode(self, length, start, end):
        return {'length': length, 'videos/cam/from_timestamp': start, 'videos/cam/to_timestamp': end}

    def test_actual_truncated_source_boundary(self):
        ep = self.episode(541, 283.28, 304.88)
        self.assertAlmostEqual(camera_timestamp(ep, 'cam', 539), 304.84)
        self.assertIsNone(camera_timestamp(ep, 'cam', 540))
        # The same timestamp is valid only for the NEXT episode, frame zero.
        next_ep = self.episode(500, 304.88, 324.88)
        self.assertAlmostEqual(camera_timestamp(next_ep, 'cam', 0), 304.88)

    def test_last_valid_frame_and_float_metadata(self):
        ep = self.episode(555, 352.15999999999997, 374.35999999999996)
        self.assertAlmostEqual(camera_timestamp(ep, 'cam', 554), 374.32)
        for index in (-1, 555):
            with self.assertRaises(IndexError):
                camera_timestamp(ep, 'cam', index)

    def test_malformed_span_rejected(self):
        with self.assertRaises(ValueError):
            camera_timestamp(self.episode(10, 0, .395), 'cam', 0)


if __name__ == '__main__':
    unittest.main()
