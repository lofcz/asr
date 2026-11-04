import unittest
from recover import window_segments


class CropCoverage(unittest.TestCase):
    def test_speaker_counts_do_not_exclude(self):
        for count in (1, 2, 5, 12):
            segments = [dict(start=i, end=i+0.5, speaker=str(i), text=f' word{i}') for i in range(count)]
            output = window_segments(segments, 0, count, count)
            self.assertEqual(len(output), count)
            self.assertEqual(''.join(s['Content'] for s in output), ''.join(s['text'] for s in segments))

    def test_zero_duration_labels_are_not_dropped_or_duplicated(self):
        segments = [dict(start=0, end=0, speaker='a', text='Start'),
                    dict(start=1, end=2, speaker='a', text=' middle'),
                    dict(start=5, end=5, speaker='b', text=' boundary'),
                    dict(start=6, end=9, speaker='b', text=' next'),
                    dict(start=10, end=10, speaker='a', text=' end')]
        left = window_segments(segments, 0, 5, 10)
        right = window_segments(segments, 5, 10, 10)
        self.assertEqual([s['Content'] for s in left + right], [s['text'] for s in segments])
        self.assertEqual(right[0]['Start'], 0)
        self.assertEqual(right[-1]['End'], 5)

    def test_unsafe_crop_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cuts speech'):
            window_segments([dict(start=1, end=4, speaker='a', text='word')], 0, 2, 5)


if __name__ == '__main__':
    unittest.main()
