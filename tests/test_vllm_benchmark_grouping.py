import unittest

from benchmarks.event_grouping import group_event_indices


class EventGroupingTests(unittest.TestCase):
    def test_single_forward_prefill(self):
        self.assertEqual(
            group_event_indices([512, 512], "prefill", batch=32, context=16, iterations=2),
            [[0], [1]],
        )

    def test_chunked_prefill(self):
        self.assertEqual(
            group_event_indices(
                [16, 496, 16, 496],
                "prefill",
                batch=32,
                context=16,
                iterations=2,
            ),
            [[0, 1], [2, 3]],
        )

    def test_decode_uses_only_decode_forward(self):
        self.assertEqual(
            group_event_indices(
                [16, 1, 16, 1],
                "decode",
                batch=1,
                context=16,
                iterations=2,
            ),
            [[1], [3]],
        )

    def test_decode_accepts_native_sp_local_token_count(self):
        groups = group_event_indices(
            [16, 4, 16, 4],
            "decode",
            batch=1,
            context=16,
            iterations=2,
            decode_token_count=4,
        )
        self.assertEqual(groups, [[1], [3]])

    def test_incomplete_logical_request_fails(self):
        with self.assertRaises(ValueError):
            group_event_indices([16, 496], "decode", batch=1, context=16, iterations=1)


if __name__ == "__main__":
    unittest.main()
