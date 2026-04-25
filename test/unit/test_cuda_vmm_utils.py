import unittest

from sglang.srt.utils.cuda_vmm import min_granularity_aligned_row_count


class TestMinGranularityAlignedRowCount(unittest.TestCase):
    def test_returns_one_for_already_aligned_rows(self):
        self.assertEqual(
            min_granularity_aligned_row_count(2 * 1024 * 1024, 2 * 1024 * 1024),
            1,
        )

    def test_returns_smallest_chunk_for_three_mib_rows(self):
        self.assertEqual(
            min_granularity_aligned_row_count(3 * 1024 * 1024, 2 * 1024 * 1024),
            2,
        )

    def test_handles_rows_smaller_than_granularity(self):
        self.assertEqual(
            min_granularity_aligned_row_count(1024 * 1024, 2 * 1024 * 1024),
            2,
        )

    def test_rejects_non_positive_row_bytes(self):
        with self.assertRaisesRegex(ValueError, "row_bytes must be positive"):
            min_granularity_aligned_row_count(0, 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
