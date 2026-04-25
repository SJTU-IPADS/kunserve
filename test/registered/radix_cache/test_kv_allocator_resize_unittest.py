import unittest

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)


class _DummyKVCache:
    def get_cpu_copy(self, indices):
        return indices

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return None


class TestKVAllocatorResize(unittest.TestCase):
    def test_token_allocator_expand_and_shrink(self):
        allocator = TokenToKVPoolAllocator(
            size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=_DummyKVCache(),
            need_sort=False,
        )

        self.assertEqual(allocator.alloc(2).tolist(), [1, 2])
        self.assertEqual(allocator.available_size(), 2)

        allocator.expand_by_slots(2)
        self.assertEqual(allocator.size, 6)
        self.assertEqual(allocator.available_size(), 4)

        allocator.shrink_tail(2)
        self.assertEqual(allocator.size, 4)
        self.assertEqual(allocator.available_size(), 2)

    def test_token_allocator_rejects_used_tail_shrink(self):
        allocator = TokenToKVPoolAllocator(
            size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=_DummyKVCache(),
            need_sort=False,
        )

        allocator.expand_by_slots(2)
        self.assertEqual(allocator.alloc(6).tolist(), [1, 2, 3, 4, 5, 6])
        with self.assertRaises(RuntimeError):
            allocator.shrink_tail(2)

    def test_paged_allocator_expand_and_shrink(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=8,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=_DummyKVCache(),
            need_sort=False,
        )

        self.assertEqual(allocator.alloc(4).tolist(), [4, 5, 6, 7])
        self.assertEqual(allocator.available_size(), 4)

        allocator.expand_by_slots(4)
        self.assertEqual(allocator.size, 12)
        self.assertEqual(allocator.available_size(), 8)

        allocator.shrink_tail(4)
        self.assertEqual(allocator.size, 8)
        self.assertEqual(allocator.available_size(), 4)

    def test_paged_allocator_requires_alignment(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=8,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=_DummyKVCache(),
            need_sort=False,
        )

        with self.assertRaises(ValueError):
            allocator.expand_by_slots(2)
        with self.assertRaises(ValueError):
            allocator.shrink_tail(2)
