import unittest
from contextlib import ExitStack

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils.cuda_vmm import (
    ExpandableVmmTensor,
    MoeWeightDonorManager,
    cuda_vmm_available,
)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestCudaVmmPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not cuda_vmm_available():
            raise unittest.SkipTest("CUDA VMM support is unavailable in this environment.")

    def test_moe_weight_donor_restore_roundtrip(self):
        w13 = ExpandableVmmTensor(
            reserve_shape=(2, 1024, 1024),
            dtype=torch.float16,
            active_rows=2,
            label="test_w13",
        )
        w2 = ExpandableVmmTensor(
            reserve_shape=(2, 1024, 1024),
            dtype=torch.float16,
            active_rows=2,
            label="test_w2",
        )

        w13.tensor[0].fill_(1)
        w13.tensor[1].fill_(2)
        w2.tensor[0].fill_(3)
        w2.tensor[1].fill_(4)
        torch.cuda.synchronize()

        donor_manager = MoeWeightDonorManager(
            layer_id=0,
            allocations={"w13_weight": w13, "w2_weight": w2},
        )
        original_bytes = donor_manager.total_bytes()

        borrowed = donor_manager.borrow_tail_experts(1)
        self.assertEqual(len(borrowed), 2)
        self.assertEqual(donor_manager.max_borrowable_experts(), 1)
        self.assertLess(donor_manager.total_bytes(), original_bytes)

        donor_manager.restore_all()
        self.assertEqual(donor_manager.total_bytes(), original_bytes)
        self.assertEqual(donor_manager.max_borrowable_experts(), 2)
        self.assertTrue(torch.allclose(w13.tensor[1], torch.full_like(w13.tensor[1], 2)))
        self.assertTrue(torch.allclose(w2.tensor[1], torch.full_like(w2.tensor[1], 4)))

    def test_moe_weight_head_borrow_restore_roundtrip(self):
        weight = ExpandableVmmTensor(
            reserve_shape=(4, 1024, 1024),
            dtype=torch.float16,
            active_rows=4,
            label="test_head_borrow",
            wrap_full_tensor=True,
        )
        for idx in range(4):
            weight.tensor[idx].fill_(idx + 1)
        torch.cuda.synchronize()

        donor0 = weight.borrow_head_rows(1, owner={"kind": "expert_head", "idx": 0})
        donor1 = weight.borrow_head_rows(1, owner={"kind": "expert_head", "idx": 1})
        self.assertEqual(weight.mapped_start_row, 2)
        self.assertEqual(weight.active_rows, 2)
        self.assertTrue(torch.allclose(weight.tensor[2], torch.full_like(weight.tensor[2], 3)))
        self.assertTrue(torch.allclose(weight.tensor[3], torch.full_like(weight.tensor[3], 4)))

        donor1.restore_to_source()
        donor0.restore_to_source()
        weight.sync_from_region()
        self.assertEqual(weight.mapped_start_row, 0)
        self.assertEqual(weight.active_rows, 4)
        self.assertTrue(torch.allclose(weight.tensor[0], torch.full_like(weight.tensor[0], 1)))
        self.assertTrue(torch.allclose(weight.tensor[1], torch.full_like(weight.tensor[1], 2)))

    def test_mha_kv_pool_vmm_expand_and_shrink(self):
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_EXPERIMENTAL_CUDA_VMM.override(True))
            stack.enter_context(envs.SGLANG_EXPERIMENTAL_VMM_KV_CACHE.override(True))
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_VMM_KV_RESERVE_SLOTS.override(8)
            )

            pool = MHATokenToKVPool(
                size=16,
                page_size=1,
                dtype=torch.float16,
                head_num=2,
                head_dim=8,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
            )

        self.assertTrue(pool._kv_vmm_enabled)
        self.assertEqual(pool.get_key_buffer(0).shape[0], 17)
        pool.k_buffer[0][1:4].fill_(5)
        expected = pool.k_buffer[0][1:4].clone()

        pool.expand_by_slots(8)
        self.assertEqual(pool.size, 24)
        self.assertEqual(pool.get_key_buffer(0).shape[0], 25)
        self.assertTrue(torch.allclose(pool.k_buffer[0][1:4], expected))

        returned = pool.shrink_tail(8)
        self.assertEqual(pool.size, 16)
        self.assertEqual(pool.get_key_buffer(0).shape[0], 17)
        self.assertGreaterEqual(returned.total_bytes, 0)
