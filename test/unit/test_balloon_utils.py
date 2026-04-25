import unittest

import torch

from sglang.srt.model_executor.balloon_utils import (
    build_dispatcher_local_expert_mapping,
    resolve_balloon_kv_slots_to_expand,
    slice_rank_local_logical_expert_ids,
)


class TestBuildDispatcherLocalExpertMapping(unittest.TestCase):
    def test_reindexes_active_suffix_into_compact_local_rows(self):
        dispatcher_mapping = build_dispatcher_local_expert_mapping(
            num_logical_experts=8,
            local_logical_expert_ids=[0, 4, 1, 5],
            active_local_expert_mapping=[2, 3],
        )

        self.assertTrue(
            torch.equal(
                dispatcher_mapping,
                torch.tensor([-1, 0, -1, -1, -1, 1, -1, -1], dtype=torch.int32),
            )
        )

    def test_rejects_non_contiguous_active_rows(self):
        with self.assertRaisesRegex(ValueError, "contiguous local expert slice"):
            build_dispatcher_local_expert_mapping(
                num_logical_experts=8,
                local_logical_expert_ids=[0, 1, 2, 3],
                active_local_expert_mapping=[0, 2],
            )

    def test_slices_local_runtime_rows_before_reindexing_for_balloon(self):
        local_logical_expert_ids = slice_rank_local_logical_expert_ids(
            physical_to_logical_map=[list(range(128))],
            layer_id=0,
            moe_ep_rank=1,
            num_local_physical_experts=64,
        )

        dispatcher_mapping = build_dispatcher_local_expert_mapping(
            num_logical_experts=128,
            local_logical_expert_ids=local_logical_expert_ids,
            active_local_expert_mapping=list(range(32, 64)),
        )

        expected = torch.full((128,), -1, dtype=torch.int32)
        expected[96:128] = torch.arange(32, dtype=torch.int32)
        self.assertTrue(torch.equal(dispatcher_mapping, expected))


class TestResolveBalloonKvSlotsToExpand(unittest.TestCase):
    def test_caps_default_request_to_kv_vmm_headroom(self):
        self.assertEqual(
            resolve_balloon_kv_slots_to_expand(
                max_slots_from_donor=4096,
                kv_vmm_headroom_slots=1024,
                num_slots_to_expand=None,
            ),
            1024,
        )

    def test_respects_explicit_request_within_donor_and_headroom(self):
        self.assertEqual(
            resolve_balloon_kv_slots_to_expand(
                max_slots_from_donor=4096,
                kv_vmm_headroom_slots=1024,
                num_slots_to_expand=512,
            ),
            512,
        )

    def test_rejects_explicit_request_beyond_kv_vmm_headroom(self):
        with self.assertRaisesRegex(ValueError, "KV VMM reserve only supports"):
            resolve_balloon_kv_slots_to_expand(
                max_slots_from_donor=4096,
                kv_vmm_headroom_slots=1024,
                num_slots_to_expand=2048,
            )


if __name__ == "__main__":
    unittest.main()
