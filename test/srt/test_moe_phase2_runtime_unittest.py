import types
import unittest
from unittest import mock

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPBuffer,
    DeepEPDispatchMode,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode


class _FakeDeepEPConfig:
    def __init__(self):
        self.normal_dispatch_config = None
        self.normal_combine_config = None
        self.num_sms = 20


class _FakeDeepEPConfigHint:
    def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes + group_size

    def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes + 2 * group_size


class _FakeDeepEPBufferImpl:
    num_sms = 20

    def __init__(
        self,
        group,
        num_nvl_bytes,
        num_rdma_bytes,
        low_latency_mode,
        num_qps_per_rank,
        allow_mnnvl,
    ):
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.allow_mnnvl = allow_mnnvl
        self.cleaned = 0

    @staticmethod
    def get_dispatch_config(group_size):
        return _FakeDeepEPConfigHint()

    @staticmethod
    def get_combine_config(group_size):
        return _FakeDeepEPConfigHint()

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank,
        hidden_size,
        group_size,
        num_experts,
    ):
        return (
            num_max_dispatch_tokens_per_rank
            + hidden_size
            + group_size
            + num_experts
        )

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank, hidden_size, num_experts
    ):
        self.cleaned += 1


class _FakeGroup:
    def __init__(self, world_size):
        self._world_size = world_size

    def size(self):
        return self._world_size


class TestMoePhase2Runtime(unittest.TestCase):
    def setUp(self):
        DeepEPBuffer._buffer_cache = {}
        DeepEPBuffer._dispatch_mode_by_group = {}
        DeepEPBuffer._default_dispatch_mode = None

    def test_expert_location_builder_supports_explicit_ep_size_and_rank(self):
        server_args = types.SimpleNamespace(
            device="cpu",
            ep_num_redundant_experts=0,
            ep_size=4,
            nnodes=1,
            ep_dispatch_algorithm="static",
        )
        model_config = types.SimpleNamespace()
        physical_to_logical_map = torch.arange(8).repeat(2, 1)

        with mock.patch.object(
            ModelConfigForExpertLocation,
            "from_model_config",
            return_value=ModelConfigForExpertLocation(
                num_layers=2, num_logical_experts=8
            ),
        ):
            local_meta = ExpertLocationMetadata.init_by_mapping(
                server_args=server_args,
                model_config=model_config,
                physical_to_logical_map=physical_to_logical_map,
                moe_ep_rank=1,
                ep_size_override=4,
                dispatch_ep_rank=1,
            )
            global_meta = ExpertLocationMetadata.init_by_mapping(
                server_args=server_args,
                model_config=model_config,
                physical_to_logical_map=physical_to_logical_map,
                moe_ep_rank=5,
                ep_size_override=8,
                dispatch_ep_rank=5,
            )

        self.assertEqual(local_meta.ep_size, 4)
        self.assertEqual(global_meta.ep_size, 8)
        self.assertEqual(global_meta.dispatch_ep_rank, 5)
        self.assertEqual(
            tuple(global_meta.logical_to_rank_dispatch_physical_map.shape), (2, 8)
        )

        local_meta.update(global_meta, [0, 1])
        self.assertEqual(local_meta.ep_size, 8)
        self.assertEqual(local_meta.dispatch_ep_rank, 5)

    def test_standard_dispatcher_supports_explicit_local_mapping(self):
        moe_runner_config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=4,
            num_fused_shared_experts=0,
            hidden_size=16,
            top_k=2,
        )
        dispatcher = StandardDispatcher(
            moe_runner_config,
            moe_ep_size=8,
            moe_ep_rank=3,
            local_expert_mapping=torch.tensor(
                [-1, -1, 0, 1, -1, -1, 2, 3], dtype=torch.int32
            ),
        )

        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((1, 2), dtype=torch.float32),
            topk_ids=torch.tensor([[2, 6]], dtype=torch.int64),
            router_logits=torch.zeros((1, 8), dtype=torch.float32),
        )
        dispatch_output = dispatcher.dispatch(
            hidden_states=torch.zeros((1, 16), dtype=torch.float32),
            topk_output=topk_output,
        )

        self.assertTrue(
            torch.equal(
                dispatch_output.topk_output.topk_ids,
                torch.tensor([[0, 2]], dtype=torch.int32),
            )
        )

    def test_deepep_buffer_is_cached_per_group_and_dispatch_mode(self):
        group_a = _FakeGroup(world_size=4)
        group_b = _FakeGroup(world_size=4)

        deepep_module = "sglang.srt.layers.moe.token_dispatcher.deepep"
        with mock.patch(f"{deepep_module}.Buffer", _FakeDeepEPBufferImpl), mock.patch(
            f"{deepep_module}.DeepEPConfig.get_instance",
            return_value=_FakeDeepEPConfig(),
        ), mock.patch(f"{deepep_module}._is_npu", True):
            normal_a = DeepEPBuffer.get_deepep_buffer(
                group=group_a,
                hidden_size=64,
                param_bytes=2,
                deepep_mode=DeepEPMode.AUTO,
                num_max_dispatch_tokens_per_rank=16,
                num_experts=8,
                dispatch_mode=DeepEPDispatchMode.NORMAL,
            )
            low_latency_a = DeepEPBuffer.get_deepep_buffer(
                group=group_a,
                hidden_size=64,
                param_bytes=2,
                deepep_mode=DeepEPMode.AUTO,
                num_max_dispatch_tokens_per_rank=16,
                num_experts=8,
                dispatch_mode=DeepEPDispatchMode.LOW_LATENCY,
            )
            normal_b = DeepEPBuffer.get_deepep_buffer(
                group=group_b,
                hidden_size=64,
                param_bytes=2,
                deepep_mode=DeepEPMode.AUTO,
                num_max_dispatch_tokens_per_rank=16,
                num_experts=8,
                dispatch_mode=DeepEPDispatchMode.NORMAL,
            )

        self.assertIsNot(normal_a, low_latency_a)
        self.assertIsNot(normal_a, normal_b)
        self.assertFalse(normal_a.low_latency_mode)
        self.assertTrue(low_latency_a.low_latency_mode)
