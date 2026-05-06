# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args


_KUNSERVE_DBG_LAST_SIG = {}


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        ep_dispatch_algorithm = get_global_server_args().ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if ep_dispatch_algorithm is None:
            # KUNSERVE-DBG: noisy on every call, but only at the warning level
            # the first time this branch is taken per layer per process so we
            # can see if the static-remap path is silently disabled.
            sig = ("none", layer_id)
            if _KUNSERVE_DBG_LAST_SIG.get(sig) is None:
                _logger.warning(
                    "[KUNSERVE-DBG] ExpertLocationDispatchInfo.init_new: "
                    "ep_dispatch_algorithm=None (no logical->physical remap will be applied) layer_id=%d",
                    layer_id,
                )
                _KUNSERVE_DBG_LAST_SIG[sig] = True
            return None

        partial_dispatch = (
            expert_location_metadata.logical_to_rank_dispatch_physical_map[
                layer_id, :
            ]
            if expert_location_metadata.logical_to_rank_dispatch_physical_map
            is not None
            else None
        )

        # KUNSERVE-DBG: log the first 8 entries of the dispatch map for layer 0
        # once per (process, layer_id) so we can confirm the values flip from
        # LOCAL identity (logical i -> i) to GLOBAL complementary (e.g. logical
        # 32 -> physical 64) at commit_balloon time.
        if layer_id == 0:
            sig = ("layer0", id(partial_dispatch))
            if _KUNSERVE_DBG_LAST_SIG.get(sig) is None:
                _KUNSERVE_DBG_LAST_SIG[sig] = True
                if partial_dispatch is None:
                    _logger.warning(
                        "[KUNSERVE-DBG] ExpertLocationDispatchInfo.init_new layer_id=0: "
                        "partial_logical_to_rank_dispatch_physical_map is None (algo=%s)",
                        ep_dispatch_algorithm,
                    )
                else:
                    sample = partial_dispatch[:8].tolist() if partial_dispatch.numel() >= 8 else partial_dispatch.tolist()
                    sample32 = (
                        int(partial_dispatch[32].item()) if partial_dispatch.numel() > 32 else None
                    )
                    sample64 = (
                        int(partial_dispatch[64].item()) if partial_dispatch.numel() > 64 else None
                    )
                    _logger.warning(
                        "[KUNSERVE-DBG] ExpertLocationDispatchInfo.init_new layer_id=0 algo=%s "
                        "first8=%s logical32->phys=%s logical64->phys=%s map_id=0x%x",
                        ep_dispatch_algorithm,
                        sample,
                        sample32,
                        sample64,
                        id(partial_dispatch),
                    )

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=partial_dispatch,
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
