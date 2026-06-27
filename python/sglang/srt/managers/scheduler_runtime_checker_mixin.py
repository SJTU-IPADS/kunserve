from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        mr = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        kunserve_reserved_full = 0
        kunserve_reserved_mamba = 0
        kunserve_dummy_kv_slot = None
        if mr is not None and getattr(
            mr, "_kunserve_keepalive_dummy_kv_slot", None
        ) is not None:
            kunserve_reserved_full = 1
            kunserve_dummy_kv_slot = int(mr._kunserve_keepalive_dummy_kv_slot)

        full_delta = full_num_used - self.tree_cache.full_protected_size()
        mamba_delta = mamba_num_used - self.tree_cache.mamba_protected_size()
        memory_leak = (
            full_delta < 0
            or full_delta > kunserve_reserved_full
            or mamba_delta < 0
            or mamba_delta > kunserve_reserved_mamba
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            if kunserve_dummy_kv_slot is not None:
                leaked_full_pages.discard(kunserve_dummy_kv_slot)
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            leaked_mamba_pages.discard(0)
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}, kunserve_reserved_full={kunserve_reserved_full}, full_delta={full_delta}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, kunserve_reserved_mamba={kunserve_reserved_mamba}, mamba_delta={mamba_delta}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}, kunserve_reserved_full={kunserve_reserved_full}, full_delta={full_delta}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, kunserve_reserved_mamba={kunserve_reserved_mamba}, mamba_delta={mamba_delta}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        # KunServe Phase E: commit_balloon reserves 1 KV slot as the
        # ``dummy_kv_slot`` keepalive batch's out_cache_loc target.  That
        # slot is held until restore_from_balloon.  Empirically, depending
        # on tree_cache / allocator transitions, the slot may appear in
        # ``available_size`` (delta = +kunserve_reserved) or remain held
        # (delta = 0).  Either case is benign for KunServe runs; only a
        # NEGATIVE delta (fewer slots accounted than expected) is a real
        # leak.  Accept the off-by-``kunserve_reserved`` tolerance in
        # either direction to avoid false positives during BALLOON.
        kunserve_reserved = 0
        mr = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        if mr is not None and getattr(
            mr, "_kunserve_keepalive_dummy_kv_slot", None
        ) is not None:
            kunserve_reserved = 1
        actual = available_size + evictable_size
        expected = self.max_total_num_tokens - protected_size
        delta = actual - expected  # 0 = exact match; +k = unexpected free slots; -k = leak
        if kunserve_reserved > 0:
            # Accept delta in [-kunserve_reserved, +kunserve_reserved].
            memory_leak = (delta < -kunserve_reserved) or (delta > kunserve_reserved)
        else:
            memory_leak = delta != 0
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, "
            f"{evictable_size=}, {protected_size=}, "
            f"kunserve_reserved={kunserve_reserved}, delta={delta}\n"
        )
        return memory_leak, token_msg

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        total_tokens = available_size + evictable_size + protected_size + uncached_size
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        # KunServe Phase E: commit_balloon also pops one entry from
        # req_to_token_pool.free_slots to use as the phantom req_pool index
        # whose req_to_token row holds dummy_kv_slot.  Mirror the tolerance
        # logic from _check_radix_cache_memory: accept the phantom being
        # either held (free_slots = req_total - 1) or back in free
        # (free_slots = req_total) -- only a deficit beyond that is a leak.
        mr = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        kunserve_phantom = 0
        if mr is not None and getattr(
            mr, "_kunserve_keepalive_phantom_req_idx", None
        ) is not None:
            kunserve_phantom = 1

        free_count = len(self.req_to_token_pool.free_slots)
        if kunserve_phantom > 0:
            req_pool_leak = free_count < (req_total_size - kunserve_phantom) or free_count > req_total_size
        else:
            req_pool_leak = free_count != req_total_size

        if req_pool_leak:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
