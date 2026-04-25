import types
import unittest

from sglang.srt.managers.io_struct import (
    CommitBalloonReqInput,
    GetBalloonStatusReqInput,
    RestoreFromBalloonReqInput,
    SyncKVCapacityReqInput,
)
from sglang.srt.managers.scheduler import Scheduler


class TestKunservePhase4Scheduler(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_total_num_tokens = 128
        scheduler.expand_requested = True
        scheduler.expand_request_reason = "retract_decode"
        scheduler.balloon_keepalive_step_ct = 0
        scheduler._balloon_keepalive_active = False
        scheduler.waiting_queue = []
        scheduler.running_batch = types.SimpleNamespace(reqs=[], is_empty=lambda: True)
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.enable_overlap = False
        scheduler.pp_size = 1
        scheduler.disaggregation_mode = None
        scheduler.model_worker = types.SimpleNamespace(max_total_num_tokens=128)
        scheduler.tp_worker = types.SimpleNamespace(
            max_total_num_tokens=128,
            get_balloon_status=lambda req: {
                "state": "prepared",
                "runtime_variant": "global",
                "max_total_num_tokens": 128,
            },
            commit_balloon=lambda req: {
                "state": "balloon",
                "runtime_variant": req.target_variant,
                "max_total_num_tokens": 192,
            },
            sync_kv_capacity=lambda req: {
                "state": "balloon",
                "runtime_variant": "global",
                "max_total_num_tokens": (
                    req.max_total_num_tokens
                    if req.max_total_num_tokens is not None
                    else 128 + req.delta_slots
                ),
            },
            restore_from_balloon=lambda req: {
                "state": "local",
                "runtime_variant": "local",
                "max_total_num_tokens": 128,
            },
        )
        scheduler._is_no_request = lambda: True
        return scheduler

    def test_commit_balloon_updates_capacity_and_clears_expand_request(self):
        scheduler = self._make_scheduler()

        result = scheduler.commit_balloon(
            CommitBalloonReqInput(target_variant="global", offload_local_experts=2)
        )

        self.assertTrue(result.success)
        self.assertEqual(scheduler.max_total_num_tokens, 192)
        self.assertEqual(scheduler.tp_worker.max_total_num_tokens, 192)
        self.assertFalse(scheduler.expand_requested)
        self.assertIsNone(scheduler.expand_request_reason)

    def test_sync_kv_capacity_updates_scheduler_cache(self):
        scheduler = self._make_scheduler()

        result = scheduler.sync_kv_capacity(SyncKVCapacityReqInput(delta_slots=32))

        self.assertTrue(result.success)
        self.assertEqual(result.status["max_total_num_tokens"], 160)
        self.assertEqual(scheduler.max_total_num_tokens, 160)

    def test_restore_from_balloon_requires_idle_when_requested(self):
        scheduler = self._make_scheduler()
        scheduler._is_no_request = lambda: False

        result = scheduler.restore_from_balloon(
            RestoreFromBalloonReqInput(require_idle=True)
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.status,
            scheduler.get_balloon_status(GetBalloonStatusReqInput()).status,
        )

    def test_balloon_keepalive_batch_only_runs_in_balloon_state(self):
        scheduler = self._make_scheduler()
        scheduler.get_idle_batch = lambda: "KEEPALIVE"
        scheduler.tp_worker.get_balloon_status = lambda req: {
            "state": "balloon",
            "runtime_variant": "global",
            "offloaded_local_experts": 2,
            "added_kv_slots": 64,
            "max_total_num_tokens": 192,
        }

        batch = scheduler._maybe_get_balloon_keepalive_batch()

        self.assertEqual(batch, "KEEPALIVE")
        self.assertEqual(scheduler.balloon_keepalive_step_ct, 1)
        self.assertTrue(scheduler._balloon_keepalive_active)

        scheduler.tp_worker.get_balloon_status = lambda req: {
            "state": "local",
            "runtime_variant": "local",
            "max_total_num_tokens": 128,
        }
        batch = scheduler._maybe_get_balloon_keepalive_batch()
        self.assertIsNone(batch)
        self.assertFalse(scheduler._balloon_keepalive_active)
