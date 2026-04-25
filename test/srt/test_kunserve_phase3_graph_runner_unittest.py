import types
import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


class _FakeGraph:
    def __init__(self):
        self.replayed = 0

    def replay(self):
        self.replayed += 1


class _FakeSpecAlgorithm:
    def is_ngram(self):
        return False

    def is_eagle(self):
        return False

    def is_standalone(self):
        return False


class TestKunservePhase3GraphRunner(unittest.TestCase):
    def _make_runner(self, runtime_variant="global", replay_enabled=True):
        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = False
        runner.enable_pdmux = False
        runner.disable_padding = False
        runner.capture_bs = [1, 4, 8]
        runner.max_bs = 8
        runner.is_encoder_decoder = False
        runner.is_dllm = False
        runner.enable_two_batch_overlap = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.num_tokens_per_bs = 1
        runner.model_runner = types.SimpleNamespace(
            get_cuda_graph_runtime_variant=lambda: runtime_variant,
            is_cuda_graph_replay_enabled=lambda: replay_enabled,
            spec_algorithm=_FakeSpecAlgorithm(),
        )
        runner.graphs = {
            ("local", 4): _FakeGraph(),
            ("global", 4): _FakeGraph(),
            ("local", 8): _FakeGraph(),
            ("global", 8): _FakeGraph(),
        }
        runner.output_buffers = {
            ("global", 4): LogitsProcessorOutput(
                next_token_logits=torch.arange(8, dtype=torch.float32).view(4, 2)
            ),
            ("local", 4): LogitsProcessorOutput(
                next_token_logits=torch.full((4, 2), -1.0)
            ),
        }
        return runner

    def test_can_run_is_variant_aware(self):
        runner = self._make_runner(runtime_variant="global", replay_enabled=True)
        forward_batch = types.SimpleNamespace(
            batch_size=3,
            can_run_dp_cuda_graph=True,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_info=types.SimpleNamespace(capture_hidden_mode=None),
            can_run_tbo=True,
            input_ids=torch.zeros(3, dtype=torch.int64),
        )

        self.assertTrue(runner.can_run(forward_batch))
        runner.model_runner = types.SimpleNamespace(
            get_cuda_graph_runtime_variant=lambda: "missing",
            is_cuda_graph_replay_enabled=lambda: True,
            spec_algorithm=_FakeSpecAlgorithm(),
        )
        self.assertFalse(runner.can_run(forward_batch))

    def test_can_run_respects_graph_replay_gate(self):
        runner = self._make_runner(replay_enabled=False)
        forward_batch = types.SimpleNamespace(
            batch_size=2,
            can_run_dp_cuda_graph=True,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_info=types.SimpleNamespace(capture_hidden_mode=None),
            can_run_tbo=True,
            input_ids=torch.zeros(2, dtype=torch.int64),
        )
        self.assertFalse(runner.can_run(forward_batch))

    def test_replay_uses_runtime_variant_key(self):
        runner = self._make_runner(runtime_variant="global", replay_enabled=True)
        runner.deepep_adapter = types.SimpleNamespace(replay=lambda: None)
        runner.raw_num_token = 2
        runner.bs = 4
        runner.buffers = types.SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
        )
        forward_batch = types.SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int64),
        )

        output = runner.replay(forward_batch, skip_attn_backend_init=True)

        self.assertEqual(runner.graphs[("global", 4)].replayed, 1)
        self.assertEqual(runner.graphs[("local", 4)].replayed, 0)
        self.assertTrue(
            torch.equal(
                output.next_token_logits,
                torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            )
        )

    def test_get_captured_variants(self):
        runner = self._make_runner()
        self.assertEqual(runner.get_captured_variants(), ["global", "local"])
