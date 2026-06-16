from __future__ import annotations

import datetime
import logging
import os
import weakref
from typing import Optional, Union

import torch
import torch.distributed as dist

from torch.distributed import ReduceOp

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    buffer_type,
    cudaStream_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
)
from sglang.srt.distributed.parallel_state import (
    _register_group,
    inplace_all_reduce,
    reg_all_gather_into_tensor,
    reg_reduce_scatter_tensor,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.utils.common import get_current_device_stream_fast

logger = logging.getLogger(__name__)


_kunserve_combine_alt_stream: Optional[torch.cuda.Stream] = None


def _alt_stream_combine_env_enabled() -> bool:
    return os.environ.get("KUNSERVE_COMBINE_ALT_STREAM", "0") in (
        "1",
        "true",
        "True",
        "yes",
        "on",
    )


def get_kunserve_combine_alt_stream(device) -> torch.cuda.Stream:
    """Singleton alt CUDA stream used by KunServe combine overlap (Phase B).

    Phase B.1 issues the lane combine NCCL collective on this stream so that
    Phase B.2's chunked-expert work on the main stream can overlap with the
    combine NCCL.  The stream is process-wide and lazily created on first
    request so that ranks that don't enable the env-gate never allocate it.
    """
    global _kunserve_combine_alt_stream
    if _kunserve_combine_alt_stream is None:
        _kunserve_combine_alt_stream = torch.cuda.Stream(device=device)
    return _kunserve_combine_alt_stream


class KunServePyNcclGroup:
    """Graph-safe KunServe collective group backed by SGLang PyNccl.

    KunServe forms cross-replica groups between independent SGLang
    instances, so those ranks are not part of the same default torch
    distributed world.  A normal ``GroupCoordinator`` cannot build such
    groups with ``torch.distributed.new_group``.  This wrapper uses
    ``StatelessProcessGroup`` only for NCCL unique-id exchange and then
    exposes the same registered collective surface that SGLang's TP
    ``GroupCoordinator`` uses during CUDA graph capture.
    """

    kunserve_graph_safe = True

    def __init__(
        self,
        *,
        name: str,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: Optional[Union[int, str, torch.device]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        if int(world_size) <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if int(rank) < 0 or int(rank) >= int(world_size):
            raise ValueError(
                f"rank must be in [0, world_size), got rank={rank}, "
                f"world_size={world_size}."
            )

        self.name = str(name)
        self.unique_name = f"kunserve_pynccl:{self.name}"
        self.rank = int(rank)
        self.rank_in_group = int(rank)
        self.world_size = int(world_size)
        self.ranks = list(range(self.world_size))

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.device_module = torch.get_device_module(self.device)

        self._diag_log_path = os.environ.get("KUNSERVE_DETAIL_LOG")
        self._diag_seen = set()
        self._diag_counts = {}

        self.stateless_group = StatelessProcessGroup.create(
            host=host,
            port=int(port),
            rank=self.rank,
            world_size=self.world_size,
            timeout_seconds=timeout_seconds,
        )
        # Keep graph-captured collectives and normal eager collectives on
        # separate NCCL communicators.  The first post-balloon step can be a
        # large re-prefill (dynamic/eager GLOBAL dispatcher) immediately after
        # GLOBAL CUDA graphs have captured NCCL ops.  Reusing the captured
        # communicator for that eager prefill has produced async illegal
        # memory access failures on the next scheduler allocation.
        self.pynccl_comm = self._new_pynccl_comm("eager")
        self.pynccl_graph_comm = self._new_pynccl_comm("graph")
        _register_group(self)
        logger.info(
            "[KUNSERVE-MS] KunServePyNcclGroup ready name=%s unique=%s "
            "rank=%d world=%d device=%s master=%s:%d "
            "pynccl_available=%s graph_pynccl_available=%s",
            self.name,
            self.unique_name,
            self.rank,
            self.world_size,
            self.device,
            host,
            int(port),
            bool(getattr(self.pynccl_comm, "available", False)),
            bool(getattr(self.pynccl_graph_comm, "available", False)),
        )

    def get_world_size(self) -> int:
        return self.world_size

    def get_rank(self) -> int:
        return self.rank

    def _new_pynccl_comm(self, kind: str) -> PyNcclCommunicator:
        comm = PyNcclCommunicator(
            group=self.stateless_group,
            device=self.device,
            use_current_stream=False,
        )
        self._diag_log(
            f"comm_ready kind={kind} name={self.name} unique={self.unique_name} "
            f"rank={self.rank}/{self.world_size} device={self.device} "
            f"available={bool(getattr(comm, 'available', False))}"
        )
        return comm

    @staticmethod
    def _is_cuda_graph_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _comm_kind(self) -> str:
        return "graph" if self._is_cuda_graph_capturing() else "eager"

    def _require_pynccl(self, *, graph: Optional[bool] = None) -> PyNcclCommunicator:
        use_graph = self._is_cuda_graph_capturing() if graph is None else bool(graph)
        comm = self.pynccl_graph_comm if use_graph else self.pynccl_comm
        if comm is None or not getattr(comm, "available", False):
            raise RuntimeError(
                f"KunServePyNcclGroup {self.name!r} has no available PyNccl "
                f"{'graph' if use_graph else 'eager'} communicator."
            )
        return comm

    @staticmethod
    def _capture_state_text() -> str:
        try:
            return "capturing" if torch.cuda.is_current_stream_capturing() else "eager"
        except Exception as exc:
            return f"capture_state_err={exc!r}"

    @staticmethod
    def _tensor_meta(tensor: Optional[torch.Tensor]) -> str:
        if not isinstance(tensor, torch.Tensor):
            return "None"
        try:
            ptr = hex(int(tensor.data_ptr()))
        except Exception as exc:
            ptr = f"<ptr_err:{exc!r}>"
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} ptr={ptr} contiguous={tensor.is_contiguous()}"
        )

    def _diag_log(self, message: str) -> None:
        if not self._diag_log_path:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(self._diag_log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"[{ts} pid={os.getpid()}] [KUNSERVE-DBG] pynccl {message}\n"
                )
        except Exception:
            pass

    def _diag_collective(
        self,
        op_name: str,
        input_: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        *,
        registered: bool,
    ) -> None:
        state = self._capture_state_text()
        key = (
            op_name,
            registered,
            state,
            tuple(input_.shape) if isinstance(input_, torch.Tensor) else None,
            str(input_.dtype) if isinstance(input_, torch.Tensor) else None,
            str(input_.device) if isinstance(input_, torch.Tensor) else None,
            tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            str(output.dtype) if isinstance(output, torch.Tensor) else None,
            str(output.device) if isinstance(output, torch.Tensor) else None,
        )
        count = int(self._diag_counts.get(op_name, 0)) + 1
        self._diag_counts[op_name] = count
        if count > 3 and key in self._diag_seen:
            return
        self._diag_seen.add(key)
        self._diag_log(
            f"collective op={op_name} registered={registered} count={count} "
            f"state={state} name={self.name} unique={self.unique_name} "
            f"comm_kind={self._comm_kind()} rank={self.rank}/{self.world_size} "
            f"device={self.device} "
            f"input={self._tensor_meta(input_)} output={self._tensor_meta(output)}"
        )

    def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
        self._diag_collective(
            "_all_reduce_in_place", input_, registered=False
        )
        if self.world_size == 1:
            return
        pynccl_comm = self._require_pynccl()
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            pynccl_comm.all_reduce(input_)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if not self._is_cuda_graph_capturing():
            self._all_reduce_in_place(input_)
            return input_
        self._diag_collective("all_reduce", input_, registered=True)
        inplace_all_reduce(input_, group_name=self.unique_name)
        return input_

    def _all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        self._diag_collective(
            "_all_gather_into_tensor", input, output, registered=False
        )
        if self.world_size == 1:
            output.copy_(input.reshape(output.shape))
            return
        pynccl_comm = self._require_pynccl()
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            pynccl_comm.all_gather(output, input)

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        if self.world_size == 1:
            output.copy_(input.reshape(output.shape))
            return
        if not self._is_cuda_graph_capturing():
            self._all_gather_into_tensor(output, input)
            return
        self._diag_collective(
            "all_gather_into_tensor", input, output, registered=True
        )
        reg_all_gather_into_tensor(output, input, group_name=self.unique_name)

    def grouped_all_gather_into_tensor(
        self,
        pairs: "list[tuple[torch.Tensor, torch.Tensor]]",
    ) -> None:
        """Phase G P1: batch multiple all-gather collectives into one ncclGroup.

        NCCL fuses same-communicator collectives within a
        ``ncclGroupStart/End`` bracket into a single kernel launch,
        saving ``len(pairs) - 1`` launch overheads per call site.  The
        graph-safe path bypasses the ``register_custom_op`` wrapper (which
        exists only for torch.compile visibility) and submits NCCL ops
        directly on the captured stream.

        ``pairs`` is a list of ``(output_tensor, input_tensor)`` tuples.
        Each pair is functionally equivalent to one
        ``all_gather_into_tensor`` call.
        """
        if not pairs:
            return
        if self.world_size == 1:
            for output, input_ in pairs:
                output.copy_(input_.reshape(output.shape))
            return
        self._diag_collective(
            "grouped_all_gather_into_tensor",
            pairs[0][1],
            pairs[0][0],
            registered=False,
        )
        pynccl_comm = self._require_pynccl()
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            pynccl_comm.group_start()
            for output, input_ in pairs:
                pynccl_comm.all_gather(output, input_)
            pynccl_comm.group_end()

    def _reduce_into_tensor(
        self,
        send: torch.Tensor,
        recv: torch.Tensor,
        root: int,
        op: ReduceOp = ReduceOp.SUM,
    ) -> None:
        """Phase B.2: ncclReduce on this group.

        All ranks call this; only ``root`` (the local lane position to land
        on) receives the sum into ``recv``.  Non-root ranks still pass a
        valid ``recv`` pointer per NCCL convention; their buffer is not
        written.  Used by chunked combine to land each replica's chunk on
        the lane rank that owns it, equivalent in semantics to a
        lane reduce-scatter split into per-chunk ncclReduce calls so the
        first chunk can run on the alt stream concurrently with the
        second chunk's expert compute.
        """
        if self.world_size == 1:
            # Single rank: just copy; ``root`` must be 0.
            if root != 0:
                raise ValueError(
                    f"_reduce_into_tensor on world_size=1 requires root=0; got {root}."
                )
            recv.copy_(send.reshape(recv.shape))
            return
        if int(root) < 0 or int(root) >= int(self.world_size):
            raise ValueError(
                f"root={root} out of range for world_size={self.world_size}."
            )
        if send.device != self.device:
            raise RuntimeError(
                f"_reduce_into_tensor send tensor on wrong device: "
                f"{send.device} vs comm device {self.device}."
            )
        if recv.device != self.device:
            raise RuntimeError(
                f"_reduce_into_tensor recv tensor on wrong device: "
                f"{recv.device} vs comm device {self.device}."
            )
        self._diag_collective(
            "_reduce_into_tensor", send, recv, registered=False
        )
        pynccl_comm = self._require_pynccl()
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            if pynccl_comm.disabled:
                return
            pynccl_comm.nccl.ncclReduce(
                buffer_type(send.data_ptr()),
                buffer_type(recv.data_ptr()),
                send.numel(),
                ncclDataTypeEnum.from_torch(send.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                int(root),
                pynccl_comm.comm,
                cudaStream_t(pynccl_comm.stream.cuda_stream),
            )

    def _reduce_scatter_tensor(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        self._diag_collective(
            "_reduce_scatter_tensor", input, output, registered=False
        )
        if self.world_size == 1:
            output.copy_(input.reshape(output.shape))
            return output
        pynccl_comm = self._require_pynccl()
        with pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
            ):
                pynccl_comm.reduce_scatter(output, input)
        return output

    def preheat_for_graph_capture(self) -> None:
        """Touch the graph communicator outside CUDA graph capture.

        Registered collectives call the private ``_all_*`` methods while the
        stream is capturing.  Without this explicit preheat, the graph
        communicator's first all_gather can happen inside capture even though
        the eager communicator was already warmed up.
        """
        if self.world_size == 1:
            return
        pynccl_comm = self._require_pynccl(graph=True)
        with torch.cuda.device(self.device), pynccl_comm.change_state(
            enable=True, stream=get_current_device_stream_fast()
        ):
            ag_in = torch.zeros(1, device=self.device)
            ag_out = torch.zeros(self.world_size, device=self.device)
            pynccl_comm.all_gather(ag_out, ag_in)
            ar_buf = torch.zeros(1, device=self.device)
            pynccl_comm.all_reduce(ar_buf)
            rs_in = torch.zeros(self.world_size, 1, device=self.device)
            rs_out = torch.zeros(1, device=self.device)
            pynccl_comm.reduce_scatter(rs_out, rs_in)
            # Phase G P1: preheat ncclGroupStart/End bracket pattern for
            # dispatch all-gather batching.  NCCL group calls have no new
            # collectives themselves but the fused-kernel codepath is a
            # different submission shape from the eager preheats above;
            # warming it here avoids any lazy init inside CUDA graph
            # capture if the static path uses grouped_all_gather_into_tensor.
            grp_in_a = torch.zeros(1, device=self.device)
            grp_out_a = torch.zeros(self.world_size, device=self.device)
            grp_in_b = torch.zeros(1, device=self.device)
            grp_out_b = torch.zeros(self.world_size, device=self.device)
            grp_in_c = torch.zeros(1, device=self.device)
            grp_out_c = torch.zeros(self.world_size, device=self.device)
            pynccl_comm.group_start()
            pynccl_comm.all_gather(grp_out_a, grp_in_a)
            pynccl_comm.all_gather(grp_out_b, grp_in_b)
            pynccl_comm.all_gather(grp_out_c, grp_in_c)
            pynccl_comm.group_end()
        # Phase B.2: ncclReduce preheat on default stream — even when the
        # alt-stream overlap is disabled, the chunked combine path may use
        # ncclReduce on the main stream once enabled, so warm the pattern
        # here so its first call inside CUDA graph capture doesn't trigger
        # lazy init.
        nccl_reduce_preheated = False
        try:
            red_send = torch.zeros(1, device=self.device)
            red_recv = torch.zeros(1, device=self.device)
            pynccl_comm.nccl.ncclReduce(
                buffer_type(red_send.data_ptr()),
                buffer_type(red_recv.data_ptr()),
                red_send.numel(),
                ncclDataTypeEnum.from_torch(red_send.dtype),
                ncclRedOpTypeEnum.from_torch(ReduceOp.SUM),
                0,  # root=0 — semantics don't matter for preheat
                pynccl_comm.comm,
                cudaStream_t(get_current_device_stream_fast().cuda_stream),
            )
            nccl_reduce_preheated = True
        except Exception as exc:
            logger.warning(
                "KunServe ncclReduce preheat failed on %s: %r",
                self.name,
                exc,
            )
        # Phase B.1: alt-stream NCCL preheat for combine overlap.
        # Issuing NCCL collectives on a fresh stream inside a CUDA graph
        # capture context can trigger lazy NCCL setup; warming the alt
        # stream here (outside any capture) avoids that hazard.  Gated by
        # env so that runs that don't enable the overlap don't pay the
        # extra preheat cost.
        alt_stream_preheated = False
        alt_stream_reduce_preheated = False
        if _alt_stream_combine_env_enabled():
            alt_stream = get_kunserve_combine_alt_stream(self.device)
            default_stream = get_current_device_stream_fast()
            alt_stream.wait_stream(default_stream)
            with torch.cuda.device(self.device), torch.cuda.stream(alt_stream):
                with pynccl_comm.change_state(enable=True, stream=alt_stream):
                    rs_in_alt = torch.zeros(
                        self.world_size, 1, device=self.device
                    )
                    rs_out_alt = torch.zeros(1, device=self.device)
                    pynccl_comm.reduce_scatter(rs_out_alt, rs_in_alt)
                    ar_buf_alt = torch.zeros(1, device=self.device)
                    pynccl_comm.all_reduce(ar_buf_alt)
                # Phase B.2: also preheat ncclReduce on alt stream — the
                # chunked combine path issues ncclReduce on alt stream
                # for the first chunk while expert kernel runs on main
                # stream for the second chunk.
                try:
                    red_send_alt = torch.zeros(1, device=self.device)
                    red_recv_alt = torch.zeros(1, device=self.device)
                    pynccl_comm.nccl.ncclReduce(
                        buffer_type(red_send_alt.data_ptr()),
                        buffer_type(red_recv_alt.data_ptr()),
                        red_send_alt.numel(),
                        ncclDataTypeEnum.from_torch(red_send_alt.dtype),
                        ncclRedOpTypeEnum.from_torch(ReduceOp.SUM),
                        0,
                        pynccl_comm.comm,
                        cudaStream_t(alt_stream.cuda_stream),
                    )
                    alt_stream_reduce_preheated = True
                except Exception as exc:
                    logger.warning(
                        "KunServe ncclReduce alt-stream preheat failed on %s: %r",
                        self.name,
                        exc,
                    )
            default_stream.wait_stream(alt_stream)
            alt_stream_preheated = True
        torch.cuda.synchronize()
        self._diag_log(
            f"graph_comm_preheated name={self.name} unique={self.unique_name} "
            f"rank={self.rank}/{self.world_size} device={self.device} "
            f"grouped_all_gather=True nccl_reduce_preheated={nccl_reduce_preheated} "
            f"alt_stream_preheated={alt_stream_preheated} "
            f"alt_stream_reduce_preheated={alt_stream_reduce_preheated}"
        )

    def reduce_scatter_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        if op != dist.ReduceOp.SUM:
            raise NotImplementedError(
                "KunServePyNcclGroup only supports SUM reduce_scatter_tensor."
            )
        if self.world_size == 1:
            output.copy_(input.reshape(output.shape))
            return output
        if not self._is_cuda_graph_capturing():
            return self._reduce_scatter_tensor(output, input)
        self._diag_collective(
            "reduce_scatter_tensor", input, output, registered=True
        )
        reg_reduce_scatter_tensor(output, input, group_name=self.unique_name)
        return output

    def barrier(self) -> None:
        self.stateless_group.barrier()

    def destroy(self) -> None:
        for kind, pynccl_comm in (
            ("eager", getattr(self, "pynccl_comm", None)),
            ("graph", getattr(self, "pynccl_graph_comm", None)),
        ):
            if pynccl_comm is None or not getattr(pynccl_comm, "available", False):
                continue
            try:
                pynccl_comm.nccl.ncclCommDestroy(pynccl_comm.comm)
            except Exception:
                logger.exception(
                    "Failed to destroy KunServe PyNccl %s communicator %s",
                    kind,
                    self.name,
                )
        try:
            import sglang.srt.distributed.parallel_state as parallel_state

            ref = parallel_state._groups.get(self.unique_name)
            if isinstance(ref, weakref.ReferenceType) and ref() is self:
                parallel_state._groups.pop(self.unique_name, None)
        except Exception:
            logger.exception("Failed to unregister KunServe group %s", self.name)
