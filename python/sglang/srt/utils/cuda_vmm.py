from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger(__name__)
_CUDA_VMM_AVAILABLE: Optional[bool] = None

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CU(call) do {                                              \
    CUresult _r = (call);                                                \
    if (_r != CUDA_SUCCESS) {                                            \
        const char* s = nullptr; cuGetErrorString(_r, &s);               \
        throw std::runtime_error(std::string(#call " failed: ")          \
                                  + (s ? s : "unknown"));                \
    }                                                                    \
} while(0)

static int current_device() {
    int dev = 0;
    CHECK_CU(cuCtxGetDevice(&dev));
    return dev;
}

// API: reserve virtual memory of given size, returning the base virtual address as int64
static int64_t vmm_reserve(int64_t size) {
    CUdeviceptr ptr = 0;
    CHECK_CU(cuMemAddressReserve(&ptr, (size_t)size, 0, 0, 0));
    return (int64_t)ptr;
}
// API: create a physical allocation handle of given size, returning the handle as int64
static int64_t vmm_create_phys(int64_t size) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = current_device();
    CUmemGenericAllocationHandle h = 0;
    CHECK_CU(cuMemCreate(&h, (size_t)size, &prop, 0));
    return (int64_t)h;
}
// API: map a physical handle into a reserved virtual address range with given offsets and size,
// then allow this device to access the mapping with read/write permissions.
static void vmm_map_with_offset(
    int64_t va,
    int64_t va_offset,
    int64_t handle,
    int64_t handle_offset,
    int64_t size
) {
    CUdeviceptr at = ((CUdeviceptr)va) + (size_t)va_offset;
    CHECK_CU(cuMemMap(
        at,
        (size_t)size,
        0,
        (CUmemGenericAllocationHandle)handle,
        (size_t)handle_offset));
    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = current_device();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(at, (size_t)size, &desc, 1));
}

static void vmm_unmap(int64_t va, int64_t offset, int64_t size) {
    CHECK_CU(cuMemUnmap(((CUdeviceptr)va) + (size_t)offset, (size_t)size));
}

static void vmm_release_handle(int64_t handle) {
    CHECK_CU(cuMemRelease((CUmemGenericAllocationHandle)handle));
}

static void vmm_address_free(int64_t va, int64_t size) {
    CHECK_CU(cuMemAddressFree((CUdeviceptr)va, (size_t)size));
}

static int64_t vmm_granularity() {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = current_device();
    size_t g = 0;
    CHECK_CU(cuMemGetAllocationGranularity(
        &g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return (int64_t)g;
}

static torch::ScalarType parse_dtype(const std::string& dtype_name) {
    if (dtype_name == "float16") {
        return torch::kFloat16;
    }
    if (dtype_name == "bfloat16") {
        return torch::kBFloat16;
    }
    if (dtype_name == "float32") {
        return torch::kFloat32;
    }
    if (dtype_name == "uint8") {
        return torch::kUInt8;
    }
    if (dtype_name == "int32") {
        return torch::kInt32;
    }
    if (dtype_name == "int64") {
        return torch::kInt64;
    }
    // FP8 variants used by KunServe + DeepGEMM. All are 1-byte and stored as
    // raw bytes in the VMM region; we just need from_blob to interpret the
    // memory with the correct ScalarType so downstream FP8 grouped GEMM
    // kernels accept it.
    if (dtype_name == "float8_e4m3fn") {
        return torch::kFloat8_e4m3fn;
    }
    if (dtype_name == "float8_e5m2") {
        return torch::kFloat8_e5m2;
    }
    if (dtype_name == "float8_e4m3fnuz") {
        return torch::kFloat8_e4m3fnuz;
    }
    if (dtype_name == "float8_e5m2fnuz") {
        return torch::kFloat8_e5m2fnuz;
    }
    throw std::runtime_error("Unsupported wrap dtype: " + dtype_name);
}

static torch::Tensor wrap_tensor(
    int64_t ptr,
    std::vector<int64_t> shape,
    const std::string& dtype_name
) {
    auto options = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, current_device()))
                       .dtype(parse_dtype(dtype_name));
    return torch::from_blob((void*)ptr, shape, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vmm_reserve", &vmm_reserve);
    m.def("vmm_create_phys", &vmm_create_phys);
    m.def("vmm_map_with_offset", &vmm_map_with_offset);
    m.def("vmm_unmap", &vmm_unmap);
    m.def("vmm_release_handle", &vmm_release_handle);
    m.def("vmm_address_free", &vmm_address_free);
    m.def("vmm_granularity", &vmm_granularity);
    m.def("wrap_tensor", &wrap_tensor);
}
"""

_WRAP_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.uint8: "uint8",
    torch.int32: "int32",
    torch.int64: "int64",
}
# FP8 dtypes are present on all torch builds we support (>= 2.2). Probe with
# getattr so this module still imports cleanly on older torch where one of the
# names is missing — the entry just isn't added in that case and the original
# "Unsupported CUDA VMM dtype" error path is preserved.
for _fp8_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _fp8_dtype = getattr(torch, _fp8_name, None)
    if _fp8_dtype is not None:
        _WRAP_DTYPE_NAMES[_fp8_dtype] = _fp8_name
del _fp8_name, _fp8_dtype


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in _WRAP_DTYPE_NAMES:
        raise ValueError(f"Unsupported CUDA VMM dtype: {dtype}")
    return _WRAP_DTYPE_NAMES[dtype]


def _prod(values: Sequence[int]) -> int:
    ans = 1
    for value in values:
        ans *= int(value)
    return ans


def _ensure_cuda_device_context() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VMM requires torch.cuda.is_available()")
    default_device = torch.device(torch.get_default_device())
    if default_device.type != "cuda":
        default_device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(0, device=default_device)


@lru_cache(maxsize=1)
def _get_ext():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VMM is only available when torch.cuda.is_available()")
    logger.info("Building CUDA VMM inline extension.")
    return load_inline(
        name="sglang_cuda_vmm_ext_v1",
        cpp_sources=_CPP_SRC,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def cuda_vmm_available() -> bool:
    global _CUDA_VMM_AVAILABLE
    if _CUDA_VMM_AVAILABLE is not None:
        return _CUDA_VMM_AVAILABLE
    try:
        _get_ext()
        _CUDA_VMM_AVAILABLE = True
        return True
    except Exception:
        logger.exception("Failed to initialize CUDA VMM support.")
        _CUDA_VMM_AVAILABLE = False
        return False


def get_granularity() -> int:
    _ensure_cuda_device_context()
    return int(_get_ext().vmm_granularity())


def round_up_to_granularity(value: int, granularity: Optional[int] = None) -> int:
    g = granularity or get_granularity()
    return ((int(value) + g - 1) // g) * g


def ensure_granularity_aligned(value: int, granularity: Optional[int] = None) -> None:
    g = granularity or get_granularity()
    if int(value) % g != 0:
        raise ValueError(f"value={value} is not aligned to granularity={g}")


def min_granularity_aligned_row_count(
    row_bytes: int, granularity: Optional[int] = None
) -> int:
    row_bytes = int(row_bytes)
    if row_bytes <= 0:
        raise ValueError(f"row_bytes must be positive, got {row_bytes}")
    g = int(granularity or get_granularity())
    if g <= 0:
        raise ValueError(f"granularity must be positive, got {g}")
    return g // math.gcd(row_bytes, g)


@dataclass
class VmmPhysicalHandle:
    handle: int
    size_bytes: int
    label: str = ""
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        _get_ext().vmm_release_handle(self.handle)
        self._released = True


@dataclass
class RegionMapping:
    va_offset_bytes: int
    size_bytes: int
    physical: VmmPhysicalHandle
    handle_offset_bytes: int = 0
    label: str = ""
    borrowed: bool = False
    source_segment: Optional["DonorSegment"] = None

    @property
    def end_offset_bytes(self) -> int:
        return self.va_offset_bytes + self.size_bytes


@dataclass
class DonorSegment:
    physical: VmmPhysicalHandle
    source_region: "VmmRegion"
    source_va_offset_bytes: int
    size_bytes: int
    handle_offset_bytes: int = 0
    label: str = ""
    owner: dict[str, object] = field(default_factory=dict)

    def split_prefix(
        self, size_bytes: int
    ) -> tuple["DonorSegment", Optional["DonorSegment"]]:
        if size_bytes <= 0 or size_bytes > self.size_bytes:
            raise ValueError(
                f"Cannot split donor segment of size {self.size_bytes} with prefix {size_bytes}"
            )
        prefix = DonorSegment(
            physical=self.physical,
            source_region=self.source_region,
            source_va_offset_bytes=self.source_va_offset_bytes,
            size_bytes=size_bytes,
            handle_offset_bytes=self.handle_offset_bytes,
            label=self.label,
            owner=dict(self.owner),
        )
        suffix_size = self.size_bytes - size_bytes
        suffix = None
        if suffix_size > 0:
            suffix = DonorSegment(
                physical=self.physical,
                source_region=self.source_region,
                source_va_offset_bytes=self.source_va_offset_bytes + size_bytes,
                size_bytes=suffix_size,
                handle_offset_bytes=self.handle_offset_bytes + size_bytes,
                label=self.label,
                owner=dict(self.owner),
            )
        return prefix, suffix

    def restore_to_source(self) -> RegionMapping:
        return self.source_region.map_existing(
            self.physical,
            va_offset_bytes=self.source_va_offset_bytes,
            size_bytes=self.size_bytes,
            handle_offset_bytes=self.handle_offset_bytes,
            label=self.label,
            borrowed=False,
        )


class DonorLedger:
    def __init__(self, label: str = ""):
        self.label = label
        self._segments: list[DonorSegment] = []

    def add(self, segment: DonorSegment) -> None:
        self._segments.append(segment)

    def extend(self, segments: Iterable[DonorSegment]) -> None:
        for segment in segments:
            self.add(segment)

    def pop_all(self) -> list[DonorSegment]:
        segments = list(self._segments)
        self._segments.clear()
        return segments

    def restore_all(self) -> None:
        for segment in reversed(self.pop_all()):
            segment.restore_to_source()

    @property
    def total_bytes(self) -> int:
        return sum(segment.size_bytes for segment in self._segments)

    def __len__(self) -> int:
        return len(self._segments)


class VmmRegion:
    def __init__(self, reserve_size_bytes: int, label: str = ""):
        self.label = label
        self.granularity = get_granularity()
        self.reserve_size_bytes = round_up_to_granularity(
            reserve_size_bytes, self.granularity
        )
        self.va = int(_get_ext().vmm_reserve(self.reserve_size_bytes))
        self._mappings: list[RegionMapping] = []
        self._owned_handles: list[VmmPhysicalHandle] = []
        self._released = False

    def create_physical(self, size_bytes: int, label: str = "") -> VmmPhysicalHandle:
        size_bytes = round_up_to_granularity(size_bytes, self.granularity)
        physical = VmmPhysicalHandle(
            handle=int(_get_ext().vmm_create_phys(size_bytes)),
            size_bytes=size_bytes,
            label=label or self.label,
        )
        self._owned_handles.append(physical)
        return physical

    def map_existing(
        self,
        physical: VmmPhysicalHandle,
        *,
        va_offset_bytes: int,
        size_bytes: int,
        handle_offset_bytes: int = 0,
        label: str = "",
        borrowed: bool = False,
        source_segment: Optional[DonorSegment] = None,
    ) -> RegionMapping:
        ensure_granularity_aligned(va_offset_bytes, self.granularity)
        ensure_granularity_aligned(size_bytes, self.granularity)
        ensure_granularity_aligned(handle_offset_bytes, self.granularity)
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")
        if va_offset_bytes + size_bytes > self.reserve_size_bytes:
            raise ValueError(
                f"Mapping [{va_offset_bytes}, {va_offset_bytes + size_bytes}) exceeds reserve_size_bytes={self.reserve_size_bytes}"
            )
        if handle_offset_bytes + size_bytes > physical.size_bytes:
            raise ValueError(
                f"Mapping [{handle_offset_bytes}, {handle_offset_bytes + size_bytes}) exceeds physical.size_bytes={physical.size_bytes}"
            )
        _get_ext().vmm_map_with_offset(
            self.va,
            va_offset_bytes,
            physical.handle,
            handle_offset_bytes,
            size_bytes,
        )
        mapping = RegionMapping(
            va_offset_bytes=va_offset_bytes,
            size_bytes=size_bytes,
            physical=physical,
            handle_offset_bytes=handle_offset_bytes,
            label=label or physical.label or self.label,
            borrowed=borrowed,
            source_segment=source_segment,
        )
        self._mappings.append(mapping)
        self._mappings.sort(key=lambda item: item.va_offset_bytes)
        return mapping

    def map_new(
        self,
        *,
        va_offset_bytes: int,
        size_bytes: int,
        label: str = "",
    ) -> RegionMapping:
        physical = self.create_physical(size_bytes, label=label)
        return self.map_existing(
            physical,
            va_offset_bytes=va_offset_bytes,
            size_bytes=physical.size_bytes,
            handle_offset_bytes=0,
            label=label,
            borrowed=False,
        )

    def unmap(self, *, va_offset_bytes: int, size_bytes: int) -> None:
        ensure_granularity_aligned(va_offset_bytes, self.granularity)
        ensure_granularity_aligned(size_bytes, self.granularity)
        _get_ext().vmm_unmap(self.va, va_offset_bytes, size_bytes)
        self._remove_range_bookkeeping(va_offset_bytes, size_bytes)

    def borrow_tail(self, size_bytes: int, *, owner: Optional[dict[str, object]] = None, label: str = "") -> DonorSegment:
        size_bytes = round_up_to_granularity(size_bytes, self.granularity)
        if not self._mappings:
            raise RuntimeError("Cannot borrow tail from an empty VMM region")
        tail = self._mappings[-1]
        if tail.end_offset_bytes != self.mapped_end_offset_bytes:
            raise RuntimeError("Tail borrow currently requires a contiguous mapped suffix")
        if tail.size_bytes < size_bytes:
            raise RuntimeError(
                f"Requested donor bytes {size_bytes} exceed tail mapping size {tail.size_bytes}"
            )
        donor_offset = tail.end_offset_bytes - size_bytes
        donor_handle_offset = tail.handle_offset_bytes + tail.size_bytes - size_bytes
        _get_ext().vmm_unmap(self.va, donor_offset, size_bytes)
        self._remove_range_bookkeeping(donor_offset, size_bytes)
        source_region = self
        source_va_offset_bytes = donor_offset
        donor_owner = dict(owner or {})
        if tail.source_segment is not None:
            source_delta = donor_handle_offset - tail.source_segment.handle_offset_bytes
            source_region = tail.source_segment.source_region
            source_va_offset_bytes = (
                tail.source_segment.source_va_offset_bytes + source_delta
            )
            donor_owner = dict(tail.source_segment.owner)
            donor_owner.update(owner or {})
        return DonorSegment(
            physical=tail.physical,
            source_region=source_region,
            source_va_offset_bytes=source_va_offset_bytes,
            size_bytes=size_bytes,
            handle_offset_bytes=donor_handle_offset,
            label=label or tail.label or self.label,
            owner=donor_owner,
        )

    def borrow_head(
        self,
        size_bytes: int,
        *,
        owner: Optional[dict[str, object]] = None,
        label: str = "",
    ) -> DonorSegment:
        size_bytes = round_up_to_granularity(size_bytes, self.granularity)
        if not self._mappings:
            raise RuntimeError("Cannot borrow head from an empty VMM region")
        head = self._mappings[0]
        mapped_start = min(mapping.va_offset_bytes for mapping in self._mappings)
        if head.va_offset_bytes != mapped_start:
            raise RuntimeError("Head borrow currently requires a contiguous mapped prefix")
        if head.size_bytes < size_bytes:
            raise RuntimeError(
                f"Requested donor bytes {size_bytes} exceed head mapping size {head.size_bytes}"
            )
        donor_offset = head.va_offset_bytes
        donor_handle_offset = head.handle_offset_bytes
        _get_ext().vmm_unmap(self.va, donor_offset, size_bytes)
        self._remove_range_bookkeeping(donor_offset, size_bytes)
        source_region = self
        source_va_offset_bytes = donor_offset
        donor_owner = dict(owner or {})
        if head.source_segment is not None:
            source_delta = donor_handle_offset - head.source_segment.handle_offset_bytes
            source_region = head.source_segment.source_region
            source_va_offset_bytes = (
                head.source_segment.source_va_offset_bytes + source_delta
            )
            donor_owner = dict(head.source_segment.owner)
            donor_owner.update(owner or {})
        return DonorSegment(
            physical=head.physical,
            source_region=source_region,
            source_va_offset_bytes=source_va_offset_bytes,
            size_bytes=size_bytes,
            handle_offset_bytes=donor_handle_offset,
            label=label or head.label or self.label,
            owner=donor_owner,
        )

    def wrap_tensor(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return _get_ext().wrap_tensor(self.va, list(map(int, shape)), _dtype_name(dtype))

    @property
    def mappings(self) -> tuple[RegionMapping, ...]:
        return tuple(self._mappings)

    @property
    def mapped_bytes(self) -> int:
        return sum(mapping.size_bytes for mapping in self._mappings)

    @property
    def mapped_end_offset_bytes(self) -> int:
        if not self._mappings:
            return 0
        return max(mapping.end_offset_bytes for mapping in self._mappings)

    def close(self) -> None:
        if self._released:
            return
        for mapping in list(self._mappings):
            try:
                _get_ext().vmm_unmap(self.va, mapping.va_offset_bytes, mapping.size_bytes)
            except Exception:
                logger.exception("Failed to unmap VMM range during close.")
        self._mappings.clear()
        for handle in reversed(self._owned_handles):
            try:
                handle.release()
            except Exception:
                logger.exception("Failed to release CUDA VMM handle during close.")
        self._owned_handles.clear()
        try:
            _get_ext().vmm_address_free(self.va, self.reserve_size_bytes)
        except Exception:
            logger.exception("Failed to free CUDA VMM address range during close.")
        self._released = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _remove_range_bookkeeping(self, va_offset_bytes: int, size_bytes: int) -> None:
        end_offset = va_offset_bytes + size_bytes
        new_mappings: list[RegionMapping] = []
        removed = False
        for mapping in self._mappings:
            if mapping.end_offset_bytes <= va_offset_bytes or mapping.va_offset_bytes >= end_offset:
                new_mappings.append(mapping)
                continue
            if not (
                mapping.va_offset_bytes <= va_offset_bytes
                and mapping.end_offset_bytes >= end_offset
            ):
                raise RuntimeError(
                    f"Cannot partially remove range [{va_offset_bytes}, {end_offset}) from non-covering mapping {mapping}"
                )
            removed = True
            prefix_size = va_offset_bytes - mapping.va_offset_bytes
            prefix_source_segment = None
            suffix_source_segment = None
            if mapping.source_segment is not None:
                if prefix_size > 0:
                    prefix_source_segment, remainder_source_segment = (
                        mapping.source_segment.split_prefix(prefix_size)
                    )
                else:
                    remainder_source_segment = mapping.source_segment
                suffix_size = mapping.end_offset_bytes - end_offset
                if suffix_size > 0:
                    if remainder_source_segment.size_bytes == suffix_size:
                        suffix_source_segment = remainder_source_segment
                    else:
                        _, suffix_source_segment = remainder_source_segment.split_prefix(
                            remainder_source_segment.size_bytes - suffix_size
                        )
            if prefix_size > 0:
                new_mappings.append(
                    RegionMapping(
                        va_offset_bytes=mapping.va_offset_bytes,
                        size_bytes=prefix_size,
                        physical=mapping.physical,
                        handle_offset_bytes=mapping.handle_offset_bytes,
                        label=mapping.label,
                        borrowed=mapping.borrowed,
                        source_segment=prefix_source_segment,
                    )
                )
            suffix_size = mapping.end_offset_bytes - end_offset
            if suffix_size > 0:
                new_mappings.append(
                    RegionMapping(
                        va_offset_bytes=end_offset,
                        size_bytes=suffix_size,
                        physical=mapping.physical,
                        handle_offset_bytes=(
                            mapping.handle_offset_bytes
                            + (end_offset - mapping.va_offset_bytes)
                        ),
                        label=mapping.label,
                        borrowed=mapping.borrowed,
                        source_segment=suffix_source_segment,
                    )
                )
        if not removed:
            raise RuntimeError(
                f"Cannot find mapped VMM range covering [{va_offset_bytes}, {end_offset})"
            )
        self._mappings = sorted(new_mappings, key=lambda item: item.va_offset_bytes)


class ExpandableVmmTensor:
    def __init__(
        self,
        *,
        reserve_shape: Sequence[int],
        dtype: torch.dtype,
        active_rows: int,
        label: str,
        wrap_full_tensor: bool = False,
    ):
        if len(reserve_shape) < 1:
            raise ValueError("reserve_shape must have at least one dimension")
        self.reserve_shape = tuple(int(x) for x in reserve_shape)
        self.dtype = dtype
        self.label = label
        self.wrap_full_tensor = wrap_full_tensor
        self.element_size = torch.empty((), dtype=dtype).element_size()
        self.row_shape = self.reserve_shape[1:]
        self.row_numel = _prod(self.row_shape)
        self.row_bytes = self.row_numel * self.element_size
        self.reserve_rows = self.reserve_shape[0]
        self.region = VmmRegion(
            reserve_size_bytes=self.reserve_rows * self.row_bytes,
            label=label,
        )
        self._tensor: Optional[torch.Tensor] = None
        self._active_rows = 0
        self._mapped_start_bytes = 0
        self._mapped_bytes = 0
        self._active_view: Optional[torch.Tensor] = None
        self.ensure_active_rows(active_rows)

    @property
    def tensor(self) -> torch.Tensor:
        assert self._tensor is not None
        return self._tensor

    @property
    def active_rows(self) -> int:
        return self._active_rows

    @property
    def active_view(self) -> torch.Tensor:
        assert self._active_view is not None
        return self._active_view

    @property
    def mapped_prefix_bytes(self) -> int:
        return self._mapped_bytes

    @property
    def mapped_start_row(self) -> int:
        if self.row_bytes == 0:
            return 0
        return self._mapped_start_bytes // self.row_bytes

    def ensure_active_rows(
        self,
        target_rows: int,
        *,
        donor_segments: Optional[Iterable[DonorSegment]] = None,
        allow_donor_split: bool = True,
    ) -> list[DonorSegment]:
        target_rows = int(target_rows)
        if target_rows < 0 or target_rows > self.reserve_rows:
            raise ValueError(
                f"target_rows={target_rows} is outside [0, {self.reserve_rows}]"
            )
        target_prefix_bytes = round_up_to_granularity(
            target_rows * self.row_bytes, self.region.granularity
        )
        returned_segments: list[DonorSegment] = []
        if self._mapped_start_bytes != 0:
            raise RuntimeError(
                "ensure_active_rows only supports prefix-managed mappings."
            )
        aligned_row_chunk_bytes: Optional[int] = None
        if self.row_bytes % self.region.granularity != 0:
            aligned_row_chunk_bytes = (
                min_granularity_aligned_row_count(
                    self.row_bytes, self.region.granularity
                )
                * self.row_bytes
            )
        if target_prefix_bytes > self._mapped_bytes:
            cursor = self._mapped_bytes
            remaining = target_prefix_bytes - self._mapped_bytes
            if donor_segments is None:
                donors: list[DonorSegment] = []
            elif isinstance(donor_segments, list):
                donors = donor_segments
            else:
                donors = list(donor_segments)
            while remaining > 0:
                if donors:
                    donor = donors.pop(0)
                    if donor.size_bytes > remaining:
                        if not allow_donor_split:
                            raise RuntimeError(
                                "Donor-backed VMM expansion requires whole donor segments; "
                                f"needed {remaining} bytes but next donor segment is "
                                f"{donor.size_bytes} bytes."
                            )
                        donor, remainder = donor.split_prefix(remaining)
                        donors.insert(0, remainder)
                    self.region.map_existing(
                        donor.physical,
                        va_offset_bytes=cursor,
                        size_bytes=donor.size_bytes,
                        handle_offset_bytes=donor.handle_offset_bytes,
                        label=donor.label,
                        borrowed=True,
                        source_segment=donor,
                    )
                    cursor += donor.size_bytes
                    remaining -= donor.size_bytes
                else:
                    if aligned_row_chunk_bytes is None:
                        mapping_size_bytes = self.row_bytes
                    else:
                        # Keep sub-granularity rows grouped into borrowable chunks so
                        # later donor borrows can unmap whole mappings instead of
                        # carving a suffix out of one giant region.
                        mapping_size_bytes = min(remaining, aligned_row_chunk_bytes)
                    mapping = self.region.map_new(
                        va_offset_bytes=cursor,
                        size_bytes=mapping_size_bytes,
                        label=self.label,
                    )
                    cursor += mapping.size_bytes
                    remaining -= mapping.size_bytes
            self._mapped_bytes = target_prefix_bytes
        elif target_prefix_bytes < self._mapped_bytes:
            tail_to_return = self.borrow_tail_bytes(
                self._mapped_bytes - target_prefix_bytes,
                owner={"label": self.label, "kind": "kv_tail"},
            )
            returned_segments.append(tail_to_return)
            self._mapped_bytes = target_prefix_bytes
        self._active_rows = target_rows
        self._refresh_wrapped_tensor()
        return returned_segments

    def borrow_tail_bytes(
        self, size_bytes: int, *, owner: Optional[dict[str, object]] = None
    ) -> DonorSegment:
        size_bytes = round_up_to_granularity(size_bytes, self.region.granularity)
        donor = self.region.borrow_tail(
            size_bytes,
            owner=owner,
            label=self.label,
        )
        self._mapped_bytes -= donor.size_bytes
        self._active_rows = min(self._active_rows, self._mapped_bytes // self.row_bytes)
        self._refresh_wrapped_tensor()
        return donor

    def borrow_head_bytes(
        self, size_bytes: int, *, owner: Optional[dict[str, object]] = None
    ) -> DonorSegment:
        size_bytes = round_up_to_granularity(size_bytes, self.region.granularity)
        donor = self.region.borrow_head(
            size_bytes,
            owner=owner,
            label=self.label,
        )
        self._mapped_start_bytes += donor.size_bytes
        self._mapped_bytes -= donor.size_bytes
        self._active_rows = min(self._active_rows, self._mapped_bytes // self.row_bytes)
        self._refresh_wrapped_tensor()
        return donor

    def borrow_tail_rows(
        self, num_rows: int, *, owner: Optional[dict[str, object]] = None
    ) -> DonorSegment:
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")
        size_bytes = num_rows * self.row_bytes
        ensure_granularity_aligned(size_bytes, self.region.granularity)
        return self.borrow_tail_bytes(size_bytes, owner=owner)

    def borrow_head_rows(
        self, num_rows: int, *, owner: Optional[dict[str, object]] = None
    ) -> DonorSegment:
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")
        size_bytes = num_rows * self.row_bytes
        ensure_granularity_aligned(size_bytes, self.region.granularity)
        return self.borrow_head_bytes(size_bytes, owner=owner)

    def mapped_logical_bytes(self) -> int:
        return self._mapped_bytes

    def sync_from_region(self) -> None:
        mappings = self.region.mappings
        if not mappings:
            self._mapped_start_bytes = 0
            self._mapped_bytes = 0
        else:
            self._mapped_start_bytes = min(mapping.va_offset_bytes for mapping in mappings)
            self._mapped_bytes = sum(mapping.size_bytes for mapping in mappings)
        self._active_rows = min(self.reserve_rows, self._mapped_bytes // self.row_bytes)
        self._refresh_wrapped_tensor()

    def _refresh_wrapped_tensor(self) -> None:
        if self.wrap_full_tensor:
            if self._tensor is None:
                self._tensor = self.region.wrap_tensor(self.reserve_shape, self.dtype)
            self._active_view = self._tensor.narrow(
                0, self.mapped_start_row, self._active_rows
            )
            return
        if self._mapped_start_bytes != 0:
            raise RuntimeError(
                "Non-full VMM tensors do not support non-zero mapped_start_bytes."
            )
        shape = (self._active_rows, *self.row_shape)
        if self._active_rows == 0:
            self._tensor = torch.empty(shape, dtype=self.dtype, device="cuda")
        else:
            self._tensor = self.region.wrap_tensor(shape, self.dtype)
        self._active_view = self._tensor


class MoeWeightDonorManager:
    def __init__(
        self,
        *,
        layer_id: int,
        allocations: dict[str, ExpandableVmmTensor],
    ):
        self.layer_id = layer_id
        self.allocations = allocations
        self.borrowed_segments = DonorLedger(label=f"layer_{layer_id}")

    def total_bytes(self) -> int:
        return sum(
            allocation.mapped_logical_bytes() for allocation in self.allocations.values()
        )

    def bytes_per_expert(self) -> int:
        return sum(allocation.row_bytes for allocation in self.allocations.values())

    def max_borrowable_experts(self) -> int:
        if not self.allocations:
            return 0
        return min(allocation.active_rows for allocation in self.allocations.values())

    def borrow_tail_experts(
        self,
        num_experts: int,
        preferred_order: Sequence[str] = ("w13_weight", "w2_weight"),
    ) -> DonorLedger:
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        ledger = DonorLedger(label=f"layer_{self.layer_id}_experts")
        try:
            for name in preferred_order:
                allocation = self.allocations.get(name)
                if allocation is None:
                    continue
                rows_per_chunk = min_granularity_aligned_row_count(
                    allocation.row_bytes, allocation.region.granularity
                )
                if num_experts % rows_per_chunk != 0:
                    raise ValueError(
                        f"Cannot borrow {num_experts} experts from allocation '{name}' "
                        f"on layer {self.layer_id}; row_bytes={allocation.row_bytes} "
                        f"requires chunks of {rows_per_chunk} rows for "
                        f"granularity={allocation.region.granularity}."
                    )
                for expert_offset in range(0, num_experts, rows_per_chunk):
                    ledger.add(
                        allocation.borrow_tail_rows(
                            rows_per_chunk,
                            owner={
                                "layer_id": self.layer_id,
                                "param_name": name,
                                "num_experts": num_experts,
                                "expert_offset": expert_offset,
                                "chunk_rows": rows_per_chunk,
                            },
                        )
                    )
        except Exception:
            ledger.restore_all()
            raise
        new_segments = ledger.pop_all()
        self.borrowed_segments.extend(new_segments)
        result = DonorLedger(label=ledger.label)
        result.extend(new_segments)
        return result

    def borrow_tail_bytes(
        self, num_bytes: int, preferred_order: Sequence[str] = ("w13_weight", "w2_weight")
    ) -> DonorLedger:
        remaining = round_up_to_granularity(num_bytes)
        ledger = DonorLedger(label=f"layer_{self.layer_id}")
        for name in preferred_order:
            if remaining <= 0:
                break
            allocation = self.allocations.get(name)
            if allocation is None:
                continue
            available = allocation.mapped_prefix_bytes
            if available <= 0:
                continue
            take = min(available, remaining)
            take = round_up_to_granularity(take, allocation.region.granularity)
            if take <= 0:
                continue
            segment = allocation.borrow_tail_bytes(
                take,
                owner={"layer_id": self.layer_id, "param_name": name},
            )
            ledger.add(segment)
            remaining -= segment.size_bytes
        if remaining > 0:
            ledger.restore_all()
            raise RuntimeError(
                f"Unable to borrow {num_bytes} bytes from MoE weight donors on layer {self.layer_id}"
            )
        new_segments = ledger.pop_all()
        self.borrowed_segments.extend(new_segments)
        result = DonorLedger(label=ledger.label)
        result.extend(new_segments)
        return result

    def restore_all(self) -> None:
        self.borrowed_segments.restore_all()
        for allocation in self.allocations.values():
            allocation.sync_from_region()
