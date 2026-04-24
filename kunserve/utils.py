import numpy as np
import psutil
import random
import subprocess as sp
import torch
import uuid
import os
from typing import TypeAlias, List
from enum import Enum

GB = 1 << 30
MB = 1 << 20

SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory

def get_gpu_memory_usage(gpu: int = 0):
    """
    Python equivalent of nvidia-smi, copied from https://stackoverflow.com/a/67722676
    and verified as being equivalent ✅
    """
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]

    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    return int(memory_use_info[gpu].split()[0])


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


cudaMemoryIpcHandle: TypeAlias = List[int]


class Stage(Enum):
    """The stage of a SingleStageLLMEngine"""

    CONTEXT = "context"
    DECODING = "decoding"
    COLOCATED = "colocated"

    def __str__(self) -> str:
        return self.value
    

class EngineState(Enum):
    INITIAL = "INITIAL"
    BALLOON = "BALLOON"
    PREPARE = "PREPARE"
    RESTORE = "RESTORE"
    STOP = "STOP"

    def __str__(self) -> str:
        return self.value
    
def get_instance_name(server_id: int, instance_id: int) -> str:
    return f"{INSTANCE_NAME_PREFIX}{server_id}_{instance_id}"

def get_log_name(log_path, qps, cv, dist, scale) -> str:
    return os.path.join(log_path, f"{qps=}_{cv=}_{dist=}_{scale=}")