from dataclasses import dataclass
import dataclasses

import argparse
import yaml
from typing import Tuple


@dataclass
class EngineConfig:
    nservers: int = None
    ngroups: int = None
    group_size: int = None
    pp: int = None
    tp: int = None
    block_size: int = None
    use_tensor_cores: bool = None

    schedule_policy: str = None
    preempt_method: str = None
    max_batch_size: int = None
    max_batch_tokens: int = None
    gpu_util: float = None

    dtype: str = None
    enable_chunked_prefill: bool = None
    chunked_prefill_size: int = None
    chunked_prefill_budget: int = None


    @classmethod
    def from_cli_args(cls, cli_args) -> "EngineConfig":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(cli_args, attr) for attr in attrs})
        return engine_args

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--nservers", type=int, default=1)
        parser.add_argument(
            "--group-size", 
            type=int, 
            default=1,
            help="The number of instances in a group for parameter sharing"
        )
        parser.add_argument(
            "--ngroups",
            type=int,
            default=1,
            help="The number of instance groups to launch.")
        parser.add_argument("--pp", type=int, default=1)
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--block-size", type=int, default=256)
        parser.add_argument(
            "--use-tensor-cores", action="store_true", default=False)
        parser.add_argument("--schedule-policy", type=str, default="prefill-first")
        parser.add_argument("--preempt-method", type=str, default="recompute")
        parser.add_argument("--max-batch-size", type=int, default=256)
        parser.add_argument("--max-batch-tokens", type=int, default=131072)
        parser.add_argument("--gpu-util", type=float, default=0.9)
        parser.add_argument("--dtype", type=str, default="bf16")
        parser.add_argument(
            "--enable-chunked-prefill", action="store_true", default=False)
        parser.add_argument("--chunked-prefill-size", type=int, default=None)
        parser.add_argument("--chunked-prefill-budget", type=int, default=4096)


@dataclass
class BalloonConfig:
    enable_balloon: bool = None
    balloon_mode: str = None

    # the blocks that are exchanged together
    balloon_ratio: float = None
    exchange_batch_size: int = None
    restore_batch_size: int = None
    max_balloon_memory_ratio: float = None
    max_restore_memory_ratio: float = None
    restore_memory_threshold: float = None

    @classmethod
    def from_cli_args(cls, cli_args) -> "BalloonConfig":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(cli_args, attr) for attr in attrs})
        return engine_args
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--enable-balloon", action="store_true", default=False)
        parser.add_argument("--balloon-mode", type=str, default="exchange")
        parser.add_argument(
            "--balloon-ratio",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--exchange-batch-size", 
            type=int, 
            default=8,
            help="The number of blocks that are exchanged together during KVCache exchange"
        )
        parser.add_argument(
            "--max-balloon-memory-ratio",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--max-restore-memory-ratio",
            type=float,
            default=0.8,
        )
        parser.add_argument(
            "--restore-memory-threshold",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--restore-batch-size",
            type=int,
            default=8,
        )
        return parser

@dataclass
class DispatchConfig:
    dispatch_strategy: str = None


    @classmethod
    def from_cli_args(cls, cli_args) -> "DispatchConfig":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(cli_args, attr) for attr in attrs})
        return engine_args
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--dispatch-strategy", type=str, default="round-robin")
        return parser


@dataclass
class BenchConfig:
    # evaluated model
    model: str = None

    # request trace
    dataset: str = None
    dataset_offset: int = None
    nlimit: int = None
    ngroups: int = None
    max_model_len: int = None

    # request rate
    qps: int = None
    cv: float = None
    dist: str = None
    trace: str = None
    trace_offset: int = None

    trace_repeat_start: int = None
    trace_repeat_end: int = None
    trace_repeat_times: int = None
    trace_repeat_interval: int = None
    # time: int = None
    seed: float = None

    dataset_scale_factor: float = 1.0
    time_scale_factor: float = 1.0
    
    enable_look_ahead: bool = False
    enable_restore: bool = False

    # @deprecated
    enable_look_ahead: bool = False
    enable_packing: bool = False
    long_threshold: float = 1.0

    # @deprecated
    enable_decode_balance: bool = False
    enable_prefill_balance: bool = False
    enable_reorder: bool = False
    zigzag_chunk_limit: int = None
    

    # the directory to save logs
    log_path: str = None
    log_name: str = None

    num_gpu_blocks: int = None
    shuffle_dataset: bool = False
    server_dispatch_strategy: str = None
    enable_migration: bool = False

    @classmethod
    def from_cli_args(cls, cli_args) -> "BenchConfig":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(cli_args, attr) for attr in attrs})
        return engine_args

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument(
            "--model", 
            type=str,
            default="",
        )
        parser.add_argument("--dataset", type=str)
        parser.add_argument("--dataset-offset", type=int, default=0)
        parser.add_argument(
            "--nlimit", type=int, default=1000, help="The number of requests to send")
        parser.add_argument("--qps", type=int, default=1.0)
        parser.add_argument("--cv", type=float, default=0.1)
        parser.add_argument(
            "--dist", 
            type=str, 
            default="poisson",
            choices=["poisson", "uniform", "gamma"],    
        )
        parser.add_argument("--max-model-len", type=int, default=16384)
        parser.add_argument("--trace", type=str)
        parser.add_argument("--trace-offset", type=int, default=0)
        parser.add_argument("--trace-repeat-start", type=int, default=None)
        parser.add_argument("--trace-repeat-end", type=int, default=None)
        parser.add_argument("--trace-repeat-times", type=int, default=None)
        parser.add_argument("--trace-repeat-interval", type=int, default=None)
        parser.add_argument("--dataset-scale-factor", type=float, default=1.0)
        parser.add_argument("--time-scale-factor", type=float, default=1.0)
        parser.add_argument("--seed", type=float, default=42)
        parser.add_argument("--log-path", type=str, default="logs")
        parser.add_argument("--log-name", type=str, default="replica")
        parser.add_argument("--enable-decode-balance", action="store_true", default=False)
        parser.add_argument("--enable-prefill-balance", action="store_true", default=False)
        parser.add_argument("--enable-look-ahead", action="store_true", default=False)
        parser.add_argument("--enable-packing", action="store_true", default=False)
        parser.add_argument("--long-threshold", type=float, default=0.5)
        parser.add_argument("--enable-reorder", action="store_true", default=False)
        parser.add_argument("--zigzag-chunk-limit", type=int, default=16)
        parser.add_argument("--enable-restore", action="store_true", default=False)
        parser.add_argument("--num-gpu-blocks", type=int, default=None)
        parser.add_argument("--shuffle-dataset", action="store_true", default=False)
        parser.add_argument("--server-dispatch-strategy", type=str, default=None)
        parser.add_argument("--enable-migration", action="store_true", default=False)

class KunServeConfig:
    engine_config: EngineConfig
    balloon_config: BalloonConfig
    dispatch_config: DispatchConfig
    bench_config: BenchConfig
    def __init__(self, engine_config: EngineConfig, balloon_config: BalloonConfig, dispatch_config: DispatchConfig, bench_config: BenchConfig) -> None:
        self.engine_config = engine_config
        self.balloon_config = balloon_config
        self.dispatch_config = dispatch_config
        self.bench_config = bench_config

def load_config_from_yaml(engine_config: EngineConfig, balloon_config: BalloonConfig, dispatch_config: DispatchConfig, bench_config: BenchConfig, fname: str):
    with open(fname, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

        print(f"{cfg=}")
        
        if "engine_config" in cfg:
            engine_config_dict = cfg["engine_config"]
            engine_config.ngroups = engine_config_dict["ngroups"]
            engine_config.group_size = engine_config_dict["group_size"]
            engine_config.pp = engine_config_dict["pp"]
            engine_config.tp = engine_config_dict["tp"]
            engine_config.max_batch_size = engine_config_dict["max_batch_size"]
            engine_config.max_batch_tokens = engine_config_dict["max_batch_tokens"]
            engine_config.block_size = engine_config_dict["block_size"]
            if "use_tensor_cores" in engine_config_dict:
                engine_config.use_tensor_cores = engine_config_dict["use_tensor_cores"]
            if "gpu_util" in engine_config_dict:
                engine_config.gpu_util = engine_config_dict["gpu_util"]
            if "schedule_policy" in engine_config_dict:
                engine_config.schedule_policy = engine_config_dict["schedule_policy"]
            if "preempt_method" in engine_config_dict:
                engine_config.preempt_method = engine_config_dict["preempt_method"]
            if "nservers" in engine_config_dict:
                engine_config.nservers = engine_config_dict["nservers"]
            if "enable_chunked_prefill" in engine_config_dict:
                engine_config.enable_chunked_prefill = engine_config_dict["enable_chunked_prefill"]
            if "chunked_prefill_budget" in engine_config_dict:
                # engine_config.enable_chunked_prefill = True
                if "chunked_prefill_size" in engine_config_dict:
                    engine_config.chunked_prefill_size = engine_config_dict["chunked_prefill_size"]
                
                engine_config.chunked_prefill_budget = engine_config_dict["chunked_prefill_budget"]
                
            
        if "balloon_config" in cfg:
            balloon_config_dict = cfg["balloon_config"]
            if "balloon_mode" in balloon_config_dict:
                balloon_config.enable_balloon = True
                balloon_config.balloon_mode = balloon_config_dict["balloon_mode"]
                balloon_config.max_balloon_memory_ratio = balloon_config_dict["max_balloon_memory_ratio"]
                balloon_config.max_restore_memory_ratio = balloon_config_dict["max_restore_memory_ratio"]
                balloon_config.restore_memory_threshold = balloon_config_dict["restore_memory_threshold"]
            if "balloon_ratio" in balloon_config_dict:
                balloon_config.balloon_ratio = balloon_config_dict["balloon_ratio"]
            if "exchange_batch_size" in balloon_config_dict:
                balloon_config.exchange_batch_size = balloon_config_dict["exchange_batch_size"]
            if "restore_batch_size" in balloon_config_dict:
                balloon_config.restore_batch_size = balloon_config_dict["restore_batch_size"]
            
        if "dispatch_config" in cfg:
            dispatch_config_dict = cfg["dispatch_config"]
            dispatch_config.dispatch_strategy = dispatch_config_dict["dispatch_strategy"]
        
        if "bench_config" in cfg:
            bench_config_dict = cfg["bench_config"]
            if "model" in bench_config_dict:
                bench_config.model = bench_config_dict["model"]
            if "dataset" in bench_config_dict:
                bench_config.dataset = bench_config_dict["dataset"]
            if "dataset_offset" in bench_config_dict:
                bench_config.dataset_offset = bench_config_dict["dataset_offset"]
            if "nlimit" in bench_config_dict:
                bench_config.nlimit = bench_config_dict["nlimit"]
            if "dist" in bench_config_dict:
                bench_config.dist = bench_config_dict["dist"]
            if "qps" in bench_config_dict:
                bench_config.qps = bench_config_dict["qps"]
            if "cv" in bench_config_dict:
                bench_config.cv = bench_config_dict["cv"]
            if "max_model_len" in bench_config_dict:
                bench_config.max_model_len = bench_config_dict["max_model_len"]
            if "log_path" in bench_config_dict:
                bench_config.log_path = bench_config_dict["log_path"]
            if "trace" in bench_config_dict:
                bench_config.trace = bench_config_dict["trace"]
            if "trace_offset" in bench_config_dict:
                bench_config.trace_offset = bench_config_dict["trace_offset"]
            if "dataset_scale_factor" in bench_config_dict:
                bench_config.dataset_scale_factor = bench_config_dict["dataset_scale_factor"]
            if "time_scale_factor" in bench_config_dict:
                bench_config.time_scale_factor = bench_config_dict["time_scale_factor"]
            if "enable_decode_balance" in bench_config_dict:
                bench_config.enable_decode_balance = bench_config_dict["enable_decode_balance"]
            if "enable_prefill_balance" in bench_config_dict:
                bench_config.enable_prefill_balance = bench_config_dict["enable_prefill_balance"]
            if "enable_look_ahead" in bench_config_dict:
                bench_config.enable_look_ahead = bench_config_dict["enable_look_ahead"]
            if "enable_packing" in bench_config_dict:
                bench_config.enable_packing = bench_config_dict["enable_packing"]
            if "long_threshold" in bench_config_dict:
                bench_config.long_threshold = bench_config_dict["long_threshold"]
            if "enable_reorder" in bench_config_dict:
                bench_config.enable_reorder = bench_config_dict["enable_reorder"]
            if "zigzag_chunk_limit" in bench_config_dict:
                bench_config.zigzag_chunk_limit = bench_config_dict["bench_config_dict"]
            if "enable_restore" in bench_config_dict:
                bench_config.enable_restore = bench_config_dict["enable_restore"]
            if "trace_repeat_times" in bench_config_dict:
                bench_config.trace_repeat_times = bench_config_dict["trace_repeat_times"]
            if "trace_repeat_start" in bench_config_dict:
                bench_config.trace_repeat_start = bench_config_dict["trace_repeat_start"]
                bench_config.trace_repeat_end = bench_config_dict["trace_repeat_end"]
            if "trace_repeat_interval" in bench_config_dict:
                bench_config.trace_repeat_interval = bench_config_dict["trace_repeat_interval"]
            if "num_gpu_blocks" in bench_config_dict:
                bench_config.num_gpu_blocks = bench_config_dict["num_gpu_blocks"]
            if "shuffle_dataset" in bench_config_dict:
                bench_config.shuffle_dataset = bench_config_dict["shuffle_dataset"]
            if "dispatch_strategy" in bench_config_dict:
                bench_config.server_dispatch_strategy = bench_config_dict["server_dispatch_strategy"]
            if "enable_migration" in bench_config_dict:
                bench_config.enable_migration = bench_config_dict["enable_migration"]
            if "duplicate_server" in bench_config_dict:
                bench_config.duplicate_server = bench_config_dict["duplicate_server"]
            

def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    EngineConfig.add_cli_args(parser)
    BalloonConfig.add_cli_args(parser)
    DispatchConfig.add_cli_args(parser)
    BenchConfig.add_cli_args(parser)
    parser.add_argument(
        "--config", "-c", type=str, default=None)
    return parser

def get_cli_args(args) -> Tuple[BalloonConfig, DispatchConfig, BenchConfig]:
    engine_config = EngineConfig.from_cli_args(args)
    balloon_config = BalloonConfig.from_cli_args(args)
    dispatch_config = DispatchConfig.from_cli_args(args)
    bench_config = BenchConfig.from_cli_args(args)
    load_config_from_yaml(
        engine_config, 
        balloon_config, 
        dispatch_config, 
        bench_config, 
        args.config
    )
    return KunServeConfig(
        engine_config=engine_config, 
        balloon_config=balloon_config, 
        dispatch_config=dispatch_config, 
        bench_config=bench_config
    )

