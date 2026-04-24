git config --global credential.helper store

NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN python3 evaluation/bench.py -c configs/burstgpt