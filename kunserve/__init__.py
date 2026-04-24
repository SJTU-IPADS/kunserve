import os
import torch

BASE_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(BASE_DIR, "../FlashTransformer/build/lib", "libflash_pybinding.so")

print(f"Try loading c++ lib from path {LIB_PATH}.")
if not os.path.exists(LIB_PATH):
    raise RuntimeError(
        f"Could not find the FlashTransformer library libflash_pybinding.so at {LIB_PATH}. "
        "Please build the FlashTransformer library first or put it at the right place."
    )
    

torch.ops.load_library(LIB_PATH)
