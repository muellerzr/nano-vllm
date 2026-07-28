from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


setup(
    ext_modules=[
        CUDAExtension(
            "nanovllm._router_gemm",
            ["csrc/router_gemm.cu"],
            include_dirs=[
                "/usr/local/cuda/include",
                str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")
            ],
            extra_compile_args={"nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
