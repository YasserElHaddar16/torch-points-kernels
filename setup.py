from setuptools import setup, find_packages
import os
import sys
import glob
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

# Debug information
print(f"CUDA_HOME: {CUDA_HOME}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
if os.getenv("FORCE_CUDA", "0") == "1":
    WITH_CUDA = True
print(f"Building with CUDA: {WITH_CUDA}")

def get_ext_modules():
    ext_modules = []
    
    # Get absolute paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_include = os.path.join(root_dir, "cpu", "include")
    cuda_include = os.path.join(root_dir, "cuda", "include")
    
    include_dirs = [
        cpu_include,
        cuda_include,
    ]

    # CUDA Extension
    if WITH_CUDA:
        print("Preparing CUDA extension...")
        
        # Define CUDA sources (including metrics files)
        cuda_sources = [
            'cuda/src/ball_query.cpp',
            'cuda/src/ball_query_gpu.cu',
            'cuda/src/bindings.cpp',
            'cuda/src/chamfer_dist.cpp',
            'cuda/src/chamfer_dist_gpu.cu',
            'cuda/src/cubic_feature_sampling.cpp',
            'cuda/src/cubic_feature_sampling_gpu.cu',
            'cuda/src/gridding.cpp',
            'cuda/src/gridding_gpu.cu',
            'cuda/src/interpolate.cpp',
            'cuda/src/interpolate_gpu.cu',
            'cuda/src/sampling.cpp',
            'cuda/src/sampling_gpu.cu',
            'cuda/src/metrics.cpp',
            'cuda/src/metrics_gpu.cu'
        ]
        
        # Get absolute paths
        cuda_sources = [os.path.join(root_dir, src) for src in cuda_sources]
        
        # Verify source files exist
        for src in cuda_sources:
            if not os.path.exists(src):
                print(f"Warning: Source file not found: {src}")

        nvcc_flags = [
            "-O3",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-DWITH_CUDA",
            f"-I{cuda_include}",
            "-arch=sm_75",  # Adjust if needed
            "-std=c++17",
        ]

        extra_compile_args = {
            "cxx": ["-O3", "-DVERSION_GE_1_3", "-std=c++17"],
            "nvcc": nvcc_flags
        }

        cuda_ext = CUDAExtension(
            name="torch_points_kernels.points_cuda",
            sources=cuda_sources,
            include_dirs=[cuda_include] + include_dirs,
            extra_compile_args=extra_compile_args,
            define_macros=[
                ("WITH_CUDA", None),
                ("TORCH_EXTENSION_NAME", '"points_cuda"'),
            ],
        )
        ext_modules.append(cuda_ext)
        print("CUDA extension prepared")

    # CPU Extension
    if WITH_CUDA:
        cpu_sources = [
            os.path.join(root_dir, "cpu", "src", "neighbors.cpp"),
            os.path.join(root_dir, "cpu", "src", "knn.cpp"),
            os.path.join(root_dir, "cpu", "src", "fps.cpp"),
            os.path.join(root_dir, "cpu", "src", "ball_query.cpp"),
            os.path.join(root_dir, "cpu", "src", "bindings.cpp"),
            os.path.join(root_dir, "cpu", "src", "interpolate.cpp"),
        ]
        
        cpu_ext = CppExtension(
            "torch_points_kernels.points_cpu",
            sources=cpu_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args
        )
        ext_modules.append(cpu_ext)
    
    return ext_modules

class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        # Disable ninja to avoid issues with CUDA compilation
        super().__init__(*args, no_python_abi_suffix=True, use_ninja=False, **kwargs)

# Force rebuild if requested
if os.getenv('FORCE_REBUILD'):
    sys.argv.extend(['clean', 'build_ext', '--inplace'])

setup(
    name="torch-points-kernels",
    version="0.7.1",
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': CustomBuildExtension},
    install_requires=[
        'torch>=2.1.0',
        'numpy>=1.20,<2.0',
        'scikit-learn'
    ],
    python_requires='>=3.8',
)