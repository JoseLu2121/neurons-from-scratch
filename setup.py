from setuptools import setup, Extension, find_packages
import pybind11

cpp_files = [
    "src/bindings.cpp",
    "src/tensor.cpp",
    "src/utils.cpp",
    "src/unit.cpp",
    "src/loss.cpp",
    "src/ops.cpp",
    "src/optimizers.cpp",
    "src/neuron.cpp",
    "src/layer.cpp",
    "src/trainer.cpp",
    "src/CPUBackend.cpp",
    "src/CPUBackendOptimized.cpp",
    "src/GEMMOptimizedBackend.cpp",
]

ext_modules = [
    Extension(
        "_learntorch",
        sorted(cpp_files),
        include_dirs=[
            pybind11.get_include(),
            "include"
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-mavx2", "-mfma"],
    ),
]

setup(
    name="learntorch",
    version="1.0",
    packages=find_packages(),
    package_data={"learntorch": ["py.typed", "*.pyi"]},
    ext_modules=ext_modules,
)
