from setuptools import setup, Extension
import pybind11
# python3 setup.py build_ext --inplace 
# Lista de todos tus archivos .cpp fuente
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
    # EXCEPTO main.cpp (no queremos un ejecutable, sino una librería)
]

ext_modules = [
    Extension(
        "learntorch", # Nombre del paquete en Python
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
    version="0.1",
    ext_modules=ext_modules,
)