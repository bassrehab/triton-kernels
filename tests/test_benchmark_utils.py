import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_utils_module():
    triton_stub = types.SimpleNamespace(testing=types.SimpleNamespace())
    sys.modules.setdefault("triton", triton_stub)

    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "utils.py"
    spec = importlib.util.spec_from_file_location("benchmark_utils_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_specs(name: str) -> dict:
    gpu_specs = load_utils_module().GPU_SPECS
    specs = gpu_specs.get(name)
    if specs is not None:
        return specs

    for known_name, known_specs in gpu_specs.items():
        if known_name in name or name in known_name:
            return known_specs

    return gpu_specs["default"]


def test_known_rtx_5090_specs_present():
    specs = load_utils_module().GPU_SPECS["NVIDIA GeForce RTX 5090"]

    assert specs["bandwidth_gb_s"] == pytest.approx(1792)
    assert specs["fp16_tflops"] == pytest.approx(210)


def test_partial_match_resolves_rtx_5090():
    gpu_specs = load_utils_module().GPU_SPECS
    specs = resolve_specs("RTX 5090")

    assert specs == gpu_specs["NVIDIA GeForce RTX 5090"]
