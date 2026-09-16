import importlib.util
from pathlib import Path


def load_worker():
    path = Path(__file__).with_name("worker.py")
    spec = importlib.util.spec_from_file_location("qrf_v024_worker", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_adapter_uses_frozen_core_and_lock():
    worker = load_worker(); identity = worker.source_identity()
    assert set(identity) == {"adapter", "qrf_model", "preprocessing", "pyproject", "lock"}
    assert worker.PROTOCOL == "QRF_FULLTRAIN_LEAF_v1" and worker.PARAMETERS["n_estimators"] == 256

