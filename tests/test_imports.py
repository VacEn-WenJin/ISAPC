import importlib


def test_core_imports():
    modules = [
        "main",
        "analysis.p2p",
        "analysis.voronoi",
        "analysis.radial",
        "utils.io",
        "utils.error_propagation",
        "muse",
        "config_manager",
    ]
    for m in modules:
        importlib.import_module(m)
