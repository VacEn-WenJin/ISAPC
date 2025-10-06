"""
Utilities to control CPU/threading to avoid BLAS oversubscription.

On many NumPy/SciPy builds, BLAS/OpenMP libraries spin up their own threads.
When combined with joblib threading/process parallelism, this causes severe
oversubscription (too many threads) and slows everything down.

Use set_safe_thread_limits() early in the program to cap BLAS threads.
"""
from __future__ import annotations

import os
from typing import Optional


def set_env_if_missing(key: str, value: str) -> None:
    """Set an environment variable only if it's not already set."""
    if key not in os.environ or not os.environ[key]:
        os.environ[key] = value


def set_safe_thread_limits(blas_threads: int = 1) -> None:
    """Pin BLAS/OpenMP/NumExpr threads to a safe value for parallel workloads.

    Parameters
    ----------
    blas_threads : int
        Number of threads for BLAS/OpenMP backends. 1 is safest to allow
        outer-level parallelism (joblib/process/thread pool) to scale.
    """
    # Environment variables commonly used by BLAS/OpenMP backends
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        # Additional variants seen in some environments
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        set_env_if_missing(var, str(max(1, int(blas_threads))))

    # Try to use threadpoolctl, if available, to enforce limits at runtime
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(limits=max(1, int(blas_threads)), user_api=["blas", "openmp"])  # noqa: F401
    except Exception:
        # It's fine if threadpoolctl isn't available; env vars will still help.
        pass


def recommend_n_jobs(default: int = 8) -> int:
    """Return a reasonable default for n_jobs on typical 16C/32T machines."""
    try:
        import os

        cpu_count = os.cpu_count() or 8
    except Exception:
        cpu_count = 8

    # Favor something like one job per core half, capped.
    return min(default, max(2, cpu_count // 4))
