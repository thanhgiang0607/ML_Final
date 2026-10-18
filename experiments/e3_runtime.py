#!/usr/bin/env python3
"""E3: Tốc độ & bộ nhớ

Người phụ trách: C  ·  Branch: e3-runtime

Thiết lập:
---------
- Sinh dữ liệu ngay trong script: make_blobs(n, centers=5, cluster_std=1.0, random_state=0), d = 2.
 * Theo n: n ∈ {500, 1.000, 2.000, 5.000, 10.000, 20.000, 50.000}. h cố định = estimate_bandwidth tính một lần trên mẫu 2.000 điểm (để chỉ n thay đổi).
 * Theo h: n = 5.000, h = α·h₀ với α ∈ {0.25, 0.5, 1, 2}.
- Các bản cần đo:
    custom brute · custom kdtree · custom kdtree + bin_seeding
    sklearn MeanShift(n_jobs=1) · sklearn + bin_seeding
- Thời gian: trung vị 3 lần. Bộ nhớ đỉnh: tracemalloc (bắt đầu trước fit, lấy get_traced_memory()[1]).
- Giới hạn: lần chạy nào > 10 phút thì bỏ các n lớn hơn của bản đó, ghi "quá thời gian" vào CSV.
- Ghi cấu hình máy (CPU, RAM, phiên bản Python/NumPy/sklearn) vào env.txt.
- Nhận xét: KD-Tree nhanh hơn brute từ n bao nhiêu; khi h lớn KD-Tree còn lợi không; bin seeding giảm số seed và thời gian bao nhiêu.

Đầu ra
--------------------
    results/e3/e3_runs.csv
    Mỗi dòng: thí nghiệm (n / h), n, h, bản, lần lặp, thời gian (s), bộ nhớ đỉnh (MB), số seed, số cụm
    results/e3/e3_summary.csv
    Trung vị thời gian, bộ nhớ theo (n, bản) và (h, bản); hệ số tăng tốc so với brute
    results/e3/e3_time_vs_n.png
    Thời gian theo n, trục log–log, mỗi bản 1 đường
    results/e3/e3_memory_vs_n.png
    Bộ nhớ đỉnh theo n
    results/e3/e3_time_vs_h.png
    Thời gian theo h (n = 5.000)
    results/e3/env.txt
    Cấu hình máy và phiên bản thư viện
    report/THUC_NGHIEM_E3.md
    Kết luận RQ3
"""


from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.datasets import make_blobs
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = [
    "custom_brute",
    "custom_kdtree",
    "custom_kdtree_bin",
    "sklearn",
    "sklearn_bin",
]
METHOD_LABELS = {
    "custom_brute": "Custom brute",
    "custom_kdtree": "Custom KD-Tree",
    "custom_kdtree_bin": "Custom KD-Tree + bin seeding",
    "sklearn": "sklearn (bin seeding off)",
    "sklearn_bin": "sklearn (bin seeding on)",
}
RAW_FIELDS = [
    "experiment", "n", "h", "alpha", "method", "repeat", "random_state",
    "elapsed_s", "peak_memory_mb", "n_seeds", "n_clusters",
    "neighbor_queries", "mean_neighbors_per_query", "mean_iterations",
    "status", "error",
]
SUMMARY_FIELDS = [
    "experiment", "n", "h", "alpha", "method", "successful_runs",
    "median_time_s", "q1_time_s", "q3_time_s", "iqr_time_s",
    "median_peak_memory_mb", "q1_peak_memory_mb", "q3_peak_memory_mb",
    "iqr_peak_memory_mb", "speedup_vs_custom_brute",
    "median_n_seeds", "median_n_clusters", "mean_neighbors_per_query",
    "mean_iterations", "status",
]


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {config_path}. "
            "Hãy đặt configs/e3_runtime.yaml trong repository hoặc dùng --config."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _read_cpu_name() -> str:
    candidates = [platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")]
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        candidates.insert(0, line.split(":", 1)[1].strip())
                        break
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                candidates.insert(0, result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return next((x for x in candidates if x and x.strip()), platform.machine())


def _read_ram_gb() -> float | None:
    try:
        import psutil  # optional; do not require it for the benchmark
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            return (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                / (1024 ** 3)
            )
        except (ValueError, OSError, AttributeError):
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    return None


def write_environment(out_dir: Path) -> dict[str, Any]:
    ram_gb = _read_ram_gb()
    env = {
        "cpu": _read_cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gb": round(ram_gb, 3) if ram_gb is not None else None,
        "operating_system": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "script": "experiments/e3_runtime.py",
        "memory_measurement": (
            "tracemalloc peak Python-traced memory during estimator.fit(X); "
            "not total native/process RSS"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "E3 — Environment",
        f"CPU: {env['cpu']}",
        f"Logical CPUs: {env['logical_cpu_count']}",
        f"RAM total (GB): {env['ram_gb'] if env['ram_gb'] is not None else 'unavailable'}",
        f"OS: {env['operating_system']}",
        f"Python: {env['python']}",
        f"NumPy: {env['numpy']}",
        f"SciPy: {env['scipy']}",
        f"scikit-learn: {env['scikit_learn']}",
        "Memory metric: tracemalloc peak during fit; native allocations may not be fully captured.",
    ]
    (out_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def _estimate_seed_count(X: np.ndarray, h: float, method: str) -> int:
    """Return n or the number of binned starting seeds, when available."""
    uses_bin = method in {"custom_kdtree_bin", "sklearn_bin"}
    if not uses_bin:
        return int(len(X))
    try:
        if method.startswith("custom"):
            from meanshift.seeding import get_bin_seeds
        else:
            sk_ms = importlib.import_module("sklearn.cluster._mean_shift")
            get_bin_seeds = sk_ms.get_bin_seeds
        seeds = get_bin_seeds(X, bin_size=h, min_bin_freq=1)
        return int(len(seeds))
    except Exception:
        # Benchmark should still run if an implementation exposes a different helper.
        return -1


def _mean_neighbor_stats(stats: dict[str, Any]) -> tuple[int, float]:
    count = int(stats.get("neighbor_queries", 0))
    if count <= 0:
        return 0, float("nan")
    return count, float(stats.get("neighbor_count_sum", 0)) / count


def _record_sklearn_neighbor_rows(stats: dict[str, Any], rows: Any) -> None:
    """Aggregate lengths of the neighbor index arrays returned by sklearn."""
    try:
        for row in rows:
            try:
                size = len(row)
            except TypeError:
                size = 1
            stats["neighbor_count_sum"] += int(size)
            stats["neighbor_queries"] += 1
    except TypeError:
        try:
            size = len(rows)
            stats["neighbor_count_sum"] += int(size)
            stats["neighbor_queries"] += 1
        except TypeError:
            pass


def _record_custom_neighbor_result(stats: dict[str, Any], result: Any) -> None:
    """Count neighbors returned by the custom API: [(idx, sq_dist), ...].

    In MeanShiftCustom._shift_seeds, query_radius is called with a batch of
    active seeds. Therefore each pair in the returned list corresponds to one
    seed-query, and len(pair[0]) is that query's number of neighbors.
    """
    try:
        # Support both the documented batched form (list of pairs) and a
        # single-query pair (idx, sq_dist), should a backend return that form.
        if isinstance(result, tuple) and len(result) == 2:
            rows = [result]
        elif isinstance(result, list):
            rows = result
        else:
            return

        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                continue
            indices = row[0]
            try:
                neighbor_count = len(indices)
            except TypeError:
                neighbor_count = 1
            stats["neighbor_count_sum"] += int(neighbor_count)
            stats["neighbor_queries"] += 1
    except Exception:
        # Counters are diagnostic only; do not make them break a model fit.
        pass


def _make_estimator(method: str, h: float, cfg: dict[str, Any]):
    max_iter = int(cfg["benchmark"].get("max_iter", 300))
    tol = float(cfg["benchmark"].get("tol", 1e-3))
    if method.startswith("custom"):
        from meanshift import MeanShiftCustom
        if method == "custom_brute":
            backend, bin_seeding = "brute", False
        elif method == "custom_kdtree":
            backend, bin_seeding = "kdtree", False
        elif method == "custom_kdtree_bin":
            backend, bin_seeding = "kdtree", True
        else:
            raise ValueError(f"Unknown custom method: {method}")
        return MeanShiftCustom(
            bandwidth=float(h),
            kernel="flat",
            neighbor_search=backend,
            bin_seeding=bin_seeding,
            max_iter=max_iter,
            tol=tol,
            cluster_all=True,
        )
    if method in {"sklearn", "sklearn_bin"}:
        from sklearn.cluster import MeanShift
        return MeanShift(
            bandwidth=float(h),
            bin_seeding=(method == "sklearn_bin"),
            n_jobs=1,
            max_iter=max_iter,
            cluster_all=True,
        )
    raise ValueError(f"Unknown method: {method}")


def _fit_worker(conn, X: np.ndarray, method: str, h: float, cfg: dict[str, Any]) -> None:
    """Child process isolates each fit so the 600s timeout can be enforced."""
    import tracemalloc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            estimator = _make_estimator(method, h, cfg)
            n_seeds = _estimate_seed_count(X, h, method)
            stats: dict[str, Any] = {
                "neighbor_queries": 0,
                "neighbor_count_sum": 0,
                "iterations": [],
            }
            restore_actions = []

            # Instrument custom radius queries to collect actual neighbor counts.
            if method.startswith("custom"):
                try:
                    neighbor_module = importlib.import_module("meanshift.neighbors")
                    for cls_name in ("BruteForceSearch", "KDTreeSearch"):
                        cls = getattr(neighbor_module, cls_name, None)
                        if cls is None or not hasattr(cls, "query_radius"):
                            continue
                        original = cls.query_radius

                        def custom_wrapper(self, *args, _original=original, **kwargs):
                            result = _original(self, *args, **kwargs)
                            _record_custom_neighbor_result(stats, result)
                            return result

                        cls.query_radius = custom_wrapper
                        restore_actions.append((cls, "query_radius", original))
                except Exception:
                    pass
            else:
                # sklearn's MeanShift uses radius_neighbors inside each seed trajectory.
                try:
                    sk_ms = importlib.import_module("sklearn.cluster._mean_shift")
                    nn_cls = sk_ms.NearestNeighbors
                    original_radius = nn_cls.radius_neighbors

                    def sklearn_radius_wrapper(self, *args, _original=original_radius, **kwargs):
                        result = _original(self, *args, **kwargs)
                        return_distance = kwargs.get("return_distance", True)
                        if len(args) >= 3:
                            return_distance = args[2]
                        indices = result[1] if return_distance and isinstance(result, tuple) else result
                        _record_sklearn_neighbor_rows(stats, indices)
                        return result

                    nn_cls.radius_neighbors = sklearn_radius_wrapper
                    restore_actions.append((nn_cls, "radius_neighbors", original_radius))

                    original_seed_fit = getattr(sk_ms, "_mean_shift_single_seed", None)
                    if original_seed_fit is not None:
                        def sklearn_seed_wrapper(*args, _original=original_seed_fit, **kwargs):
                            result = _original(*args, **kwargs)
                            if isinstance(result, tuple) and len(result) >= 3:
                                stats["iterations"].append(float(result[2]))
                            return result
                        sk_ms._mean_shift_single_seed = sklearn_seed_wrapper
                        restore_actions.append((sk_ms, "_mean_shift_single_seed", original_seed_fit))
                except Exception:
                    pass

            # Signal the parent immediately before the timed fit.
            conn.send({"event": "started"})
            tracemalloc.start()
            start = time.perf_counter()
            try:
                estimator.fit(X)
            finally:
                elapsed = time.perf_counter() - start
                _current, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                for obj, attr, original in reversed(restore_actions):
                    setattr(obj, attr, original)

            n_clusters = -1
            centers = getattr(estimator, "cluster_centers_", None)
            if centers is not None:
                try:
                    n_clusters = int(len(centers))
                except TypeError:
                    pass
            if n_clusters < 0:
                labels = getattr(estimator, "labels_", None)
                if labels is not None:
                    n_clusters = int(len(np.unique(labels[labels >= 0])))

            mean_iterations = float("nan")
            # MeanShiftCustom exposes seed_n_iter_ (per-seed values) and
            # n_iter_ (the maximum). Prefer per-seed data so this CSV field is
            # genuinely a mean rather than accidentally reporting the maximum.
            per_seed_iterations = getattr(estimator, "seed_n_iter_", None)
            if per_seed_iterations is None:
                per_seed_iterations = getattr(estimator, "n_iter_", None)
            if per_seed_iterations is not None:
                try:
                    arr = np.asarray(per_seed_iterations, dtype=float).ravel()
                    if arr.size:
                        mean_iterations = float(np.mean(arr))
                except (TypeError, ValueError):
                    pass
            if not math.isfinite(mean_iterations) and stats["iterations"]:
                mean_iterations = float(np.mean(stats["iterations"]))

            n_queries, mean_neighbors = _mean_neighbor_stats(stats)
            conn.send({
                "event": "result",
                "data": {
                    "status": "ok",
                    "elapsed_s": float(elapsed),
                    "peak_memory_mb": float(peak_bytes / (1024 ** 2)),
                    "n_seeds": int(n_seeds),
                    "n_clusters": int(n_clusters),
                    "neighbor_queries": int(n_queries),
                    "mean_neighbors_per_query": float(mean_neighbors),
                    "mean_iterations": float(mean_iterations),
                    "error": "",
                },
            })
    except BaseException as exc:
        try:
            conn.send({
                "event": "result",
                "data": {
                    "status": "error",
                    "elapsed_s": float("nan"),
                    "peak_memory_mb": float("nan"),
                    "n_seeds": -1,
                    "n_clusters": -1,
                    "neighbor_queries": 0,
                    "mean_neighbors_per_query": float("nan"),
                    "mean_iterations": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
                },
            })
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _safe_poll(conn, timeout=0):
    try:
        return conn.poll(timeout)
    except (EOFError, OSError):
        # Pipe đã đóng hoặc worker kết thúc bất thường.
        return False

def fit_with_timeout(
    X: np.ndarray, method: str, h: float, cfg: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    """Run one fit in a child process and enforce timeout on the fit itself."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_fit_worker, args=(child_conn, X, method, float(h), cfg))
    proc.start()
    child_conn.close()
    fit_started_at: float | None = None
    result = None
    timed_out = False

    try:
        while proc.is_alive():
            if _safe_poll(parent_conn, 0.1):
                try:
                    message = parent_conn.recv()
                except EOFError:
                    message = None
                if message:
                    if message.get("event") == "started":
                        fit_started_at = time.monotonic()
                    elif message.get("event") == "result":
                        result = message.get("data")
            if fit_started_at is not None and time.monotonic() - fit_started_at > timeout_seconds:
                timed_out = True
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()
                    proc.join(timeout=3)
                break
        proc.join(timeout=1)
        # A result message may already be waiting after the process exits.
        while _safe_poll(parent_conn):
            try:
                message = parent_conn.recv()
            except EOFError:
                break
            if message and message.get("event") == "result":
                result = message.get("data")
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

    if timed_out:
        return {
            "status": "timeout", "elapsed_s": float(timeout_seconds),
            "peak_memory_mb": float("nan"), "n_seeds": -1, "n_clusters": -1,
            "neighbor_queries": 0, "mean_neighbors_per_query": float("nan"),
            "mean_iterations": float("nan"),
            "error": f"quá thời gian (> {timeout_seconds:g} giây)",
        }
    if result is None:
        return {
            "status": "error", "elapsed_s": float("nan"),
            "peak_memory_mb": float("nan"), "n_seeds": -1, "n_clusters": -1,
            "neighbor_queries": 0, "mean_neighbors_per_query": float("nan"),
            "mean_iterations": float("nan"),
            "error": f"Worker kết thúc bất thường (exitcode={proc.exitcode}) mà không trả kết quả.",
        }
    return result


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return value if math.isfinite(float(value)) else ""
    return value


def _append_row(writer: csv.DictWriter, file_obj, row: dict[str, Any]) -> None:
    writer.writerow({k: _to_csv_value(row.get(k, "")) for k in RAW_FIELDS})
    file_obj.flush()


def _base_row(experiment: str, n: int, h: float, alpha: float | None,
              method: str, repeat: int, random_state: int, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "experiment": experiment,
        "n": int(n),
        "h": float(h),
        "alpha": "" if alpha is None else float(alpha),
        "method": method,
        "repeat": int(repeat),
        "random_state": int(random_state),
    }
    row.update(result)
    return row


def _skipped_result(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status, "elapsed_s": float("nan"), "peak_memory_mb": float("nan"),
        "n_seeds": -1, "n_clusters": -1, "neighbor_queries": 0,
        "mean_neighbors_per_query": float("nan"), "mean_iterations": float("nan"),
        "error": reason,
    }


def _run_sweep_n(writer: csv.DictWriter, f, cfg: dict[str, Any], h0: float,
                 n_values: list[int], repeats: int, timeout: float) -> list[dict[str, Any]]:
    ds = cfg["dataset"]
    all_rows: list[dict[str, Any]] = []
    random_state = int(ds.get("random_state", 0))
    for method in METHODS:
        stop_larger_n = False
        for n in n_values:
            if stop_larger_n:
                for repeat in range(1, repeats + 1):
                    row = _base_row(
                        "n", n, h0, None, method, repeat, random_state,
                        _skipped_result("skipped_after_timeout", "bỏ n lớn hơn sau timeout của biến thể này"),
                    )
                    _append_row(writer, f, row)
                    all_rows.append(row)
                continue

            X, _ = make_blobs(
                n_samples=int(n), centers=int(ds.get("centers", 5)),
                cluster_std=float(ds.get("cluster_std", 1.0)),
                random_state=random_state, n_features=int(ds.get("dimensions", 2)),
            )
            timeout_at_this_n = False
            for repeat in range(1, repeats + 1):
                result = fit_with_timeout(X, method, h0, cfg, timeout)
                row = _base_row("n", n, h0, None, method, repeat, random_state, result)
                _append_row(writer, f, row)
                all_rows.append(row)
                print(
                    f"[E3:n] method={method:18s} n={n:6d} repeat={repeat}/{repeats} "
                    f"status={result['status']} time={result['elapsed_s'] if math.isfinite(float(result['elapsed_s'])) else 'NA'}",
                    flush=True,
                )
                if result["status"] == "timeout":
                    timeout_at_this_n = True
                    for remaining in range(repeat + 1, repeats + 1):
                        skipped = _base_row(
                            "n", n, h0, None, method, remaining, random_state,
                            _skipped_result("skipped_after_timeout", "bỏ lượt lặp còn lại sau timeout"),
                        )
                        _append_row(writer, f, skipped)
                        all_rows.append(skipped)
                    break
            if timeout_at_this_n:
                stop_larger_n = True
    return all_rows


def _run_sweep_h(writer: csv.DictWriter, f, cfg: dict[str, Any], h0: float,
                 n: int, alphas: list[float], repeats: int, timeout: float) -> list[dict[str, Any]]:
    ds = cfg["dataset"]
    random_state = int(ds.get("random_state", 0))
    X, _ = make_blobs(
        n_samples=int(n), centers=int(ds.get("centers", 5)),
        cluster_std=float(ds.get("cluster_std", 1.0)),
        random_state=random_state, n_features=int(ds.get("dimensions", 2)),
    )
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        stop_larger_alpha = False
        for alpha in alphas:
            h = float(alpha) * float(h0)
            if stop_larger_alpha:
                for repeat in range(1, repeats + 1):
                    row = _base_row(
                        "h", n, h, alpha, method, repeat, random_state,
                        _skipped_result("skipped_after_timeout", "bỏ các alpha lớn hơn sau timeout của biến thể này"),
                    )
                    _append_row(writer, f, row)
                    all_rows.append(row)
                continue
            timeout_at_alpha = False
            for repeat in range(1, repeats + 1):
                result = fit_with_timeout(X, method, h, cfg, timeout)
                row = _base_row("h", n, h, alpha, method, repeat, random_state, result)
                _append_row(writer, f, row)
                all_rows.append(row)
                print(
                    f"[E3:h] method={method:18s} n={n:6d} alpha={alpha:g} h={h:.6g} "
                    f"repeat={repeat}/{repeats} status={result['status']} "
                    f"time={result['elapsed_s'] if math.isfinite(float(result['elapsed_s'])) else 'NA'}",
                    flush=True,
                )
                if result["status"] == "timeout":
                    timeout_at_alpha = True
                    for remaining in range(repeat + 1, repeats + 1):
                        skipped = _base_row(
                            "h", n, h, alpha, method, remaining, random_state,
                            _skipped_result("skipped_after_timeout", "bỏ lượt lặp còn lại sau timeout"),
                        )
                        _append_row(writer, f, skipped)
                        all_rows.append(skipped)
                    break
            # Additional safety: after a timeout at h, skip larger alpha values for this method.
            if timeout_at_alpha:
                stop_larger_alpha = True
    return all_rows


def _numeric(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            v = float(row.get(field, ""))
            if math.isfinite(v):
                vals.append(v)
        except (TypeError, ValueError):
            continue
    return np.asarray(vals, dtype=float)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("experiment", "n", "h", "alpha", "method")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = tuple(row.get(k, "") for k in key_fields)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        good = [r for r in group if r.get("status") == "ok"]
        time_vals = _numeric(good, "elapsed_s")
        mem_vals = _numeric(good, "peak_memory_mb")
        seed_vals = _numeric(good, "n_seeds")
        cluster_vals = _numeric(good, "n_clusters")
        neighbor_vals = _numeric(good, "mean_neighbors_per_query")
        iter_vals = _numeric(good, "mean_iterations")

        def quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
            if values.size == 0:
                return (float("nan"),) * 4
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            return float(med), float(q1), float(q3), float(q3 - q1)

        med_t, q1_t, q3_t, iqr_t = quantiles(time_vals)
        med_m, q1_m, q3_m, iqr_m = quantiles(mem_vals)
        status = "ok" if good else next((r.get("status", "unknown") for r in group), "unknown")
        summary.append({
            **dict(zip(key_fields, key)),
            "successful_runs": int(len(good)),
            "median_time_s": med_t, "q1_time_s": q1_t, "q3_time_s": q3_t, "iqr_time_s": iqr_t,
            "median_peak_memory_mb": med_m, "q1_peak_memory_mb": q1_m,
            "q3_peak_memory_mb": q3_m, "iqr_peak_memory_mb": iqr_m,
            "speedup_vs_custom_brute": float("nan"),
            "median_n_seeds": float(np.median(seed_vals)) if seed_vals.size else float("nan"),
            "median_n_clusters": float(np.median(cluster_vals)) if cluster_vals.size else float("nan"),
            "mean_neighbors_per_query": float(np.mean(neighbor_vals)) if neighbor_vals.size else float("nan"),
            "mean_iterations": float(np.mean(iter_vals)) if iter_vals.size else float("nan"),
            "status": status,
        })

    # Speedup baseline: custom brute for the same experiment/n/h/alpha.
    by_key = {}
    for row in summary:
        k = (row["experiment"], row["n"], row["h"], row["alpha"])
        by_key.setdefault(k, {})[row["method"]] = row
    for group in by_key.values():
        baseline = group.get("custom_brute")
        if not baseline or baseline.get("status") != "ok":
            continue
        base_time = baseline.get("median_time_s")
        if not isinstance(base_time, (int, float)) or not math.isfinite(float(base_time)) or base_time <= 0:
            continue
        for row in group.values():
            t = row.get("median_time_s")
            if isinstance(t, (int, float)) and math.isfinite(float(t)) and float(t) > 0:
                row["speedup_vs_custom_brute"] = float(base_time) / float(t)
    return sorted(summary, key=lambda r: (r["experiment"], str(r["method"]), float(r["n"]), float(r["h"])))


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_csv_value(row.get(k, "")) for k in SUMMARY_FIELDS})


def _summary_value(rows: list[dict[str, Any]], experiment: str, method: str,
                   n: int | None = None, alpha: float | None = None) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["experiment"] == experiment and r["method"] == method]
    if n is not None:
        candidates = [r for r in candidates if int(float(r["n"])) == int(n)]
    if alpha is not None:
        candidates = [r for r in candidates if math.isclose(float(r["alpha"]), alpha, rel_tol=1e-9, abs_tol=1e-12)]
    candidates = [r for r in candidates if r["status"] == "ok" and isinstance(r.get("median_time_s"), (int, float)) and math.isfinite(float(r["median_time_s"]))]
    return candidates[0] if candidates else None


def create_plots(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Runtime vs n: log-log. Use n sweep rows only.
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "n" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: int(float(r["n"])))
        xs = [int(float(r["n"])) for r in rows if _valid_positive(r.get("median_time_s"))]
        ys = [float(r["median_time_s"]) for r in rows if _valid_positive(r.get("median_time_s"))]
        if xs and ys:
            ax.loglog(xs, ys, marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xlabel("Số điểm dữ liệu n")
    ax.set_ylabel("Thời gian fit trung vị (giây)")
    ax.set_title("E3 — Thời gian chạy theo kích thước dữ liệu (log–log)")
    ax.grid(True, which="both", alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_time_vs_n.png", dpi=150)
    plt.close(fig)

    # Peak traced memory vs n.
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "n" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: int(float(r["n"])))
        pairs = [(int(float(r["n"])), float(r["median_peak_memory_mb"])) for r in rows
                 if _valid_positive(r.get("median_peak_memory_mb"))]
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xscale("log")
    ax.set_xlabel("Số điểm dữ liệu n (thang log)")
    ax.set_ylabel("Bộ nhớ đỉnh tracemalloc (MB)")
    ax.set_title("E3 — Bộ nhớ đỉnh theo kích thước dữ liệu")
    ax.grid(True, which="both", alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_memory_vs_n.png", dpi=150)
    plt.close(fig)

    # Runtime vs h at fixed n.
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "h" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: float(r["h"]))
        pairs = [(float(r["h"]), float(r["median_time_s"])) for r in rows
                 if _valid_positive(r.get("h")) and _valid_positive(r.get("median_time_s"))]
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel("Thời gian fit trung vị (giây)")
    ax.set_title("E3 — Thời gian chạy theo bandwidth h")
    ax.grid(True, alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_time_vs_h.png", dpi=150)
    plt.close(fig)


def _valid_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def write_report(report_path: Path, cfg: dict[str, Any], h0: float, env: dict[str, Any],
                 summary_rows: list[dict[str, Any]], repeats: int, timeout: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n_values = cfg["n_sweep"]["values"]
    h_sweep = cfg["h_sweep"]
    alphas = h_sweep["alphas"]
    lines = [
        "# Thực nghiệm E3 — Tốc độ và bộ nhớ",
        "",
        "## 1. Mục tiêu",
        "Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, và trong điều kiện nào?",
        "",
        "## 2. Cấu hình thực nghiệm",
        f"- Dữ liệu: `make_blobs`, centers={cfg['dataset'].get('centers', 5)}, cluster_std={cfg['dataset'].get('cluster_std', 1.0)}, d={cfg['dataset'].get('dimensions', 2)}, random_state={cfg['dataset'].get('random_state', 0)}.",
        f"- Bandwidth tham chiếu: h0={h0:.8g}, ước lượng một lần trên mẫu n={cfg['dataset'].get('reference_n', 2000)} điểm với quantile={cfg['dataset'].get('bandwidth_quantile', 0.3)}.",
        f"- Quét n: {', '.join(str(x) for x in n_values)}; giữ h=h0.",
        f"- Quét h: n={h_sweep['n']}; alpha={', '.join(str(a) for a in alphas)}; h=alpha*h0.",
        f"- Số lần lặp theo cấu hình hiện tại: {repeats}; timeout fit: {timeout:g} giây/lượt.",
        "- Các biến thể: Custom brute; Custom KD-Tree; Custom KD-Tree + bin seeding; sklearn MeanShift (bin seeding off/on, n_jobs=1).",
        "- Đo thời gian bằng `time.perf_counter()`; bộ nhớ đỉnh bằng `tracemalloc` bắt đầu trước `fit(X)`. Báo cáo median và IQR.",
        f"- Môi trường: CPU {env['cpu']}; RAM tổng (GB) {env['ram_gb']}; Python {env['python']}; NumPy {env['numpy']}; SciPy {env['scipy']}; scikit-learn {env['scikit_learn']}.",
        "",
        "## 3. Kết quả tóm tắt",
        "Bảng chi tiết nằm ở `../results/e3/e3_summary.csv`; dữ liệu từng lượt chạy ở `../results/e3/e3_runs.csv` (không commit theo quy ước nhóm).",
        "",
        "### 3.1. Thời gian theo n",
        "| n | Custom brute (s) | Custom KD-Tree (s) | KD-Tree + bin seeding (s) | sklearn off (s) | sklearn on (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in n_values:
        vals = []
        for method in METHODS:
            r = _summary_value(summary_rows, "n", method, n=int(n))
            vals.append(f"{float(r['median_time_s']):.4g}" if r else "—")
        lines.append("| " + str(n) + " | " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "### 3.2. Thời gian theo h",
        "| alpha | h | Custom brute (s) | Custom KD-Tree (s) | KD-Tree + bin seeding (s) | sklearn off (s) | sklearn on (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for alpha in alphas:
        h = float(alpha) * h0
        vals = []
        for method in METHODS:
            r = _summary_value(summary_rows, "h", method, n=int(h_sweep["n"]), alpha=float(alpha))
            vals.append(f"{float(r['median_time_s']):.4g}" if r else "—")
        lines.append(f"| {alpha:g} | {h:.5g} | " + " | ".join(vals) + " |")

    lines.extend(["", "## 4. Trả lời các ý của RQ3"])
    n_rows = [r for r in summary_rows if r["experiment"] == "n" and r["status"] == "ok"]
    first_kd_win = None
    for n in sorted({int(float(r["n"])) for r in n_rows}):
        brute = _summary_value(summary_rows, "n", "custom_brute", n=n)
        kd = _summary_value(summary_rows, "n", "custom_kdtree", n=n)
        if brute and kd and float(kd["median_time_s"]) < float(brute["median_time_s"]):
            first_kd_win = n
            break
    if first_kd_win is not None:
        lines.append(f"- **Ngưỡng quan sát KD-Tree nhanh hơn brute:** trong các n đã chạy, lần đầu ghi nhận tại n={first_kd_win}. Đây là kết quả thực nghiệm của môi trường đã ghi ở trên, không phải ngưỡng phổ quát.")
    else:
        lines.append("- **KD-Tree so với brute:** không ghi nhận KD-Tree nhanh hơn brute ở các kích thước n có đủ dữ liệu hợp lệ, hoặc một trong hai biến thể chưa có số liệu để kết luận.")

    slopes = []
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "n" and r["method"] == method and r["status"] == "ok"
                and _valid_positive(r.get("n")) and _valid_positive(r.get("median_time_s"))]
        if len(rows) >= 2:
            slope = float(np.polyfit(np.log10([float(r["n"]) for r in rows]),
                                     np.log10([float(r["median_time_s"]) for r in rows]), 1)[0])
            slopes.append((method, slope, len(rows)))
    if slopes:
        lines.append("- **Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian) theo log10(n); chỉ tính trên các điểm hợp lệ):")
        for method, slope, count in slopes:
            lines.append(f"  - {METHOD_LABELS[method]}: {slope:.3f} (dựa trên {count} mức n).")
    else:
        lines.append("- **Độ dốc log–log:** chưa đủ ít nhất hai điểm dữ liệu hợp lệ cho từng biến thể để ước lượng.")

    bin_speedups = []
    for n in sorted({int(float(r["n"])) for r in n_rows}):
        kd = _summary_value(summary_rows, "n", "custom_kdtree", n=n)
        kdb = _summary_value(summary_rows, "n", "custom_kdtree_bin", n=n)
        if kd and kdb and float(kdb["median_time_s"]) > 0:
            ratio = float(kd["median_time_s"]) / float(kdb["median_time_s"])
            bin_speedups.append((n, ratio, kd, kdb))
    if bin_speedups:
        best = max(bin_speedups, key=lambda x: x[1])
        lines.append(f"- **Bin seeding:** tại n={best[0]}, KD-Tree có bin seeding đạt tốc độ khoảng {best[1]:.3g}× so với KD-Tree không bin seeding (tỷ số thời gian median trên số liệu hợp lệ). Số seed trung vị ở hai cấu hình lần lượt là {best[3]['median_n_seeds']:.0f} và {best[2]['median_n_seeds']:.0f} nếu có số liệu.")
    else:
        lines.append("- **Bin seeding:** chưa đủ số liệu hợp lệ để định lượng mức tăng tốc so với KD-Tree không bin seeding.")

    h_rows = [r for r in summary_rows if r["experiment"] == "h" and r["status"] == "ok"]
    if len({float(r["alpha"]) for r in h_rows}) >= 2:
        lines.append("- **Ảnh hưởng của h:** tham chiếu bảng thời gian theo alpha ở trên và biểu đồ `e3_time_vs_h.png`. Diễn giải KD-Tree cần đối chiếu xu hướng thời gian với `mean_neighbors_per_query`; số láng giềng tăng theo h là cơ chế dự kiến trong thiết kế nghiên cứu, còn mức tăng cụ thể phụ thuộc vào dữ liệu và kết quả đo.")
    else:
        lines.append("- **Ảnh hưởng của h:** chưa đủ ít nhất hai mức alpha có kết quả hợp lệ để kết luận xu hướng.")

    lines.extend([
        "",
        "## 5. Biểu đồ",
        "- `../results/e3/e3_time_vs_n.png` — thời gian theo n, trục log–log.",
        "- `../results/e3/e3_memory_vs_n.png` — bộ nhớ đỉnh theo n.",
        "- `../results/e3/e3_time_vs_h.png` — thời gian theo h.",
        "",
        "## 6. Giới hạn và lưu ý diễn giải",
        "- `tracemalloc` ghi nhận bộ nhớ do allocator Python theo dõi; có thể không bao quát đầy đủ bộ nhớ native do NumPy, SciPy, KD-Tree hoặc thư viện sklearn cấp phát. Không diễn giải số đo này như RSS tổng của tiến trình.",
        "- Thời gian là thời gian `fit(X)` trong môi trường đã ghi. So sánh không tự động chứng minh một thuật toán luôn nhanh hơn trên mọi máy hoặc mọi dữ liệu.",
        "- Các lượt timeout/skip được ghi trong CSV; chúng không được dùng để tính median thời gian thành công.",
        "- Số láng giềng trung bình và số vòng lặp được ghi khi implementation/API cho phép instrument. Nếu giá trị trống, cần kiểm tra tương thích API hoặc báo cáo là không khả dụng, không tự suy ra giá trị.",
        "",
        "## 7. File đầu ra",
        "- `../results/e3/e3_runs.csv` — dữ liệu từng lượt chạy.",
        "- `../results/e3/e3_summary.csv` — median, Q1, Q3, IQR và hệ số tăng tốc so với custom brute.",
        "- `../results/e3/env.txt` và `../results/e3/e3_metadata.json` — môi trường và cấu hình chạy.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="E3 Mean-Shift runtime/memory benchmark")
    parser.add_argument("--config", default="configs/e3_runtime.yaml", help="Đường dẫn YAML tương đối repository root hoặc đường dẫn tuyệt đối")
    parser.add_argument("--repeats", type=int, default=None, help="Số lượt đo/cấu hình; mặc định theo YAML (3 theo phân công C)")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Timeout cho mỗi lần fit; mặc định theo YAML (600 giây)")
    parser.add_argument("--quick", action="store_true", help="Chạy kiểm tra rút gọn, không dùng làm kết quả báo cáo cuối")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(config_path)

    out_dir = Path(cfg.get("outputs", {}).get("directory", "results/e3"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.get("outputs", {}).get("report", "report/THUC_NGHIEM_E3.md"))
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    repeats_cfg = int(cfg.get("benchmark", {}).get("repeats", 3))
    repeats = int(args.repeats) if args.repeats is not None else repeats_cfg
    timeout = float(args.timeout_seconds if args.timeout_seconds is not None
                    else cfg.get("benchmark", {}).get("timeout_seconds", 600))
    if repeats < 1:
        parser.error("--repeats phải >= 1")
    if timeout <= 0:
        parser.error("--timeout-seconds phải > 0")

    n_values = [int(n) for n in cfg["n_sweep"]["values"]]
    h_n = int(cfg["h_sweep"]["n"])
    alphas = [float(a) for a in cfg["h_sweep"]["alphas"]]
    if args.quick:
        n_values = n_values[:2]
        h_n = min(h_n, max(n_values))
        alphas = alphas[:2]
        if args.repeats is None:
            repeats = 1
        timeout = min(timeout, 60.0)

    # Keep h fixed during the n sweep: estimate it once on a separate 2,000-point sample.
    from meanshift import estimate_bandwidth
    ds = cfg["dataset"]
    reference_n = int(ds.get("reference_n", 2000))
    X_ref, _ = make_blobs(
        n_samples=reference_n,
        centers=int(ds.get("centers", 5)),
        cluster_std=float(ds.get("cluster_std", 1.0)),
        random_state=int(ds.get("random_state", 0)),
        n_features=int(ds.get("dimensions", 2)),
    )
    h0 = float(estimate_bandwidth(
        X_ref,
        quantile=float(ds.get("bandwidth_quantile", 0.3)),
        n_samples=ds.get("bandwidth_n_samples", None),
        random_state=int(ds.get("random_state", 0)),
    ))
    if not math.isfinite(h0) or h0 <= 0:
        raise RuntimeError(f"estimate_bandwidth trả về h0 không hợp lệ: {h0}")

    print(f"Repository root: {ROOT}")
    print(f"Config: {config_path}")
    print(f"h0={h0:.8g}; n sweep={n_values}; h sweep n={h_n}, alpha={alphas}")
    print(f"Repeats={repeats}; timeout={timeout:g}s; output={out_dir}")
    env = write_environment(out_dir)

    raw_path = out_dir / "e3_runs.csv"
    all_rows: list[dict[str, Any]] = []
    with raw_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        all_rows.extend(_run_sweep_n(writer, f, cfg, h0, n_values, repeats, timeout))
        all_rows.extend(_run_sweep_h(writer, f, cfg, h0, h_n, alphas, repeats, timeout))

    summary_rows = summarize(all_rows)
    write_summary(out_dir / "e3_summary.csv", summary_rows)
    create_plots(summary_rows, out_dir)

    metadata = {
        "task": "E3 — runtime and memory",
        "config": cfg,
        "config_file": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "h0": h0,
        "repeats": repeats,
        "timeout_seconds": timeout,
        "quick_mode": bool(args.quick),
        "methods": METHODS,
        "method_labels": METHOD_LABELS,
        "environment": env,
        "raw_csv": str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
        "summary_csv": str((out_dir / "e3_summary.csv").relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir / "e3_summary.csv"),
        "note": "tracemalloc peak is Python-traced memory, not complete process RSS.",
    }
    (out_dir / "e3_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, cfg, h0, env, summary_rows, repeats, timeout)

    successes = sum(1 for r in all_rows if r.get("status") == "ok")
    timeouts = sum(1 for r in all_rows if r.get("status") == "timeout")
    errors = sum(1 for r in all_rows if r.get("status") == "error")
    print("\nE3 hoàn tất.")
    print(f"- Lượt chạy thành công: {successes}")
    print(f"- Timeout: {timeouts}; lỗi: {errors}")
    print(f"- Raw: {raw_path}")
    print(f"- Summary: {out_dir / 'e3_summary.csv'}")
    print(f"- Hình: {out_dir / 'e3_time_vs_n.png'}, {out_dir / 'e3_memory_vs_n.png'}, {out_dir / 'e3_time_vs_h.png'}")
    print(f"- Environment: {out_dir / 'env.txt'}")
    print(f"- Report: {report_path}")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
