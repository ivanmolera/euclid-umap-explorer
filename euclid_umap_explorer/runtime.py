from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import tempfile

from .config import APP_TITLE, APP_VERSION

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("euclid_umap_explorer")


def log_app_event(event_type: str, **fields: object) -> None:
    payload = {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "event_type": event_type,
        **fields,
    }
    LOGGER.info(json.dumps(payload, default=str, sort_keys=True))


class AlgorithmTimeoutError(TimeoutError):
    pass


def _run_in_process_worker(
    result_queue: mp.Queue,
    result_path: str,
    function,
    args: tuple,
    kwargs: dict,
) -> None:
    try:
        result = function(*args, **kwargs)
        with open(result_path, "wb") as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
        result_queue.put(("ok", None))
    except BaseException as exc:
        result_queue.put(("error", exc))


def run_with_timeout(function, *args, timeout_seconds: int, **kwargs):
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    result_queue = ctx.Queue(maxsize=1)
    result_file = tempfile.NamedTemporaryFile(
        prefix="euclid_algorithm_result_",
        suffix=".pkl",
        delete=False,
    )
    result_path = result_file.name
    result_file.close()
    process = ctx.Process(
        target=_run_in_process_worker,
        args=(result_queue, result_path, function, args, kwargs),
    )
    try:
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            raise AlgorithmTimeoutError(
                f"{function.__name__} exceeded {format_duration(timeout_seconds)}."
            )

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                f"{function.__name__} finished without returning a result."
            ) from exc

        if status == "error":
            raise payload

        with open(result_path, "rb") as file:
            return pickle.load(file)
    finally:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"

    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {remaining_seconds:.0f} s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(remaining_minutes)} min"
