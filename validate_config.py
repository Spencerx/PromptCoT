from __future__ import annotations

import os
import sys
from typing import Mapping, List, Optional
from urllib.parse import urlparse

from env_config import load_env


def _is_probably_path(value: str) -> bool:
    # Heuristic: treat obvious filesystem-y strings as paths; otherwise assume a remote model id.
    return (
        value.startswith((".", "/", "~"))
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    )


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _get(environ: Mapping[str, str], key: str) -> Optional[str]:
    raw = environ.get(key)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def validate_environ(environ: Mapping[str, str]) -> List[str]:
    """
    Validate configuration values from an environment mapping.

    Returns a list of human-readable error strings. Empty list means OK.
    """
    errors: List[str] = []

    # Required keys
    for key in ("MODEL_PATH", "N_GPUS", "DATA_DIR", "OUTPUT_DIR"):
        if _get(environ, key) is None:
            errors.append(f"{key} is required but not set")

    # N_GPUS
    n_gpus_raw = _get(environ, "N_GPUS")
    if n_gpus_raw is not None:
        try:
            n_gpus = int(n_gpus_raw)
            if n_gpus <= 0:
                errors.append("N_GPUS must be an integer > 0")
        except ValueError:
            errors.append("N_GPUS must be an integer > 0")

    # MODEL_PATH (path or model id)
    model_path = _get(environ, "MODEL_PATH")
    if model_path is not None and _is_probably_path(model_path):
        mp = _expand_path(model_path)
        if not os.path.exists(mp):
            errors.append(f"MODEL_PATH does not exist: {mp}")
        elif not os.path.isdir(mp):
            errors.append(f"MODEL_PATH must be a directory: {mp}")
        elif not os.access(mp, os.R_OK):
            errors.append(f"MODEL_PATH is not readable: {mp}")

    # TOKENIZER_PATH (optional; path or model id)
    tokenizer_path = _get(environ, "TOKENIZER_PATH")
    if tokenizer_path is not None and _is_probably_path(tokenizer_path):
        tp = _expand_path(tokenizer_path)
        if not os.path.exists(tp):
            errors.append(f"TOKENIZER_PATH does not exist: {tp}")
        elif not os.path.isdir(tp):
            errors.append(f"TOKENIZER_PATH must be a directory: {tp}")
        elif not os.access(tp, os.R_OK):
            errors.append(f"TOKENIZER_PATH is not readable: {tp}")

    # DATA_DIR / OUTPUT_DIR (directories)
    for key in ("DATA_DIR", "OUTPUT_DIR"):
        val = _get(environ, key)
        if val is None:
            continue
        path = _expand_path(val)
        if not os.path.exists(path):
            errors.append(f"{key} does not exist: {path}")
            continue
        if not os.path.isdir(path):
            errors.append(f"{key} must be a directory: {path}")
            continue
        if not os.access(path, os.R_OK):
            errors.append(f"{key} is not readable: {path}")
            continue
        if key == "OUTPUT_DIR" and not os.access(path, os.W_OK):
            errors.append(f"{key} is not writable: {path}")

    # Script-specific paths (optional). Prefer namespaced keys; validate both to catch mistakes early.
    data_path_keys = [
        "DATA_PATH",
        "SPLIT_MERGE_DATA_PATH",
        "SELF_PLAY_DATA_PATH",
        "TEST_CASES_GEN_DATA_PATH",
        "SELF_PLAY_EVAL_DATA_PATH",
        "PREPARE_SELF_PLAY_DATA_DATA_PATH",
        "PREPARE_SFT_DATA_CODE_DATA_PATH",
        "TEST_CASES_POSTPROCESS_INPUT_FILE",
    ]
    for key in data_path_keys:
        data_path = _get(environ, key)
        if data_path is None:
            continue
        dp = _expand_path(data_path)
        if not os.path.exists(dp):
            errors.append(f"{key} does not exist: {dp}")
        elif not os.path.isfile(dp):
            errors.append(f"{key} must be a file: {dp}")
        elif not os.access(dp, os.R_OK):
            errors.append(f"{key} is not readable: {dp}")

    output_path_keys = [
        "OUTPUT_PATH",
        "SPLIT_MERGE_OUTPUT_PATH",
        "SELF_PLAY_OUTPUT_PATH",
        "TEST_CASES_GEN_OUTPUT_PATH",
        "SELF_PLAY_EVAL_OUTPUT_PATH",
        "PREPARE_SELF_PLAY_DATA_OUTPUT_PATH",
        "PREPARE_SFT_DATA_CODE_OUTPUT_PATH",
        "TEST_CASES_POSTPROCESS_OUTPUT_FILE",
    ]
    for key in output_path_keys:
        output_path = _get(environ, key)
        if output_path is None:
            continue
        op = _expand_path(output_path)
        parent = os.path.dirname(op) or "."
        parent = _expand_path(parent)
        if not os.path.exists(parent):
            errors.append(f"{key} parent directory does not exist: {parent}")
        elif not os.path.isdir(parent):
            errors.append(f"{key} parent is not a directory: {parent}")
        elif not os.access(parent, os.W_OK):
            errors.append(f"{key} parent directory is not writable: {parent}")

    # Optional DB connection validation
    db_uri = _get(environ, "DB_URI")
    if db_uri is not None:
        parsed = urlparse(db_uri)
        if parsed.scheme == "sqlite":
            # sqlite:////abs/path.db or sqlite:///rel/path.db
            sqlite_path = parsed.path or ""
            if db_uri.startswith("sqlite:////"):
                # urlparse keeps the extra leading slash in path (e.g. '//var/db.sqlite3');
                # if we feed that into sqlite3's file: URI it becomes 'file://var/..' (authority='var').
                while sqlite_path.startswith("//"):
                    sqlite_path = sqlite_path[1:]
            else:
                # sqlite:///relative.db -> parsed.path='/relative.db' (strip to make it relative)
                sqlite_path = sqlite_path.lstrip("/")

            sqlite_path = _expand_path(sqlite_path)
            if not os.path.exists(sqlite_path):
                errors.append(f"DB_URI sqlite file does not exist: {sqlite_path}")
            else:
                try:
                    import sqlite3

                    # Read-only open to avoid creating files during validation.
                    sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True).close()
                except Exception as e:
                    errors.append(f"DB_URI sqlite connection failed: {e}")
        else:
            if not parsed.scheme:
                errors.append("DB_URI must include a scheme, e.g. sqlite:///path/to.db")
            elif not parsed.netloc and parsed.scheme not in {"sqlite"}:
                errors.append("DB_URI must include a host, e.g. postgresql://user:pass@host:5432/db")

    return errors


def main() -> int:
    load_env()
    errors = validate_environ(os.environ)
    if errors:
        print("Configuration invalid:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Configuration OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
