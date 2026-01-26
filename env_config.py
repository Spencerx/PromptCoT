from __future__ import annotations

import os
from typing import Optional, Mapping

__all__ = [
    "env_bool",
    "env_float",
    "env_int",
    "env_present",
    "env_str",
    "load_env",
]


def load_env(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """
    Load environment variables from a local .env file (if python-dotenv is installed).

    Precedence is preserved naturally:
    - CLI args override .env because argparse uses explicit argv values
    - .env overrides code defaults by supplying argparse defaults
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    return bool(load_dotenv(dotenv_path=dotenv_path, override=override))


def _raw_env(name: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = environ if environ is not None else os.environ
    val = env.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def env_str(name: str, default: Optional[str] = None, *, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    return _raw_env(name, environ=environ) or default


def env_int(name: str, default: Optional[int] = None, *, environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    raw = _raw_env(name, environ=environ)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        # This is usually used for argparse defaults; fail fast without a noisy traceback.
        raise SystemExit(f"Invalid integer environment variable {name}={raw!r}")


def env_float(name: str, default: Optional[float] = None, *, environ: Optional[Mapping[str, str]] = None) -> Optional[float]:
    raw = _raw_env(name, environ=environ)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"Invalid float environment variable {name}={raw!r}")


def env_bool(name: str, default: Optional[bool] = None, *, environ: Optional[Mapping[str, str]] = None) -> Optional[bool]:
    raw = _raw_env(name, environ=environ)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise SystemExit(
        f"Invalid boolean environment variable {name}={raw!r} (expected one of: "
        "1/0, true/false, yes/no, on/off)"
    )


def env_present(name: str, *, environ: Optional[Mapping[str, str]] = None) -> bool:
    return _raw_env(name, environ=environ) is not None
