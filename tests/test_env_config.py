import os
import unittest
import types
from unittest import mock


def _with_env(env: dict):
    """
    Tiny helper to set env vars for a test and restore afterwards.
    Use as: `with _with_env({...}): ...`
    """
    class _EnvCtx:
        def __enter__(self):
            self._old = {}
            for k, v in env.items():
                self._old[k] = os.environ.get(k)
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, old in self._old.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    return _EnvCtx()


class TestEnvConfig(unittest.TestCase):
    def test_env_int_parses(self):
        from env_config import env_int

        with _with_env({"N_GPUS": "8"}):
            self.assertEqual(env_int("N_GPUS"), 8)

    def test_env_int_default_when_unset(self):
        from env_config import env_int

        with _with_env({"N_GPUS": None}):
            self.assertEqual(env_int("N_GPUS", default=3), 3)

    def test_env_float_parses(self):
        from env_config import env_float

        with _with_env({"TEMPERATURE": "0.8"}):
            self.assertAlmostEqual(env_float("TEMPERATURE"), 0.8)

    def test_env_bool_parses_truthy(self):
        from env_config import env_bool

        with _with_env({"DEBUG": "true"}):
            self.assertTrue(env_bool("DEBUG"))

    def test_env_bool_parses_falsy(self):
        from env_config import env_bool

        with _with_env({"DEBUG": "0"}):
            self.assertFalse(env_bool("DEBUG"))

    def test_env_present_empty_string_is_false(self):
        from env_config import env_present

        with _with_env({"MODEL_PATH": ""}):
            self.assertFalse(env_present("MODEL_PATH"))

    def test_env_bool_invalid_exits_cleanly(self):
        from env_config import env_bool

        with _with_env({"DEBUG": "maybe"}):
            with self.assertRaises(SystemExit) as ctx:
                env_bool("DEBUG")
            self.assertIn("Invalid boolean environment variable DEBUG", str(ctx.exception))

    def test_load_env_works_when_dotenv_available(self):
        # Don't require python-dotenv to actually be installed in the test environment.
        import env_config

        fake_dotenv = types.SimpleNamespace(load_dotenv=lambda **kwargs: True)
        with mock.patch.dict("sys.modules", {"dotenv": fake_dotenv}):
            self.assertTrue(env_config.load_env())

    def test_empty_string_treated_as_unset(self):
        from env_config import env_str

        with _with_env({"MODEL_PATH": ""}):
            self.assertIsNone(env_str("MODEL_PATH"))
