import unittest
import tempfile
import os
import sqlite3


class TestValidateConfig(unittest.TestCase):
    def test_missing_required(self):
        from validate_config import validate_environ

        errors = validate_environ({})
        self.assertTrue(any("MODEL_PATH" in e for e in errors))
        self.assertTrue(any("N_GPUS" in e for e in errors))
        self.assertTrue(any("DATA_DIR" in e for e in errors))
        self.assertTrue(any("OUTPUT_DIR" in e for e in errors))

    def test_valid_minimal(self):
        from validate_config import validate_environ

        with tempfile.TemporaryDirectory() as td:
            env = {
                "MODEL_PATH": td,
                "DATA_DIR": td,
                "OUTPUT_DIR": td,
                "N_GPUS": "1",
            }
            self.assertEqual(validate_environ(env), [])

    def test_db_uri_sqlite_absolute_ok(self):
        from validate_config import validate_environ

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.sqlite3")
            sqlite3.connect(db_path).close()

            env = {
                "MODEL_PATH": td,
                "DATA_DIR": td,
                "OUTPUT_DIR": td,
                "N_GPUS": "1",
                # sqlite:////abs/path
                "DB_URI": f"sqlite:////{db_path.lstrip(os.sep)}",
            }
            self.assertEqual(validate_environ(env), [])

    def test_db_uri_sqlite_relative_ok(self):
        from validate_config import validate_environ

        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            try:
                os.chdir(td)
                db_path = "rel.sqlite3"
                sqlite3.connect(db_path).close()

                env = {
                    "MODEL_PATH": td,
                    "DATA_DIR": td,
                    "OUTPUT_DIR": td,
                    "N_GPUS": "1",
                    "DB_URI": f"sqlite:///{db_path}",
                }
                self.assertEqual(validate_environ(env), [])
            finally:
                os.chdir(cwd)

    def test_db_uri_missing_scheme_errors(self):
        from validate_config import validate_environ

        with tempfile.TemporaryDirectory() as td:
            env = {
                "MODEL_PATH": td,
                "DATA_DIR": td,
                "OUTPUT_DIR": td,
                "N_GPUS": "1",
                "DB_URI": "not-a-uri",
            }
            errors = validate_environ(env)
            self.assertTrue(any("DB_URI must include a scheme" in e for e in errors))

    def test_namespaced_output_path_parent_missing_errors(self):
        from validate_config import validate_environ

        with tempfile.TemporaryDirectory() as td:
            env = {
                "MODEL_PATH": td,
                "DATA_DIR": td,
                "OUTPUT_DIR": td,
                "N_GPUS": "1",
                "SPLIT_MERGE_OUTPUT_PATH": os.path.join(td, "does-not-exist", "out.jsonl"),
            }
            errors = validate_environ(env)
            self.assertTrue(any("SPLIT_MERGE_OUTPUT_PATH parent directory does not exist" in e for e in errors))
