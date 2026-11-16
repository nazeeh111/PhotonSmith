"""Backend selection remains explicit at the CLI boundary."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from photonsmith import cli


class BackendCliTests(TestCase):
    def test_optimize_passes_selected_backend_and_defaults_to_legacy(self):
        for choice, expected in [(None, "legacy"), ("stable", "stable")]:
            with self.subTest(choice=choice), TemporaryDirectory() as directory:
                output = Path(directory) / "result.json"
                args = ["optimize", "--method", "random", "--output", str(output)]
                if choice:
                    args += ["--backend", choice]
                with mock.patch.object(cli, "optimize", return_value={"backend": expected}) as run:
                    cli._run(args)
                self.assertEqual(run.call_args.kwargs["backend"], expected)
                self.assertEqual(json.loads(output.read_text())["backend"], expected)

    def test_field_workflows_reject_backend_option(self):
        for action in ("simulate", "validate"):
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli._run([action, "--backend", "stable", "--output", "unused.json"])
                self.assertEqual(raised.exception.code, 2)
