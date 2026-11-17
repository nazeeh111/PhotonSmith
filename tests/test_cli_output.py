"""Result publication preserves evidence through collision, failure and retry."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from photonsmith import cli


class OutputTests(unittest.TestCase):
    def test_legal_name_max_destination_publishes_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            limit = os.pathconf(folder, 'PC_NAME_MAX') if hasattr(os, 'pathconf') else 255
            limit = limit if limit > 0 else 255
            target = Path(folder) / ('r' * (limit - 5) + '.json')
            # Establish that the complete destination is actually legal on this
            # filesystem before attributing any failure to the publisher.
            try:
                target.write_bytes(b'owned filename preflight')
            except OSError as exc:
                if exc.errno == errno.ENAMETOOLONG or getattr(exc, 'winerror', None) == 206:
                    self.skipTest('This filesystem does not admit a NAME_MAX destination at the temporary path')
                raise
            target.unlink()
            cli._publish_result(target, '{"complete": true}')
            self.assertEqual(json.loads(target.read_text()), {'complete': True})
            with self.assertRaises(FileExistsError):
                cli._publish_result(target, '{"replacement": true}')
            self.assertEqual(json.loads(target.read_text()), {'complete': True})
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_every_command_refuses_existing_output_before_computation(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'result.json'
            target.write_bytes(b'previous evidence\n')
            for action in ('simulate', 'validate', 'optimize', 'converge'):
                with self.subTest(action=action), patch.object(cli, 'public_simulation') as simulate, \
                        patch.object(cli, 'validate_design') as validate, patch.object(cli, 'optimize') as optimize, \
                        patch('photonsmith.convergence.spectral_study') as study:
                    with self.assertRaises(SystemExit) as caught:
                        cli.main([action, '--output', str(target)])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(target.read_bytes(), b'previous evidence\n')
                    for compute in (simulate, validate, optimize, study):
                        compute.assert_not_called()

    def test_partial_write_failure_leaves_no_result_and_retry_succeeds(self):
        original = cli.tempfile.NamedTemporaryFile
        class BrokenWriter:
            def __init__(self, *args, **kwargs):
                self.file = original(*args, **kwargs)
                self.name = self.file.name
            def __enter__(self):
                self.file.__enter__()
                return self
            def __exit__(self, *args):
                return self.file.__exit__(*args)
            def write(self, text):
                self.file.write(text[:8])
                raise OSError('synthetic partial write')
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'result.json'
            with patch.object(cli.tempfile, 'NamedTemporaryFile', BrokenWriter):
                with self.assertRaisesRegex(OSError, 'partial write'):
                    cli._publish_result(target, '{"complete": true}')
            self.assertEqual(list(Path(folder).iterdir()), [])
            cli._publish_result(target, '{"complete": true}')
            self.assertEqual(json.loads(target.read_text()), {'complete': True})
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_flush_or_link_failure_leaves_no_result_and_retry_succeeds(self):
        for operation in ('fsync', 'link'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'result.json'
                with patch.object(cli.os, operation, side_effect=OSError('synthetic publication refusal')):
                    with self.assertRaisesRegex(OSError, 'publication refusal'):
                        cli._publish_result(target, '{"complete": true}')
                self.assertEqual(list(Path(folder).iterdir()), [])
                cli._publish_result(target, '{"complete": true}')
                self.assertEqual(json.loads(target.read_text()), {'complete': True})

    def test_competing_writers_publish_exactly_one_complete_result(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'result.json'
            gate = threading.Barrier(8)
            def writer(index):
                text = json.dumps({'writer': index, 'payload': str(index) * 10000})
                gate.wait(timeout=5)
                try:
                    cli._publish_result(target, text)
                    return index
                except FileExistsError:
                    return None
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(writer, range(8)))
            winners = [x for x in outcomes if x is not None]
            self.assertEqual(len(winners), 1)
            result = json.loads(target.read_text())
            self.assertEqual(result, {'writer': winners[0], 'payload': str(winners[0]) * 10000})
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_existing_symlink_is_not_followed_or_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / 'prior.json'
            reference.write_bytes(b'previous evidence')
            target = root / 'output.json'
            target.symlink_to(reference)
            with self.assertRaises(FileExistsError):
                cli._publish_result(target, 'replacement')
            self.assertTrue(target.is_symlink())
            self.assertEqual(reference.read_bytes(), b'previous evidence')
            reference.unlink()
            with self.assertRaises(SystemExit) as caught:
                cli.main(['simulate', '--output', str(target)])
            self.assertEqual(caught.exception.code, 2)
            self.assertTrue(target.is_symlink())
            self.assertFalse(reference.exists())

    def test_invalid_serialization_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'config.json'
            config.write_text('{}')
            target = root / 'new' / 'output.json'
            with patch.object(cli, 'public_simulation', return_value={'power': float('nan')}):
                with self.assertRaises(SystemExit) as caught:
                    cli.main(['simulate', '--config', str(config), '--output', str(target)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(target.exists())
            self.assertFalse(target.parent.exists())


if __name__ == '__main__':
    unittest.main()
