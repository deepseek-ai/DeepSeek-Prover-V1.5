"""Verify REPL response handling without starting Lean or a model."""

import json
import subprocess
import unittest
from unittest.mock import patch

from prover.lean.verifier import verify_lean4_file


class LeanVerifierResponseTests(unittest.TestCase):
    code = 'example : True := by trivial'

    def verify(self, response, returncode=0, stderr='', **kwargs):
        output = subprocess.CompletedProcess(
            ['lake', 'exe', 'repl'], returncode,
            stdout=json.dumps(response), stderr=stderr,
        )
        with patch('prover.lean.verifier.subprocess.run', return_value=output):
            return verify_lean4_file(self.code, **kwargs)

    def assert_system_failure(self, result):
        self.assertFalse(result['pass'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['system_errors'])

    def test_unknown_environment_is_not_a_completed_proof(self):
        # The pinned REPL emits Error.message, not CommandResponse.messages.
        result = self.verify({'message': 'Unknown environment.'}, last_env=0)
        self.assert_system_failure(result)
        self.assertIn('Unknown environment.', result['system_errors'])

    def test_command_exception_is_not_a_completed_proof(self):
        result = self.verify({'message': 'failed to process input'})
        self.assert_system_failure(result)
        self.assertIn('failed to process input', result['system_errors'])

    def test_error_takes_precedence_over_environment_field(self):
        self.assert_system_failure(self.verify({'env': 0, 'message': 'error'}))

    def test_empty_object_is_not_a_command_response(self):
        self.assert_system_failure(self.verify({}))

    def test_proof_step_response_is_not_a_command_response(self):
        self.assert_system_failure(self.verify({'proofState': 0, 'goals': []}))

    def test_environment_must_be_a_natural_number(self):
        for env in (None, True, False, -1, '0', 0.0, [], {}):
            with self.subTest(env=env):
                self.assert_system_failure(self.verify({'env': env}))

    def test_nonobject_response_is_rejected(self):
        for response in (None, [], 'text', 0):
            with self.subTest(response=response):
                self.assert_system_failure(self.verify(response))

    def test_nonzero_exit_is_not_success_even_with_command_json(self):
        result = self.verify({'env': 0}, returncode=1, stderr='REPL failed')
        self.assert_system_failure(result)
        self.assertIn('CalledProcessError', result['system_errors'])

    def test_signalled_process_is_not_success(self):
        self.assert_system_failure(self.verify({'env': 0}, returncode=-9))

    def test_minimal_clean_command_response_remains_successful(self):
        # CommandResponse omits empty lists; requiring messages would reject it.
        result = self.verify({'env': 0})
        self.assertTrue(result['pass'])
        self.assertTrue(result['complete'])
        self.assertIsNone(result['system_errors'])
        self.assertEqual(result['verified_code'], self.code)

    def test_empty_optional_lists_remain_successful(self):
        result = self.verify({'env': 1, 'messages': [], 'sorries': [], 'tactics': []})
        self.assertTrue(result['pass'])
        self.assertTrue(result['complete'])

    def test_lean_error_remains_a_verification_failure(self):
        message = {'severity': 'error', 'data': 'unsolved goals'}
        result = self.verify({'env': 0, 'messages': [message]})
        self.assertFalse(result['pass'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['errors'], [message])
        self.assertIsNone(result['system_errors'])

    def test_sorry_is_not_complete(self):
        result = self.verify({'env': 0, 'sorries': [{'goal': '⊢ True'}]})
        self.assertTrue(result['pass'])
        self.assertFalse(result['complete'])

    def test_sorry_or_failed_warning_is_not_complete(self):
        for data in ("declaration uses 'sorry'", 'tactic failed'):
            with self.subTest(data=data):
                result = self.verify({'env': 0, 'messages': [
                    {'severity': 'warning', 'data': data},
                ]})
                self.assertTrue(result['pass'])
                self.assertFalse(result['complete'])

    def test_info_and_ordinary_warning_remain_successful(self):
        result = self.verify({'env': 0, 'messages': [
            {'severity': 'info', 'data': 'True.intro : True'},
            {'severity': 'warning', 'data': 'unused variable'},
        ]})
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['infos']), 1)
        self.assertEqual(len(result['warnings']), 1)

    def test_stderr_alone_does_not_reject_clean_command(self):
        result = self.verify({'env': 0}, stderr='Cannot allocate memory')
        self.assertTrue(result['complete'])
        # Do not newly trigger the server's system_messages-based retry loop.
        self.assertEqual(result['system_messages'], '')

    def test_malformed_json_remains_a_system_failure(self):
        output = subprocess.CompletedProcess([], 0, stdout='{', stderr='diagnostic')
        with patch('prover.lean.verifier.subprocess.run', return_value=output):
            result = verify_lean4_file(self.code)
        self.assert_system_failure(result)

    def test_timeout_remains_a_system_failure(self):
        with patch('prover.lean.verifier.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('lake', 1)):
            result = verify_lean4_file(self.code)
        self.assert_system_failure(result)


if __name__ == '__main__':
    unittest.main()
