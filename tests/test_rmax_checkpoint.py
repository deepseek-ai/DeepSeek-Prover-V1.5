"""CPU-only checkpoint tests; the Lean root analysis is mocked."""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prover.algorithms.rmax_tree_search import RMaxTS, TreeNode


class RMaxTSCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.primary = Path(directory.name) / 'checkpoint.pkl'
        self.backup = Path(str(self.primary) + '.backup')
        # Checkpoint recovery does not require loading a model or tokenizer.
        self.algorithm = RMaxTS.__new__(RMaxTS)
        self.algorithm.ckpt_path = str(self.primary)
        self.algorithm.node_cls = TreeNode
        self.algorithm.scheduler = Mock()
        self.algorithm.process_print = Mock()

    def checkpoint(self, count):
        root = TreeNode(code={'tactic_code': '', 'state_comment': ''}, depth=0)
        child = TreeNode(
            parent=root,
            code={'tactic_code': f'-- sample {count}', 'state_comment': ''},
            depth=1,
        )
        root.children[f'goal-{count}'] = child
        child.update_reward(1, gamma=0.99)
        return {
            'root': root.to_dict(),
            'sample_count': count,
            'yield_cache': [(f'proof-{count}', {'cost': count})],
        }

    def write_checkpoint(self, path, checkpoint):
        with path.open('wb') as stream:
            pickle.dump(checkpoint, stream)

    def load_tree(self):
        data = {'formal_statement': 'theorem example_goal : True := by\n'}
        with patch('prover.algorithms.rmax_tree_search.ProofSummarizer') as summarizer:
            summarizer.return_value.analyze.return_value = SimpleNamespace(
                result={'pass': True, 'sorries': [{'goal': '⊢ True'}]},
            )
            result = self.algorithm._tree_setup(data)
        summarizer.assert_called_once_with(data=data, scheduler=self.algorithm.scheduler)
        summarizer.return_value.analyze.assert_called_once_with(
            '  sorry', require_verification=True,
        )
        self.assertEqual(self.algorithm.root_goal, '⊢ True')
        self.assertIs(self.algorithm._last_selected_node, result[0])
        return result

    def assert_restored(self, expected):
        root, count, cache = self.load_tree()
        self.assertEqual(count, expected['sample_count'])
        self.assertEqual(cache, expected['yield_cache'])
        self.assertEqual(root.to_dict(), expected['root'])
        for child in root.children.values():
            self.assertIs(child.parent, root)

    def assert_fresh(self):
        root, count, cache = self.load_tree()
        self.assertEqual(count, 0)
        self.assertEqual(cache, [])
        self.assertEqual(root['depth'], 0)
        self.assertEqual(root.children, {})
        self.assertEqual(root.code, {'tactic_code': '', 'state_comment': ''})

    def test_primary_takes_precedence_over_valid_backup(self):
        primary = self.checkpoint(256)
        self.write_checkpoint(self.primary, primary)
        self.write_checkpoint(self.backup, self.checkpoint(128))
        self.assert_restored(primary)

    def test_valid_primary_does_not_read_corrupt_backup(self):
        primary = self.checkpoint(256)
        self.write_checkpoint(self.primary, primary)
        self.backup.write_bytes(b'not a pickle')
        with patch('pickle.load', wraps=pickle.load) as load:
            self.assert_restored(primary)
        self.assertEqual(load.call_count, 1)
        self.algorithm.process_print.assert_called_once_with(
            'Load checkpoint from sample_count=256',
        )

    def test_valid_primary_without_backup(self):
        primary = self.checkpoint(256)
        self.write_checkpoint(self.primary, primary)
        self.assert_restored(primary)

    def test_missing_primary_uses_backup(self):
        backup = self.checkpoint(128)
        self.write_checkpoint(self.backup, backup)
        self.assert_restored(backup)

    def test_corrupt_primary_uses_backup(self):
        backup = self.checkpoint(128)
        self.primary.write_bytes(b'not a pickle')
        self.write_checkpoint(self.backup, backup)
        self.assert_restored(backup)
        self.algorithm.process_print.assert_any_call(
            f'Checkpoint saved at {self.primary} is broken.',
        )

    def test_none_primary_still_uses_backup(self):
        backup = self.checkpoint(128)
        self.write_checkpoint(self.primary, None)
        self.write_checkpoint(self.backup, backup)
        self.assert_restored(backup)

    def assert_malformed_primary_uses_backup(self, primary):
        backup = self.checkpoint(128)
        self.write_checkpoint(self.primary, primary)
        self.write_checkpoint(self.backup, backup)
        self.assert_restored(backup)

    def test_primary_missing_root_uses_backup(self):
        primary = self.checkpoint(256)
        del primary['root']
        self.assert_malformed_primary_uses_backup(primary)

    def test_primary_missing_count_uses_backup(self):
        primary = self.checkpoint(256)
        del primary['sample_count']
        self.assert_malformed_primary_uses_backup(primary)

    def test_primary_missing_cache_uses_backup(self):
        primary = self.checkpoint(256)
        del primary['yield_cache']
        self.assert_malformed_primary_uses_backup(primary)

    def test_primary_with_invalid_subtree_uses_backup(self):
        primary = self.checkpoint(256)
        child = primary['root']['children']['goal-256']
        del child['children']
        self.assert_malformed_primary_uses_backup(primary)

    def test_missing_checkpoints_start_fresh(self):
        self.assert_fresh()

    def test_corrupt_primary_without_backup_starts_fresh(self):
        self.primary.write_bytes(b'not a pickle')
        self.assert_fresh()

    def test_both_corrupt_start_fresh(self):
        self.primary.write_bytes(b'not a pickle')
        self.backup.write_bytes(b'not a pickle either')
        self.assert_fresh()

    def test_save_then_resume_keeps_latest_tree_count_and_cache(self):
        previous = self.checkpoint(128)
        latest = self.checkpoint(256)
        self.algorithm._save_ckpt(previous)
        self.algorithm._save_ckpt(latest)
        with self.backup.open('rb') as stream:
            self.assertEqual(pickle.load(stream), previous)
        with self.primary.open('rb') as stream:
            self.assertEqual(pickle.load(stream), latest)
        self.assert_restored(latest)


if __name__ == '__main__':
    unittest.main()
