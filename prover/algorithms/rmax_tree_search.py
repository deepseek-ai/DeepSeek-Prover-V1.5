import os
import gc
import time
import math
import random
import pickle
import subprocess

import numpy as np

from .base import SamplingAlgorithmBase
from prover.lean.proof import ProofSummarizer
from prover.utils import ConcurrentJob


class TreeNode(object):
    def __init__(self, parent=None, code=None, **kwargs):
        self.parent = parent
        self.children = dict()
        self._info = {key: val for key, val in kwargs.items()}
        if code is not None:
            self.update_code(code)
        
        if '_discounted_rewards' not in self._info:
            self._info['_discounted_rewards'] = 0.0
        if '_discounted_visitation' not in self._info:
            self._info['_discounted_visitation'] = 0.0
        if '_subtree_discounted_rewards' not in self._info:
            self._info['_subtree_discounted_rewards'] = 0.0
        if '_subtree_discounted_visitation' not in self._info:
            self._info['_subtree_discounted_visitation'] = 0.0
        self._num_running_jobs = 0
        self._subtree_num_running_jobs = 0
        self._update_value(gamma=0.0)  # gamma=0.0 is okay for initialization
    
    # basic tree supports
    @property
    def code(self):
        return random.choice(self._info['_code_list'])

    def update_code(self, code):
        if '_code_list' not in self._info:
            self._info['_code_list'] = []
        if code not in self._info['_code_list']:
            self._info['_code_list'].append(code)
    
    def __getitem__(self, key):
        return self._info[key]
    
    def to_node_list(self):
        return sum([child.to_node_list() for _, child in self.children.items()], start=[self])
    
    def to_dict(self):
        return dict(
            info=self._info,
            children={
                edge: child_node.to_dict()
                for edge, child_node in self.children.items()
            }
        )
    
    @classmethod
    def from_dict(cls, dict_data, parent=None):
        node = cls(
            parent=parent,
            **dict_data['info'],
        )
        node.children = {
            edge: cls.from_dict(child_dict, parent=node)
            for edge, child_dict in dict_data['children'].items()
        }
        return node
    
    # algorithm supports
    def update_reward(self, reward, gamma, first_node=True):
        if first_node:
            self._info['_discounted_rewards'] = self._info['_discounted_rewards'] * gamma + reward
            self._info['_discounted_visitation'] = self._info['_discounted_visitation'] * gamma + 1.0
        self._info['_subtree_discounted_rewards'] = self._info['_subtree_discounted_rewards'] * gamma + reward
        self._info['_subtree_discounted_visitation'] = self._info['_subtree_discounted_visitation'] * gamma + 1.0
        self._update_value(gamma)
        if self.parent is not None:
            self.parent.update_reward(reward, gamma, first_node=False)
    
    def start_new_job(self, gamma, first_node=True):
        if first_node:
            self._num_running_jobs += 1
        self._subtree_num_running_jobs += 1
        self._update_value(gamma)
        if self.parent is not None:
            self.parent.start_new_job(gamma, first_node=False)
    
    def complete_job(self, gamma, first_node=True):
        if first_node:
            self._num_running_jobs -= 1
        self._subtree_num_running_jobs -= 1
        self._update_value(gamma)
        if self.parent is not None:
            self.parent.complete_job(gamma, first_node=False)
    
    def _update_value(self, gamma):
        discounted_rewards = self._info['_discounted_rewards'] * (gamma ** self._num_running_jobs)
        discounted_visitation = \
            self._info['_discounted_visitation'] * (gamma ** self._num_running_jobs) \
            + (1.0 - (gamma ** self._num_running_jobs)) / (1.0 - gamma)
        self.value = discounted_rewards / max(discounted_visitation, 1e-2)
        self.visitation = discounted_visitation

        subtree_discounted_rewards = self._info['_subtree_discounted_rewards'] * (gamma ** self._subtree_num_running_jobs)
        subtree_discounted_visitation = \
            self._info['_subtree_discounted_visitation'] * (gamma ** self._subtree_num_running_jobs) \
            + (1.0 - (gamma ** self._subtree_num_running_jobs)) / (1.0 - gamma)
        self.subtree_value = subtree_discounted_rewards / max(subtree_discounted_visitation, 1e-2)
        self.subtree_visitation = subtree_discounted_visitation


class RMaxTS(SamplingAlgorithmBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = self.cfg.get('gamma', 0.99)
        self.sample_num = self.cfg.get('sample_num', 6400)
        self.concurrent_num = self.cfg.get('concurrent_num', 32)
        self.tactic_state_comment = self.cfg.get('tactic_state_comment', True)
        self.ckpt_interval = self.cfg.get('ckpt_interval', 128)

        self.ckpt_filename = 'checkpoint.pkl'
        self.node_cls = TreeNode
        self.algorithm_pipeline = [
            self._tactic_tree_generate_proof,
            self._tactic_tree_parse_proof,
            self._rmax_exploration_summarize_results,
        ]
    
    # basic supports
    def _save_ckpt(self, ckpt_dict: dict):
        # save a backup before overwriting the checkpoint file
        if os.path.exists(self.ckpt_path):
            subprocess.run(['cp', self.ckpt_path, self.ckpt_path + '.backup'])
        # overwrite the checkpoint file
        with open(self.ckpt_path, 'wb') as pkl_f:
            pickle.dump(ckpt_dict, pkl_f)
    
    # tree structure supports
    def _tree_setup(self, data):
        # initialize tree
        ckpt_info = None
        for _ckpt_path in [self.ckpt_path, self.ckpt_path + '.backup']:
            if os.path.exists(_ckpt_path):
                try:
                    with open(_ckpt_path, 'rb') as pkl_f:
                        ckpt_info = pickle.load(pkl_f)
                except:
                    self.process_print(f'Checkpoint saved at {_ckpt_path} is broken.')
        if ckpt_info is not None:
            root = self.node_cls.from_dict(ckpt_info['root'])
            sample_count = ckpt_info['sample_count']
            yield_cache = ckpt_info['yield_cache']
            self.process_print(f'Load checkpoint from sample_count={sample_count}')
        else:
            root = self.node_cls(code=dict(tactic_code=str(), state_comment=str()), depth=0)
            sample_count = 0
            yield_cache = []
        
        # compile the root node with `sorry`
        self.proof_summarizer = ProofSummarizer(data=data, scheduler=self.scheduler)
        root_sorry = self.proof_summarizer.analyze('  sorry', require_verification=True)
        assert root_sorry.result['pass'], "Cannot parse a `sorry` tactic on root."
        self.root_goal = root_sorry.result['sorries'][-1]['goal']

        # other initialization
        self._last_selected_node = root
        
        return root, sample_count, yield_cache
    
    def _tree_new_child(self, parent):
        return self.node_cls(
            parent=parent,
            depth=parent['depth'] + 1,
        )
    
    def _tree_step(self, node, edge, code):
        if edge not in node.children:
            new_node = self._tree_new_child(node)
            node.children[edge] = new_node
            self.node_list.append(new_node)
        child_node = node.children[edge]
        child_node.update_code(code)
        return child_node
    
    def _tree_update(self, proof):
        node_walk, partial_code = self.root, str()

        # use tactic goals as tree edges
        segments = proof.segmentation()
        prev_goal = self.root_goal
        for info in segments:
            partial_code += info.tactic_code
            code = partial_code + info.state_comment if self.tactic_state_comment else partial_code
            if self._encode_length(code) < self.max_tokens:
                if info.goal != prev_goal:
                    node_walk = self._tree_step(
                        node=node_walk, edge=info.goal,
                        code=dict(tactic_code=partial_code, state_comment=info.state_comment)
                    )
                    prev_goal = info.goal
        return node_walk
    
    # algorithm pipeline
    def _select_node(self):
        node = self.root
        while len(node.children) > 0:
            num_choice = 1 + len(node.children)
            total_visitation = node.visitation + np.sum([child.subtree_visitation for _, child in node.children.items()])
            def hoeffding_ucb(node_visitation):
                return math.sqrt(2.0 * math.log(max(total_visitation, 2)) / max(node_visitation, 1e-2))
            choice_list = [(node.value + hoeffding_ucb(node.visitation), None)]
            for _, child in node.children.items():
                choice_list.append((child.subtree_value + hoeffding_ucb(child.subtree_visitation), child))
            choice_list.sort(reverse=True, key=lambda x: x[0])
            if choice_list[0][1] is None:
                node.start_new_job(gamma=self.gamma)
                return node
            else:
                node = choice_list[0][1]
        node.start_new_job(gamma=self.gamma)
        return node
    
    def _tactic_tree_generate_proof(self, data, node):
        code_prefix = node.code
        extra_prompt = code_prefix['tactic_code']
        if self.tactic_state_comment:
            extra_prompt += code_prefix['state_comment']
        return dict(
            node=node,
            code_prefix=code_prefix,
            generator_request_id=self.scheduler.generator_submit_request(
                self._preprocess_data({**data, '_extra_prompt': extra_prompt}),
            ),
        )
    
    def _tactic_tree_parse_proof(self, node, code_prefix, generator_request_id):
        code = self.scheduler.generator_get_request_status(generator_request_id)
        if code is None:
            return None
        code = code_prefix['tactic_code'] + code
        proof = self.proof_summarizer.analyze(code, require_verification=True)
        return dict(node=node, proof=proof)
    
    def _rmax_exploration_summarize_results(self, node, proof):
        if not proof.is_result_ready():
            return None
        
        num_nodes_before = len(self.node_list)
        self._tree_update(proof)
        # RMax reward
        node.update_reward(int(len(self.node_list) > num_nodes_before), gamma=self.gamma)
        node.complete_job(gamma=self.gamma)
        
        return dict(
            code=proof.cleaned_code,
            result=proof.result,
        )
    
    # sampler interface
    def sample(self, data, prob_log_dir, **kwargs):
        self.ckpt_path = os.path.join(prob_log_dir, self.ckpt_filename)
        self.root, sample_count, yield_cache = self._tree_setup(data)
        self.node_list = self.root.to_node_list()
        for _proposal, _sample_info in yield_cache:
            yield _proposal, _sample_info
        gc.collect()  # release memory

        job_slots = [
            ConcurrentJob(self.algorithm_pipeline)
            for _ in range(self.concurrent_num)
        ]

        sample_budget = self.sample_num - sample_count if len(yield_cache) == 0 else 0
        while (sample_budget > 0) or any([not job.is_idle() for job in job_slots]):
            for job in job_slots:
                if job.is_idle() and sample_budget > 0:
                    node = self._select_node()
                    self._last_selected_node = node
                    job.start(data=data, node=node)
                    sample_budget -= 1
                if not job.is_idle():
                    info = job.get_status()
                    if info is not None:
                        # output samples
                        sample_count += 1
                        if info['result']['complete']:
                            _proposal, _sample_info = info['code'], self._post_sample_info(
                                cost=sample_count, tree_size=len(self.node_list),
                            )
                            yield_cache.append((_proposal, _sample_info))
                            yield _proposal, _sample_info
                        # logging
                        if sample_count % self.log_interval == 0:
                            self.process_print('Progress: {} / {}    Tree Size: {}'.format(
                                sample_count, self.sample_num, len(self.node_list),
                            ))
                        # saving checkpoints
                        if sample_count % self.ckpt_interval == 0:
                            self._save_ckpt(dict(
                                root=self.root.to_dict(),
                                sample_count=sample_count,
                                yield_cache=yield_cache,
                            ))
                            if len(yield_cache) > 0:
                                # return after saving the checkpoint
                                # avoid overestimation caused by interrupt-restart loop
                                sample_budget = 0
            time.sleep(0.1)
        
        # save the final tree structure
        self._save_ckpt(dict(
            root=self.root.to_dict(),
            sample_count=sample_count,
            yield_cache=yield_cache,
        ))
