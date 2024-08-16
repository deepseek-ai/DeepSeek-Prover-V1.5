from prover.utils import AttrDict
from prover.algorithms import RMaxTS


# dataset
data_path = 'datasets/minif2f.jsonl'
data_split = 'test'
data_repeat = 16  # run 16 * 6400

# verifier
lean_max_concurrent_requests = 64
lean_memory_limit = 10
lean_timeout = 300

# model
batch_size = 512
model_path = 'deepseek-ai/DeepSeek-Prover-V1.5-RL'
model_args = AttrDict(
    mode='cot', # `cot` or `non-cot`
    temperature=1,
    max_tokens=2048,
    top_p=0.95,
)

# algorithm
n_search_procs = 256
sampler = dict(
    algorithm=RMaxTS,
    gamma=0.99,
    sample_num=6400,
    concurrent_num=32,
    tactic_state_comment=True,
    ckpt_interval=128,
    log_interval=32,
)
