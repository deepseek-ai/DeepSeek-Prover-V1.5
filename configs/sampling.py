from prover.utils import AttrDict
from prover.algorithms import Sampling


# dataset
data_path = 'datasets/minif2f.jsonl'
data_split = ['valid', 'test']
data_repeat = 1

# verifier
lean_max_concurrent_requests = 64
lean_memory_limit = 10
lean_timeout = 300

# model
batch_size = 32
model_path = 'deepseek-ai/DeepSeek-Prover-V1.5-RL'
model_args = AttrDict(
    mode='cot',  # `cot` or `non-cot`
    temperature=1,
    max_tokens=2048,
    top_p=0.95,
)

# algorithm
n_search_procs = 64
sampler = dict(
    algorithm=Sampling,
    sample_num=128,
    log_interval=32,
)
