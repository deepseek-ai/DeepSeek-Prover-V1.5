from .base import SamplingAlgorithmBase


class Sampling(SamplingAlgorithmBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_num = self.cfg.get('sample_num', 32)

    def sample(self, data, **kwargs):
        request_id_list = [
            self.scheduler.generator_submit_request(
                # add few-shot prompts
                self._preprocess_data(data),
            ) for _ in range(self.sample_num)
        ]
        for _idx, request_id in enumerate(request_id_list):
            outputs = self.scheduler.generator_get_request_outputs(request_id)
            yield outputs, self._post_sample_info(cost=_idx+1)
            if _idx + 1 < self.sample_num and (_idx + 1) % self.log_interval == 0:
                self.process_print('Progress: {} / {}'.format(
                    _idx + 1, self.sample_num
                ))
