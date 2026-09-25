from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric


class MMMetrics(Metric):
    """Compatibility metric for optional multi-sample generation evaluation."""

    full_state_update = True

    def __init__(
        self,
        cfg,
        dataname="humanml3d",
        mm_num_times=10,
        dist_sync_on_step=True,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "MultiModality scores"
        self.cfg = cfg
        self.dataname = dataname
        self.mm_num_times = mm_num_times
        self.metrics = ["MultiModality"]

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "MultiModality", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def compute(self, sanity_flag):
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if not sanity_flag:
            self.reset()
        return metrics

    def update(self, feats_rst: Tensor, lengths_rst: List[int]):
        self.count += sum(lengths_rst)
        self.count_seq += len(lengths_rst)
