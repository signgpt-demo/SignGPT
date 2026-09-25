import string
from typing import List

import jiwer
import torch
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce
from torchmetrics import Metric

from .utils import bleu, rouge


class M2TMetrics(Metric):
    def __init__(
        self,
        cfg,
        w_vectorizer,
        dataname="humanml3d",
        top_k=3,
        bleu_k=4,
        R_size=32,
        max_text_len=40,
        diversity_times=300,
        dist_sync_on_step=True,
        unit_length=4,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.cfg = cfg
        self.dataname = dataname
        self.w_vectorizer = w_vectorizer
        self.name = "sign-to-text metrics"
        self.max_text_len = max_text_len
        self.top_k = top_k
        self.bleu_k = bleu_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.unit_length = unit_length
        self.level = "word"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.metrics = []

        for name in ("Matching_score", "gt_Matching_score"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append(name)
        for prefix in ("R_precision_top", "gt_R_precision_top"):
            for k in range(1, top_k + 1):
                name = f"{prefix}_{k}"
                self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
                self.metrics.append(name)

        for name in ("WER", "ROUGE_L", "Bert_F1"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append(name)
        for k in range(1, bleu_k + 1):
            name = f"Bleu_{k}"
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append(name)

        self.pred_texts = []
        self.gt_texts = []

    @torch.no_grad()
    def compute(self, sanity_flag):
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if sanity_flag:
            return metrics

        bleu_scores = bleu(self.gt_texts, self.pred_texts, level=self.level)
        for k in range(1, self.bleu_k + 1):
            value = torch.tensor(bleu_scores[f"bleu{k}"], device=self.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                all_reduce(value, op=ReduceOp.AVG)
            metrics[f"Bleu_{k}"] = value

        rouge_value = torch.tensor(
            rouge(self.gt_texts, self.pred_texts, level=self.level),
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_reduce(rouge_value, op=ReduceOp.AVG)
        metrics["ROUGE_L"] = rouge_value
        metrics["WER"] = torch.tensor(
            self._compute_wer(self.pred_texts, self.gt_texts), device=self.device
        )

        self.reset()
        self.gt_texts = []
        self.pred_texts = []
        return metrics

    @torch.no_grad()
    def update(
        self,
        feats_ref: Tensor,
        pred_texts: List[str],
        gt_texts: List[str],
        lengths: List[int],
        word_embs: Tensor = None,
        pos_ohot: Tensor = None,
        text_lengths: Tensor = None,
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def _compute_wer(self, predictions, references):
        if not predictions or not references:
            return 1.0
        predictions = [self._normalize_text(item) for item in predictions]
        references = [self._normalize_text(item) for item in references]
        return jiwer.wer(references, predictions)
