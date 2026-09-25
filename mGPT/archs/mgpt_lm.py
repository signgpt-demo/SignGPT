import os
from typing import List, Optional, Union

import heapq
import math
import numpy as np
import random
import re
import time

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput



class _LossOutput:
    """Tiny loss wrapper so the trainer can read .loss / .text_loss /
    .motion_loss regardless of whether the model is single-head or multi-head."""

    def __init__(self, loss, text_loss=None, motion_loss=None):
        self.loss = loss
        self.text_loss = text_loss if text_loss is not None else loss.detach()
        self.motion_loss = motion_loss if motion_loss is not None else loss.detach()


class Base_MLM(nn.Module):
    """Single-codebook, single-head, decoder-only motion-text LLM.

    The LLM's own embed_tokens and lm_head are extended in place:
        [0, old_vocab)                          : original text tokens
        [old_vocab, +motion_codebook_size)      : motion code ids
        old_vocab + motion_codebook_size        : <motion_start>
        old_vocab + motion_codebook_size + 1    : <motion_end>

    No external codebook / projection layers; codebook_params_path is unused.
    Forward and generation both go through the single lm_head, so text and
    motion are predicted by the same autoregressive objective over the
    extended vocabulary.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 256,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 256,
        lora: bool = False,
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        my_task: str = "mix",
        init_from_scratch: bool = False,
        scratch_hidden_size: int = 2048,
        scratch_intermediate_size: int = 8192,
        scratch_num_hidden_layers: int = 16,
        scratch_num_attention_heads: int = 32,
        scratch_num_key_value_heads: int = 8,
        scratch_max_position_embeddings: int = 131072,
        scratch_attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters kept for yaml backwards-compat (only some are actively used).
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.my_task = my_task
        self.new_token_type = new_token_type
        self.lora = lora

        # Tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add the two motion sentinel tokens.
        self.tokenizer.add_tokens(['<motion_start>', '<motion_end>'])
        self.motion_start_token_id = self.tokenizer.convert_tokens_to_ids('<motion_start>')
        self.motion_end_token_id = self.tokenizer.convert_tokens_to_ids('<motion_end>')

        # Build / load the LLM.
        if model_type == "llama":
            self.lm_type = 'dec_only'
            if init_from_scratch:
                # Final vocab size = old + motion_codebook_size + 2 sentinels.
                # add_tokens above already added the 2 sentinels, so the
                # placeholder config just needs to be at least that large.
                placeholder_vocab = len(self.tokenizer) + motion_codebook_size
                print(f"[Base_MLM] Initializing LLaMA from scratch: "
                      f"layers={scratch_num_hidden_layers}, "
                      f"hidden={scratch_hidden_size}, "
                      f"heads={scratch_num_attention_heads}")
                head_dim = scratch_hidden_size // scratch_num_attention_heads
                scratch_config = LlamaConfig(
                    vocab_size=placeholder_vocab,
                    hidden_size=scratch_hidden_size,
                    intermediate_size=scratch_intermediate_size,
                    num_hidden_layers=scratch_num_hidden_layers,
                    num_attention_heads=scratch_num_attention_heads,
                    num_key_value_heads=scratch_num_key_value_heads,
                    head_dim=head_dim,
                    max_position_embeddings=scratch_max_position_embeddings,
                    hidden_act="silu",
                    initializer_range=0.02,
                    rms_norm_eps=1e-05,
                    rope_theta=500000.0,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    model_type="llama",
                    attention_bias=False,
                    attention_dropout=scratch_attention_dropout,
                    mlp_bias=False,
                    tie_word_embeddings=False,
                    use_cache=True,
                )
                self.language_model = LlamaForCausalLM(scratch_config)
            else:
                print(f"[Base_MLM] Loading pretrained LLaMA from: {model_path}")
                self.language_model = LlamaForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float32,
                )
        else:
            raise ValueError(
                f"Base_MLM single-codebook variant only supports model_type='llama', got '{model_type}'"
            )

        # Extended vocab layout.
        self.old_vocab_size = len(self.tokenizer) - 2  # exclude the two sentinels
        self.motion_offset = self.old_vocab_size
        self.motion_start_idx = self.old_vocab_size + motion_codebook_size
        self.motion_end_idx = self.motion_start_idx + 1
        self.extended_vocab_size = self.motion_end_idx + 1

        # Resize embed_tokens / lm_head to cover the extended vocab.
        self.language_model.resize_token_embeddings(self.extended_vocab_size)
        assert self.language_model.lm_head.out_features == self.extended_vocab_size, (
            f"lm_head out_features={self.language_model.lm_head.out_features} "
            f"!= extended_vocab_size={self.extended_vocab_size}"
        )

        self.embedding_dim = self.language_model.config.hidden_size

        # LoRA.
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model
            print("[Base_MLM] Applying LoRA to the language model...")
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=(
                    ["q_proj", "v_proj"]
                    + ["k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    + ["embed_tokens", "lm_head"]
                ),
                lora_dropout=lora_dropout, bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            self.language_model.print_trainable_parameters()

    # Forward
    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict, istrain: bool = True, **kwargs):
        if self.lm_type == 'dec_only':
            return self.forward_dec_only(texts, motion_tokens, lengths, tasks)
        raise NotImplementedError(
            "Base_MLM single-codebook variant only supports model_type='llama' (dec_only)."
        )

    def forward_dec_only(self, texts: List[str], motion_tokens: Tensor,
                         lengths: List[int], tasks: dict):
        device = motion_tokens.device

        if self.my_task == 't2m':
            tasks = [{'input': ['<Caption_Placeholder>\n'],
                      'output': ['<Motion_Placeholder>']}] * len(lengths)
        elif self.my_task == 'm2t':
            tasks = [{'input': ['<Motion_Placeholder>\n'],
                      'output': ['<Caption_Placeholder>']}] * len(lengths)
        # else 'mix': keep caller-provided tasks.

        inputs, outputs = self.template_fulfill(tasks, lengths, texts, texts)
        combined_strings = [inp + out for inp, out in zip(inputs, outputs)]

        sample_tasks = []
        for inp in inputs:
            sample_tasks.append('t2m' if 'Caption_Placeholder' in inp else 'm2t')

        inputs_embeds, attention_mask = self._create_hybrid_inputs(
            string_list=combined_strings, motion_tokens=motion_tokens,
            text_list=texts, lengths=lengths, device=device,
            add_eos=True, sample_tasks=sample_tasks,
        )
        labels = self._create_hybrid_labels(
            string_list=combined_strings, motion_tokens=motion_tokens,
            text_list=texts, lengths=lengths, device=device,
            sample_tasks=sample_tasks,
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Manual shifting for causal LM.
        shift_hidden_states = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attention_mask = attention_mask[:, :-1].contiguous()

        active_mask = shift_attention_mask.view(-1) == 1
        active_hidden_states = shift_hidden_states.view(-1, self.embedding_dim)[active_mask]
        active_labels = shift_labels.view(-1)[active_mask]

        # Single lm_head predicts text or motion depending on which column
        # the label sits in.
        logits = self.language_model.lm_head(active_hidden_states)
        loss = F.cross_entropy(logits, active_labels, ignore_index=-100)
        loss = torch.where(torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss)

        # Single-head CE: text and motion share the same loss tensor.
        # `loss` keeps the autograd graph; only the motion_loss log slot is
        # detached so the trainer-side aggregation in mGPT/losses/mgpt.py
        # does not double-count the backward signal.
        out = _LossOutput(loss=loss, text_loss=loss, motion_loss=loss.detach())
        # Expose the wrapper so the trainer can split text vs motion on the
        # progress bar via trainer.callback_metrics.
        self.last_loss_output = out
        return out

    # Hybrid input / label builders (single codebook, single head).
    def _create_hybrid_inputs(self, string_list, motion_tokens, text_list, lengths,
                              device, add_eos=False, padding_side='right',
                              sample_tasks=None):
        batch_size = len(string_list)
        text_embedder = self.language_model.get_input_embeddings()
        placeholders = {'<Motion_Placeholder>', '<Caption_Placeholder>'}
        placeholder_pattern = re.compile(f"({'|'.join(placeholders)})")

        # Per-sample flat motion embeddings (lengths[i] is the motion length).
        motion_emb_per_sample = self._embed_motion_chunk(motion_tokens, lengths, device)
        final_embeds = []

        for i in range(batch_size):
            text = string_list[i]
            parts = [part for part in placeholder_pattern.split(text) if part]
            cur_task = sample_tasks[i] if sample_tasks else 't2m'
            chunks = [text_embedder(torch.tensor([self.tokenizer.bos_token_id], device=device))]

            for part in parts:
                if part not in placeholders:
                    token_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids.to(device).view(-1)
                    if token_ids.numel() > 0:
                        chunks.append(text_embedder(token_ids))
                elif part == '<Motion_Placeholder>':
                    chunks.append(text_embedder(torch.tensor([self.motion_start_idx], device=device)))
                    if motion_emb_per_sample[i].shape[0] > 0:
                        chunks.append(motion_emb_per_sample[i])
                    chunks.append(text_embedder(torch.tensor([self.motion_end_idx], device=device)))
                elif part == '<Caption_Placeholder>':
                    caption_ids = self.tokenizer(text_list[i], add_special_tokens=False, return_tensors="pt").input_ids.to(device).view(-1)
                    if caption_ids.numel() > 0:
                        chunks.append(text_embedder(caption_ids))

            if add_eos and cur_task == 'm2t':
                chunks.append(text_embedder(torch.tensor([self.tokenizer.eos_token_id], device=device)))

            final_embeds.append(torch.cat(chunks, dim=0))

        max_len_in_batch = max(e.shape[0] for e in final_embeds) if final_embeds else 0
        effective_max_len = min(max_len_in_batch, self.max_length)
        if effective_max_len == self.max_length and max_len_in_batch > self.max_length:
            print(f"[Base_MLM] Warning: max length {self.max_length} exceeded (actual {max_len_in_batch}). Truncating.")

        pad_embedding_vector = text_embedder(
            torch.tensor([self.tokenizer.pad_token_id], device=device)
        ).squeeze(0)
        padded_embeds = torch.full(
            (batch_size, effective_max_len, self.embedding_dim),
            fill_value=0.0, device=device,
        )
        padded_embeds[:] = pad_embedding_vector
        padded_masks = torch.zeros(batch_size, effective_max_len, dtype=torch.long, device=device)

        for i, emb in enumerate(final_embeds):
            len_to_copy = min(emb.shape[0], effective_max_len)
            if padding_side == 'right':
                padded_embeds[i, :len_to_copy] = emb[:len_to_copy]
                padded_masks[i, :len_to_copy] = 1
            else:  # padding_side == 'left'
                start_index = effective_max_len - len_to_copy
                padded_embeds[i, start_index:] = emb[:len_to_copy]
                padded_masks[i, start_index:] = 1

        return padded_embeds, padded_masks

    def _create_hybrid_labels(self, string_list, motion_tokens, text_list, lengths,
                             device, sample_tasks):
        batch_size = len(string_list)
        placeholders = {'<Motion_Placeholder>', '<Caption_Placeholder>'}
        placeholder_pattern = re.compile(f"({'|'.join(placeholders)})")

        motion_ids_per_sample = self._motion_to_extended_ids(motion_tokens, lengths, device)
        final_labels = []

        for i in range(batch_size):
            cur_task = sample_tasks[i]
            text = string_list[i]
            parts = [part for part in placeholder_pattern.split(text) if part]
            chunks = [torch.tensor([-100], device=device, dtype=torch.long)]  # BOS

            for part in parts:
                if part not in placeholders:
                    token_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids.to(device).view(-1)
                    if token_ids.numel() > 0:
                        chunks.append(torch.full((token_ids.shape[0],), -100, device=device, dtype=torch.long))
                elif part == '<Caption_Placeholder>':
                    caption_ids = self.tokenizer(text_list[i], add_special_tokens=False, return_tensors="pt").input_ids.to(device).view(-1)
                    if caption_ids.numel() > 0:
                        if cur_task == 'm2t':
                            chunks.append(caption_ids)
                        else:
                            chunks.append(torch.full((caption_ids.shape[0],), -100, device=device, dtype=torch.long))
                elif part == '<Motion_Placeholder>':
                    chunks.append(torch.tensor([-100], device=device, dtype=torch.long))  # <motion_start>
                    L = int(lengths[i])
                    if L > 0 and motion_ids_per_sample[i].shape[0] > 0:
                        if cur_task == 't2m':
                            chunks.append(motion_ids_per_sample[i])
                        else:
                            chunks.append(torch.full((motion_ids_per_sample[i].shape[0],), -100, device=device, dtype=torch.long))
                    # <motion_end>: train the LLM to predict it after the last motion code.
                    # Position L_last_motion_code hidden -> predicts motion_end_idx.
                    chunks.append(torch.tensor([self.motion_end_idx], device=device, dtype=torch.long))

            if cur_task == 'm2t':
                chunks.append(torch.tensor([self.tokenizer.eos_token_id], device=device, dtype=torch.long))

            final_labels.append(torch.cat(chunks, dim=0))

        max_len_in_batch = max(t.shape[0] for t in final_labels) if final_labels else 0
        effective_max_len = min(max_len_in_batch, self.max_length)

        padded_labels = torch.full(
            (batch_size, effective_max_len), fill_value=-100, device=device, dtype=torch.long,
        )
        for i, labs in enumerate(final_labels):
            len_to_copy = min(labs.shape[0], effective_max_len)
            padded_labels[i, :len_to_copy] = labs[:len_to_copy]

        return padded_labels

    def _iter_motion_samples(self, motion_tokens, lengths):
        """Yield ``(tokens_for_sample_i, L_i)`` regardless of whether
        ``motion_tokens`` is a stacked tensor or a Python list of tensors.

        When ``motion_tokens`` is None (e.g. t2m generation, where there is
        no source motion to embed), yield an empty tensor for each sample so
        downstream code can iterate uniformly.
        """
        if lengths is None:
            return
        length_iter = (
            lengths.tolist() if hasattr(lengths, 'tolist') else lengths
        )
        for i, L in enumerate(length_iter):
            L = int(L)
            if motion_tokens is None:
                yield torch.zeros(0, dtype=torch.long), L
            else:
                sample = motion_tokens[i] if isinstance(motion_tokens, list) \
                    else motion_tokens[i, :L]
                yield sample, L

    def _embed_motion_chunk(self, motion_tokens, lengths, device):
        """Embed each sample's motion tokens as a [L_i, D] tensor.

        motion_tokens are assumed to be a flat single-codebook sequence
        (dataloader is expected to collapse any per-codebook columns into
        one stream). Each id k is mapped to extended-vocab id
        ``motion_offset + k`` and looked up via the LLM's embed_tokens.
        """
        out = []
        for sample, L in self._iter_motion_samples(motion_tokens, lengths):
            if L == 0:
                out.append(torch.zeros(0, self.embedding_dim, device=device))
                continue
            toks = sample.long().to(device)[:L]
            ids = toks + self.motion_offset
            out.append(self.language_model.get_input_embeddings()(ids))
        return out

    def _motion_to_extended_ids(self, motion_tokens, lengths, device):
        """Map each sample's motion tokens to ids in the extended vocab."""
        out = []
        for sample, L in self._iter_motion_samples(motion_tokens, lengths):
            if L == 0:
                out.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            toks = sample.long().to(device)[:L]
            out.append(toks + self.motion_offset)
        return out

    # Generation
    def _sample_logits(self, logits, do_sample, temperature, top_k):
        if do_sample:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _generate_motion_loop(self, next_step_hidden_state, past_key_values,
                              max_new_tokens, do_sample, temperature, top_k):
        device = self.language_model.device
        batch_size = next_step_hidden_state.shape[0]
        generated_motion_lists = [[] for _ in range(batch_size)]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_hidden_state = next_step_hidden_state

        # State machine: 0 = expect motion_start, 1 = expect motion codes / motion_end
        expect_start = torch.ones(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if not unfinished_sequences.any():
                    break

                squeezed = current_hidden_state.squeeze(1)
                logits = self.language_model.lm_head(squeezed)

                # Mask logits so we can only sample motion tokens in the correct state:
                # - Before <motion_start>: only allow motion_start_idx
                # - After <motion_start>:  allow motion codes [motion_offset, motion_offset+m_codebook_size)
                #                            and motion_end_idx
                # All text vocab ids (and pad/bos/eos) are masked to -inf so the LLM
                # cannot accidentally emit them as "motion" tokens.
                masked_logits = logits.clone()
                # Mask everything by default
                masked_logits.fill_(-float('Inf'))
                for b in range(batch_size):
                    if not unfinished_sequences[b].item():
                        continue
                    if expect_start[b].item():
                        # Only allow motion_start_idx
                        masked_logits[b, self.motion_start_idx] = logits[b, self.motion_start_idx]
                    else:
                        # Allow motion codes and motion_end_idx
                        motion_lo = self.motion_offset
                        motion_hi = self.motion_offset + self.m_codebook_size
                        masked_logits[b, motion_lo:motion_hi] = logits[b, motion_lo:motion_hi]
                        masked_logits[b, self.motion_end_idx] = logits[b, self.motion_end_idx]

                tok = self._sample_logits(masked_logits, do_sample, temperature, top_k)
                tok_flat = tok.squeeze(1) if tok.dim() == 2 else tok

                is_end = (tok_flat == self.motion_end_idx)
                is_start = (tok_flat == self.motion_start_idx)
                just_finished = is_end & unfinished_sequences
                just_started = is_start & expect_start

                for i in range(batch_size):
                    if not unfinished_sequences[i].item() or just_finished[i].item():
                        continue
                    tok_id = int(tok_flat[i].item())
                    if just_started[i].item():
                        # <motion_start>: do not append, just transition state
                        continue
                    # Must be a motion code (masking guarantees this)
                    if self.motion_offset <= tok_id < self.motion_offset + self.m_codebook_size:
                        generated_motion_lists[i].append(tok_id - self.motion_offset)

                expect_start = expect_start & (~just_started)
                expect_start = expect_start & (~just_finished)
                unfinished_sequences = unfinished_sequences & (~just_finished)

                tok_for_embed = tok_flat
                if tok_for_embed.dim() == 1:
                    tok_for_embed = tok_for_embed.unsqueeze(1)
                next_embed = self.language_model.get_input_embeddings()(tok_for_embed)

                model_outputs = self.language_model(
                    inputs_embeds=next_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = model_outputs.past_key_values
                current_hidden_state = model_outputs.hidden_states[-1]

        final_motion_tensors = []
        for m_list in generated_motion_lists:
            if m_list:
                final_motion_tensors.append(torch.tensor(m_list, dtype=torch.long, device=device))
            else:
                final_motion_tensors.append(torch.empty(0, device=device, dtype=torch.long))
        return final_motion_tensors, [""] * batch_size

    def _generate_text_loop(self, next_step_hidden_state, past_key_values,
                            max_new_tokens, do_sample, temperature, top_k):
        device = self.language_model.device
        batch_size = next_step_hidden_state.shape[0]
        generated_text_ids = [[] for _ in range(batch_size)]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if not unfinished_sequences.any():
                    break

                squeezed = next_step_hidden_state.squeeze(1)
                text_logits = self.language_model.lm_head(squeezed)
                next_token_ids = self._sample_logits(text_logits, do_sample, temperature, top_k)
                if next_token_ids.dim() == 2 and next_token_ids.size(1) == 1:
                    next_token_ids = next_token_ids.squeeze(1)

                just_finished = (next_token_ids == self.tokenizer.eos_token_id)
                unfinished_sequences = unfinished_sequences & (~just_finished)

                for i in range(batch_size):
                    if unfinished_sequences[i].item():
                        generated_text_ids[i].append(int(next_token_ids[i].item()))

                if next_token_ids.dim() == 1:
                    next_token_ids = next_token_ids.unsqueeze(1)
                next_embed = self.language_model.get_input_embeddings()(next_token_ids)
                if next_embed.dim() == 2:
                    next_embed = next_embed.unsqueeze(1)

                model_outputs = self.language_model.model(
                    inputs_embeds=next_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = model_outputs.past_key_values
                next_step_hidden_state = model_outputs.hidden_states[-1]

        final_texts = self.tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)
        return None, final_texts

    def generate_direct(self, task, task_prompts, motion_tokens, motion_lengths,
                        caption_texts, max_new_tokens=128, do_sample=False,
                        temperature=1.0, top_k=50):
        self.language_model.eval()
        device = self.language_model.device

        inputs_embeds, attention_mask = self._create_hybrid_inputs(
            string_list=task_prompts, motion_tokens=motion_tokens,
            text_list=caption_texts, lengths=motion_lengths, device=device,
            add_eos=False, padding_side='left',
        )

        with torch.no_grad():
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            next_step_hidden_state = outputs.hidden_states[-1][:, -1:, :]

        if task == 't2m':
            return self._generate_motion_loop(
                next_step_hidden_state, past_key_values,
                max_new_tokens, do_sample, temperature, top_k,
            )
        elif task == 'm2t':
            return self._generate_text_loop(
                next_step_hidden_state, past_key_values,
                max_new_tokens, do_sample, temperature, top_k,
            )
        raise ValueError(f"Unsupported task: {task}")

    def generate_conditional(self, texts=None, motion_tokens=None, lengths=None,
                             task="t2m", with_len=False, stage='train', tasks=None):
        self.language_model.eval()
        device = self.language_model.device

        if task == 't2m':
            assert texts is not None
            inputs = ['<Caption_Placeholder>\n'] * len(lengths)
            outputs_tokens, _ = self.generate_direct(
                task, task_prompts=inputs, motion_tokens=None,
                motion_lengths=lengths, caption_texts=texts,
                max_new_tokens=128, do_sample=False, temperature=0.8,
            )
            return outputs_tokens
        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None
            inputs = ['<Motion_Placeholder>\n'] * len(lengths)
            _, cleaned_text = self.generate_direct(
                task, task_prompts=inputs, motion_tokens=motion_tokens,
                motion_lengths=lengths, caption_texts=texts,
                max_new_tokens=128, do_sample=False, temperature=0.8,
            )
            return cleaned_text
        raise ValueError(f"Unsupported task: {task}")

    # Backwards-compat stubs kept so any external caller referencing the
    # legacy T5 / GPT-2 string-templating helpers does not crash. They are
    # not exercised by the single-codebook forward / generate paths.
    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            token_list = [
                int(t.split('_')[-1].replace('>', ''))
                for t in string_list[1:-1]
            ]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list, dtype=int).to(self.language_model.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))
        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str):
        return prompt

    def template_fulfill(self, tasks, lengths, motion_strings, texts, stage='test'):
        inputs, outputs = [], []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            inputs.append(self.placeholder_fulfill(input_template, lengths[i], motion_strings[i], texts[i]))
            outputs.append(self.placeholder_fulfill(output_template, lengths[i], motion_strings[i], texts[i]))
        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except Exception:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'
        return f'<motion_id_{self.m_codebook_size}>' + content[startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def freeze_LLM(self):
        if not self.lora:
            for param in self.language_model.parameters():
                param.requires_grad = False
            print("[Base_MLM] LLM frozen.")
        else:
            print("[Base_MLM] LLM base weights frozen by PEFT; LoRA adapters trainable.")


class MLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_type: str = "llama",
        stage: str = "lm_pretrain",
        decoding_strategy: str = "multi_head",
        fusion_lambda: float = 1/3,
        body_codebook_size: int = 256,
        lh_codebook_size: int = 512,
        rh_codebook_size: int = 512,
        codebook_params_path: str = "",
        max_length: int = 192,
        lora: bool = False,
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        init_from_scratch: bool = False,
        scratch_hidden_size: int = 2048,
        scratch_intermediate_size: int = 8192,
        scratch_num_hidden_layers: int = 16,
        scratch_num_attention_heads: int = 32,
        scratch_num_key_value_heads: int = 8,
        scratch_max_position_embeddings: int = 131072,
        scratch_attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:

        super().__init__()
        self.max_length = max_length
        self.stage = stage
        self.lora = lora
        self.decoding_strategy = decoding_strategy
        self.fusion_lambda = fusion_lambda

        # 1. === Codebook Sizing & Special Token Index Definition ===
        # Store original sizes
        self.body_codebook_size_orig = body_codebook_size
        self.lh_codebook_size_orig = lh_codebook_size
        self.rh_codebook_size_orig = rh_codebook_size

        # Define indices for new special tokens for each codebook. This is more robust.
        self.body_motion_start_idx = self.body_codebook_size_orig
        self.body_motion_end_idx = self.body_codebook_size_orig + 1
        self.body_text_gen_idx = self.body_codebook_size_orig + 2

        self.lh_motion_start_idx = self.lh_codebook_size_orig
        self.lh_motion_end_idx = self.lh_codebook_size_orig + 1
        self.lh_text_gen_idx = self.lh_codebook_size_orig + 2

        self.rh_motion_start_idx = self.rh_codebook_size_orig
        self.rh_motion_end_idx = self.rh_codebook_size_orig + 1
        self.rh_text_gen_idx = self.rh_codebook_size_orig + 2

        # Calculate new, extended codebook sizes
        self.body_codebook_size_extended = body_codebook_size + 3
        self.lh_codebook_size_extended = lh_codebook_size + 3
        self.rh_codebook_size_extended = rh_codebook_size + 3

        self.my_task = 'mix'
        self.model_type = model_type
        self.init_from_scratch = init_from_scratch

        # 2. === Tokenizer and Base Model Initialization ===
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.add_tokens(['<motion_gen>'])
        self.motion_gen_token_id = self.tokenizer.convert_tokens_to_ids('<motion_gen>')

        self.old_vocabulary_size = len(self.tokenizer) - 1
        self.total_token_num = self.old_vocabulary_size + body_codebook_size + lh_codebook_size + rh_codebook_size

        if model_type == "llama":
            self.lm_type = 'dec_only'
            if init_from_scratch:
                print(f"[MLM] Initializing LLaMA from scratch: layers={scratch_num_hidden_layers}, "
                      f"hidden={scratch_hidden_size}, heads={scratch_num_attention_heads}")
                scratch_config = LlamaConfig(
                    vocab_size=len(self.tokenizer),
                    hidden_size=scratch_hidden_size,
                    intermediate_size=scratch_intermediate_size,
                    num_hidden_layers=scratch_num_hidden_layers,
                    num_attention_heads=scratch_num_attention_heads,
                    num_key_value_heads=scratch_num_key_value_heads,
                    head_dim=scratch_hidden_size // scratch_num_attention_heads,
                    max_position_embeddings=scratch_max_position_embeddings,
                    hidden_act="silu",
                    initializer_range=0.02,
                    rms_norm_eps=1e-05,
                    rope_theta=500000.0,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    model_type="llama",
                    attention_bias=False,
                    attention_dropout=scratch_attention_dropout,
                    mlp_bias=False,
                    tie_word_embeddings=False,
                    use_cache=True,
                )
                self.language_model = LlamaForCausalLM(scratch_config)
            else:
                print(f"[MLM] Loading pretrained LLaMA from: {model_path}")
                self.language_model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
                self.language_model.resize_token_embeddings(len(self.tokenizer))
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if self.lora:
            from peft import LoraConfig, TaskType, get_peft_model
            print("Applying LoRA to the language model...")
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=(
                    ["q_proj", "v_proj"]+
                    ["k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]+
                    ["emb_tokens", "lm_head"]
                ),
                lora_dropout=lora_dropout, bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            self.language_model.print_trainable_parameters()

        self.embedding_dim = self.language_model.config.hidden_size  # 2048
        codebook_dim = 1024
        self.body_codebook = nn.Embedding(self.body_codebook_size_extended, codebook_dim)
        self.lh_codebook = nn.Embedding(self.lh_codebook_size_extended, codebook_dim)
        self.rh_codebook = nn.Embedding(self.rh_codebook_size_extended, codebook_dim)

        self.body_projection = nn.Linear(codebook_dim, self.embedding_dim)
        self.lh_projection = nn.Linear(codebook_dim, self.embedding_dim)
        self.rh_projection = nn.Linear(codebook_dim, self.embedding_dim)
        # 3. === Codebook & Projection Loading from VQ-VAE ===
        if codebook_params_path and os.path.exists(codebook_params_path):
            print(f"Loading codebook params from: {codebook_params_path}")
            saved_params = torch.load(codebook_params_path, map_location='cpu')

            try:
                body_codebook_tensor = saved_params['body_codebook']
                lh_codebook_tensor = saved_params['left_hand_codebook']
                rh_codebook_tensor = saved_params['right_hand_codebook']
                assert body_codebook_tensor.shape[0] == self.body_codebook_size_orig, \
                    f"Body codebook size mismatch: file has {body_codebook_tensor.shape[0]}, expected {self.body_codebook_size_orig}"
                assert lh_codebook_tensor.shape[0] == self.lh_codebook_size_orig, \
                    f"LH codebook size mismatch: file has {lh_codebook_tensor.shape[0]}, expected {self.lh_codebook_size_orig}"
                assert rh_codebook_tensor.shape[0] == self.rh_codebook_size_orig, \
                    f"RH codebook size mismatch: file has {rh_codebook_tensor.shape[0]}, expected {self.rh_codebook_size_orig}"
                self.body_codebook.weight.data[:self.body_codebook_size_orig] = body_codebook_tensor
                self.lh_codebook.weight.data[:self.lh_codebook_size_orig] = lh_codebook_tensor
                self.rh_codebook.weight.data[:self.rh_codebook_size_orig] = rh_codebook_tensor
                self.freeze_original_codebook_parts()
                print("="*80+'\n'+ "Successfully loaded codebook"+'\n'+"="*80)
            except Exception as e:
                print("="*80+'\n'+ f"Fail to load codebook: {e}"+'\n'+"="*80)

            try:
                self.body_projection.load_state_dict(saved_params['body_proj_up'])
                self.lh_projection.load_state_dict(saved_params['lh_proj_up'])
                self.rh_projection.load_state_dict(saved_params['rh_proj_up'])
                print("="*80+'\n'+ "Successfully loaded projection layer weights."+'\n'+"="*80)
            except Exception as e:
                print("="*80+'\n'+ f"Fail to load projection layer weights: {e}"+'\n'+"="*80)
        else:
            if codebook_params_path:
                print(f"WARNING: codebook_params_path not found: {codebook_params_path}")
            else:
                print("WARNING: codebook_params_path not set, codebook/projection initialized randomly.")

        # 5. === Multi-Head Architecture Initialization ===
        self.body_head = nn.Linear(self.embedding_dim, self.body_codebook_size_extended, bias=False)
        self.lh_head = nn.Linear(self.embedding_dim, self.lh_codebook_size_extended, bias=False)
        self.rh_head = nn.Linear(self.embedding_dim, self.rh_codebook_size_extended, bias=False)

        if self.stage == 'vae':
            self.freeze_LLM()
            self.freeze_motion_components()

    def freeze_LLM(self):
        if not self.lora:
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
            print("LLM frozen (standard method).")
        else:
            print("LLM base weights are frozen by PEFT. Only LoRA adapters are trainable.")

    def freeze_codebook_and_projection(self):
        """Freeze codebook embeddings and projection layers during LLM training.
        Output heads (body_head, lh_head, rh_head) remain trainable."""
        print("Freezing codebook embeddings and projection layers for LLM training...")
        components = [
            self.body_codebook, self.lh_codebook, self.rh_codebook,
        ]
        for component in components:
            for param in component.parameters():
                param.requires_grad = False
        print("Codebook and projection layers frozen. Output heads remain trainable.")

    def freeze_motion_components(self):
        """Freezes ALL motion components (codebooks, projections, heads) during VAE training."""
        print("Freezing all motion-related components (codebooks, projections, heads)...")
        components = [
            self.body_codebook, self.lh_codebook, self.rh_codebook,
            self.body_projection, self.lh_projection, self.rh_projection,
            self.body_head, self.lh_head, self.rh_head
        ]
        for component in components:
            for param in component.parameters():
                param.requires_grad = False
        print("Motion components have been frozen.")

    def forward(self, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict, istrain: bool=True):
        if self.lm_type == 'dec_only':
            return self.forward_dec_only(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only decoder-only models are supported")

    def freeze_original_codebook_parts(self):
        def create_grad_hook(original_size):
            def hook(grad):
                grad[:original_size, :] = 0
                return grad
            return hook

        self.body_codebook.weight.register_hook(create_grad_hook(self.body_codebook_size_orig))
        self.lh_codebook.weight.register_hook(create_grad_hook(self.lh_codebook_size_orig))
        self.rh_codebook.weight.register_hook(create_grad_hook(self.rh_codebook_size_orig))
        print(f"Original parts of codebooks frozen. Special tokens remain trainable.")

    def forward_dec_only(self, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict):
        device = motion_tokens.device

        if self.my_task == 't2m':
            tasks = [{'input': ['<Caption_Placeholder>\n'], 'output': ['<Motion_Placeholder>']}] * len(lengths)
        elif self.my_task == 'm2t':
            tasks = [{'input': ['<Motion_Placeholder>\n'], 'output': ['<Caption_Placeholder>']}] * len(lengths)
        else:
            pass

        inputs, outputs = self.template_fulfill(tasks, lengths, texts, texts)
        combined_strings = [inp + out for inp, out in zip(inputs, outputs)]

        sample_tasks = []
        for inp in inputs:
            if 'Caption_Placeholder' in inp:
                sample_tasks.append('t2m')
            else:
                sample_tasks.append('m2t')

        inputs_embeds, attention_mask = self._create_hybrid_inputs_multi_head(
            string_list=combined_strings, motion_tokens=motion_tokens,
            text_list=texts, lengths=lengths, device=device, add_eos=True, sample_tasks=sample_tasks
        )

        text_labels, motion_labels = self._create_hybrid_labels_multi_head_nogentoken(
            string_list=combined_strings, motion_tokens=motion_tokens,
            text_list=texts, lengths=lengths, device=device, sample_tasks=sample_tasks
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Shift hidden states and labels for causal next-token prediction.
        shift_hidden_states = hidden_states[:, :-1, :].contiguous()

        shift_text_labels = text_labels[:, 1:].contiguous()
        shift_motion_labels = motion_labels[:, 1:, :].contiguous()

        shift_attention_mask = attention_mask[:, :-1].contiguous()

        active_mask = shift_attention_mask.view(-1) == 1
        active_hidden_states = shift_hidden_states.view(-1, self.embedding_dim)[active_mask]

        # Text Loss
        active_text_labels = shift_text_labels.view(-1)[active_mask]
        text_logits = self.language_model.lm_head(active_hidden_states)
        text_loss = F.cross_entropy(text_logits, active_text_labels, ignore_index=-100)
        text_loss = torch.where(torch.isnan(text_loss), torch.tensor(0.0, device=text_loss.device), text_loss)
        # Motion Loss
        active_motion_labels = shift_motion_labels.view(-1, 3)[active_mask]
        body_labels, lh_labels, rh_labels = active_motion_labels.T

        body_logits = self.body_head(active_hidden_states)
        lh_logits = self.lh_head(active_hidden_states)
        rh_logits = self.rh_head(active_hidden_states)

        body_loss = F.cross_entropy(body_logits, body_labels, ignore_index=-100)
        lh_loss = F.cross_entropy(lh_logits, lh_labels, ignore_index=-100)
        rh_loss = F.cross_entropy(rh_logits, rh_labels, ignore_index=-100)

        motion_loss = body_loss + lh_loss + rh_loss
        motion_loss = torch.where(torch.isnan(motion_loss), torch.tensor(0.0, device=motion_loss.device), motion_loss)

        total_loss = text_loss + motion_loss

        class LossOutput:
            def __init__(self, loss, text_loss, motion_loss):
                self.loss = loss
                self.text_loss = text_loss
                self.motion_loss = motion_loss

        return LossOutput(total_loss, text_loss, motion_loss)

    def _get_fused_special_embed(self, body_idx, lh_idx, rh_idx, device):
        body_e = self.body_projection(self.body_codebook(torch.tensor(body_idx, device=device)))
        lh_e = self.lh_projection(self.lh_codebook(torch.tensor(lh_idx, device=device)))
        rh_e = self.rh_projection(self.rh_codebook(torch.tensor(rh_idx, device=device)))
        lam = self.fusion_lambda
        return (1 - 2 * lam) * body_e + lam * lh_e + lam * rh_e

    def _create_hybrid_inputs_multi_head(self, string_list, motion_tokens, text_list, lengths, device, add_eos=False, padding_side='right', sample_tasks=None):
        batch_size = len(string_list)
        final_embeddings = []
        text_embedder = self.language_model.get_input_embeddings()
        placeholders = {'<Motion_Placeholder>', '<Caption_Placeholder>'}
        placeholder_pattern = re.compile(f"({'|'.join(placeholders)})")

        motion_start_embed = self._get_fused_special_embed(self.body_motion_start_idx, self.lh_motion_start_idx, self.rh_motion_start_idx, device)
        motion_end_embed = self._get_fused_special_embed(self.body_motion_end_idx, self.lh_motion_end_idx, self.rh_motion_end_idx, device)

        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id], device=device)
        eos_embedding = text_embedder(eos_token_id_tensor)
        pad_token_id_tensor = torch.tensor([self.tokenizer.pad_token_id], device=device)
        pad_embedding = text_embedder(pad_token_id_tensor)
        bos_embedding = text_embedder(torch.tensor([self.tokenizer.bos_token_id], device=device))

        for i in range(batch_size):
            text = string_list[i]
            parts = [part for part in placeholder_pattern.split(text) if part]
            embedding_chunks = [bos_embedding]
            for part in parts:
                if part not in placeholders:
                    token_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    if token_ids.numel() > 0: embedding_chunks.append(text_embedder(token_ids.squeeze(0)))
                elif part == '<Motion_Placeholder>':
                    num_frames = lengths[i] // 3
                    if num_frames > 0:
                        embedding_chunks.append(motion_start_embed.unsqueeze(0))
                        flat_motion_toks = motion_tokens[i][:lengths[i]].long().view(-1, 3)
                        body_toks, lh_toks, rh_toks = flat_motion_toks.T
                        body_embeds = self.body_projection(self.body_codebook(body_toks))
                        lh_embeds = self.lh_projection(self.lh_codebook(lh_toks))
                        rh_embeds = self.rh_projection(self.rh_codebook(rh_toks))
                        fused_embeds = (1 - 2 * self.fusion_lambda) * body_embeds + self.fusion_lambda * lh_embeds + self.fusion_lambda * rh_embeds
                        embedding_chunks.append(fused_embeds)
                        embedding_chunks.append(motion_end_embed.unsqueeze(0))
                elif part == '<Caption_Placeholder>':
                    caption_ids = self.tokenizer(text_list[i], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    if caption_ids.numel() > 0: embedding_chunks.append(text_embedder(caption_ids.squeeze(0)))

            cur_task = sample_tasks[i] if sample_tasks else 't2m'
            if add_eos and cur_task == 'm2t':
                embedding_chunks.append(eos_embedding)

            if embedding_chunks:
                final_embeddings.append(torch.cat(embedding_chunks, dim=0))

        max_len_in_batch = max(emb.shape[0] for emb in final_embeddings) if final_embeddings else 0
        effective_max_len = min(max_len_in_batch, self.max_length)
        if effective_max_len==self.max_length and max_len_in_batch > self.max_length:
            print(f"Warning: Max length {self.max_length} exceeded. Actual length: {max_len_in_batch}. Truncating.")

        pad_embedding_vector = pad_embedding.squeeze(0)
        padded_embeddings = torch.full((batch_size, effective_max_len, self.embedding_dim), fill_value=0.0, device=device)
        padded_embeddings[:] = pad_embedding_vector
        padded_masks = torch.zeros(batch_size, effective_max_len, dtype=torch.long, device=device)

        for i, embedding in enumerate(final_embeddings):
            len_to_copy = min(embedding.shape[0], effective_max_len)
            if padding_side == 'right':
                padded_embeddings[i, :len_to_copy] = embedding[:len_to_copy]
                padded_masks[i, :len_to_copy] = 1
            else: # padding_side == 'left'
                start_index = effective_max_len - len_to_copy
                padded_embeddings[i, start_index:] = embedding[:len_to_copy]
                padded_masks[i, start_index:] = 1

        return padded_embeddings, padded_masks

    def _create_hybrid_labels_multi_head_nogentoken(self, string_list, motion_tokens, text_list, lengths, device, sample_tasks):
        batch_size = len(string_list)
        text_labels_list, motion_labels_list = [], []
        placeholders = {'<Motion_Placeholder>', '<Caption_Placeholder>'}
        placeholder_pattern = re.compile(f"({'|'.join(placeholders)})")

        start_motion_label = torch.tensor([self.body_motion_start_idx, self.lh_motion_start_idx, self.rh_motion_start_idx], device=device, dtype=torch.long)
        end_motion_label = torch.tensor([self.body_motion_end_idx, self.lh_motion_end_idx, self.rh_motion_end_idx], device=device, dtype=torch.long)
        eos_label_tensor = torch.tensor([self.tokenizer.eos_token_id], device=device)

        for i in range(batch_size):
            cur_task = sample_tasks[i]
            text = string_list[i]
            parts = [part for part in placeholder_pattern.split(text) if part]
            text_label_chunks, motion_label_chunks = [torch.tensor([-100], device=device, dtype=torch.long)], [torch.full((1, 3), -100, device=device, dtype=torch.long)]

            for part in parts:
                if part not in placeholders:
                    token_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
                    text_label_chunks.append(torch.full_like(token_ids, -100))
                    motion_label_chunks.append(torch.full((len(token_ids), 3), -100, device=device, dtype=torch.long))
                elif part == '<Caption_Placeholder>':
                    caption_ids = self.tokenizer(text_list[i], add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
                    if cur_task == 't2m':
                        text_label_chunks.append(torch.full_like(caption_ids, -100))
                        motion_label_chunks.append(torch.full((len(caption_ids), 3), -100, device=device, dtype=torch.long))
                    else:
                        text_label_chunks.append(caption_ids)
                        motion_label_chunks.append(torch.full((len(caption_ids), 3), -100, device=device, dtype=torch.long))
                elif part == '<Motion_Placeholder>':
                    num_frames = lengths[i] // 3
                    if num_frames > 0:
                        total_motion_len = num_frames + 2
                        if cur_task == 't2m':
                            text_label_chunks.append(torch.full((total_motion_len,), -100, device=device, dtype=torch.long))
                            motion_label_chunks.append(start_motion_label.unsqueeze(0))
                            motion_label_chunks.append(motion_tokens[i, :lengths[i]].long().view(-1, 3))
                            motion_label_chunks.append(end_motion_label.unsqueeze(0))
                        else:
                            text_label_chunks.append(torch.full((total_motion_len,), -100, device=device, dtype=torch.long))
                            motion_label_chunks.append(torch.full((total_motion_len, 3), -100, device=device, dtype=torch.long))

            if cur_task == 'm2t':
                text_label_chunks.append(eos_label_tensor)
                motion_label_chunks.append(torch.full((1, 3), -100, device=device, dtype=torch.long))

            if text_label_chunks: text_labels_list.append(torch.cat(text_label_chunks))
            if motion_label_chunks: motion_labels_list.append(torch.cat(motion_label_chunks))

        padded_text_labels = torch.nn.utils.rnn.pad_sequence(text_labels_list, batch_first=True, padding_value=-100)
        padded_motion_labels = torch.nn.utils.rnn.pad_sequence(motion_labels_list, batch_first=True, padding_value=-100)
        return padded_text_labels, padded_motion_labels

    def _create_hybrid_labels_multi_head_with_gentoken(self, string_list, motion_tokens, text_list, lengths, device, sample_tasks):
        batch_size = len(string_list)
        text_labels_list, motion_labels_list = [], []
        placeholders = {'<Motion_Placeholder>', '<Caption_Placeholder>'}
        placeholder_pattern = re.compile(f"({'|'.join(placeholders)})")

        text_gen_motion_label = torch.tensor([self.body_text_gen_idx, self.lh_text_gen_idx, self.rh_text_gen_idx], device=device, dtype=torch.long)
        start_motion_label = torch.tensor([self.body_motion_start_idx, self.lh_motion_start_idx, self.rh_motion_start_idx], device=device, dtype=torch.long)
        end_motion_label = torch.tensor([self.body_motion_end_idx, self.lh_motion_end_idx, self.rh_motion_end_idx], device=device, dtype=torch.long)
        eos_label_tensor = torch.tensor([self.tokenizer.eos_token_id], device=device)

        for i in range(batch_size):
            cur_task = sample_tasks[i]
            text = string_list[i]
            parts = [part for part in placeholder_pattern.split(text) if part]
            text_label_chunks, motion_label_chunks = [torch.tensor([-100], device=device, dtype=torch.long)], [torch.full((1, 3), -100, device=device, dtype=torch.long)]

            for part in parts:
                if part not in placeholders:
                    token_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
                    text_label_chunks.append(torch.full_like(token_ids, -100))
                    motion_label_chunks.append(torch.full((len(token_ids), 3), -100, device=device, dtype=torch.long))

                elif part == '<Caption_Placeholder>':
                    caption_ids = self.tokenizer(text_list[i], add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
                    if cur_task == 't2m':
                        text_label_chunks.append(torch.full_like(caption_ids, -100))
                        motion_label_chunks.append(torch.full((len(caption_ids), 3), -100, device=device, dtype=torch.long))
                    else:
                        text_label_chunks.append(caption_ids)
                        motion_label_chunks.append(text_gen_motion_label.expand(len(caption_ids), -1))

                elif part == '<Motion_Placeholder>':
                    num_frames = lengths[i] // 3
                    if num_frames > 0:
                        total_motion_len = num_frames + 2

                        if cur_task == 'm2t':
                            text_label_chunks.append(torch.full((total_motion_len,), -100, device=device, dtype=torch.long))
                            motion_label_chunks.append(torch.full((total_motion_len, 3), -100, device=device, dtype=torch.long))
                        else:
                            text_label_chunks.append(torch.tensor([self.motion_gen_token_id], device=device))
                            text_label_chunks.append(torch.full((num_frames,), self.motion_gen_token_id, device=device))
                            text_label_chunks.append(torch.tensor([self.motion_gen_token_id], device=device))

                            motion_label_chunks.append(start_motion_label.unsqueeze(0))
                            motion_label_chunks.append(motion_tokens[i, :lengths[i]].long().view(-1, 3))
                            motion_label_chunks.append(end_motion_label.unsqueeze(0))

            if cur_task == 'm2t':
                text_label_chunks.append(eos_label_tensor)
                motion_label_chunks.append(text_gen_motion_label.unsqueeze(0))

            if text_label_chunks: text_labels_list.append(torch.cat(text_label_chunks))
            if motion_label_chunks: motion_labels_list.append(torch.cat(motion_label_chunks))

        padded_text_labels = torch.nn.utils.rnn.pad_sequence(text_labels_list, batch_first=True, padding_value=-100)
        padded_motion_labels = torch.nn.utils.rnn.pad_sequence(motion_labels_list, batch_first=True, padding_value=-100)
        return padded_text_labels, padded_motion_labels

    def _generate_text_loop(self, next_step_hidden_state, past_key_values, max_new_tokens, do_sample, temperature, top_k):
        device = self.language_model.device
        batch_size = next_step_hidden_state.shape[0]
        generated_text_ids = [[] for _ in range(batch_size)]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for k in range(max_new_tokens):
                text_logits = self.language_model.lm_head(next_step_hidden_state)
                next_token_ids = self._sample_logits(text_logits.squeeze(1), do_sample, temperature, top_k)

                if next_token_ids.dim() == 2 and next_token_ids.size(1) == 1:
                    next_token_ids = next_token_ids.squeeze(1)

                just_finished = (next_token_ids == self.tokenizer.eos_token_id)
                unfinished_sequences = unfinished_sequences & (~just_finished)

                for i in range(batch_size):
                    if unfinished_sequences[i].item():
                        generated_text_ids[i].append(next_token_ids[i].item())

                if not unfinished_sequences.any():
                    break

                if next_token_ids.dim() == 1:
                    next_token_ids = next_token_ids.unsqueeze(1)

                next_input_embed = self.language_model.get_input_embeddings()(next_token_ids)

                if next_input_embed.dim() == 2:
                    next_input_embed = next_input_embed.unsqueeze(1)

                outputs = self.language_model.model(
                    inputs_embeds=next_input_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values
                next_step_hidden_state = outputs.hidden_states[-1]

        final_texts = self.tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)
        return None, final_texts

    def _generate_motion_loop(self, next_step_hidden_state, past_key_values, max_new_frames, do_sample, temperature, top_k):
        device = self.language_model.device
        batch_size = next_step_hidden_state.shape[0]

        generated_motion_lists = [[] for _ in range(batch_size)]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        current_hidden_state = next_step_hidden_state

        with torch.no_grad():
            for k in range(max_new_frames):
                if k == max_new_frames-1:
                    print("Inference achieve max length")
                if not unfinished_sequences.any():
                    break

                squeezed_hidden_state = current_hidden_state.squeeze(1)

                body_logits = self.body_head(squeezed_hidden_state)
                lh_logits = self.lh_head(squeezed_hidden_state)
                rh_logits = self.rh_head(squeezed_hidden_state)

                body_tok = self._sample_logits(body_logits, do_sample, temperature, top_k)
                lh_tok = self._sample_logits(lh_logits, do_sample, temperature, top_k)
                rh_tok = self._sample_logits(rh_logits, do_sample, temperature, top_k)

                body_tok_flat = body_tok.squeeze(1) if body_tok.dim() == 2 else body_tok
                lh_tok_flat = lh_tok.squeeze(1) if lh_tok.dim() == 2 else lh_tok
                rh_tok_flat = rh_tok.squeeze(1) if rh_tok.dim() == 2 else rh_tok

                is_body_end = (body_tok_flat == self.body_motion_end_idx)
                is_lh_end = (lh_tok_flat == self.lh_motion_end_idx)
                is_rh_end = (rh_tok_flat == self.rh_motion_end_idx)
                any_end_token_generated = is_body_end | is_lh_end | is_rh_end
                just_finished = any_end_token_generated & unfinished_sequences

                for i in range(batch_size):
                    if not just_finished[i].item() and unfinished_sequences[i].item():
                        # Exclude control tokens from the generated motion sequence.
                        is_special_token = (body_tok[i] >= self.body_codebook_size_orig or
                                            lh_tok[i] >= self.lh_codebook_size_orig or
                                            rh_tok[i] >= self.rh_codebook_size_orig)
                        if not is_special_token:
                            generated_motion_lists[i].append(torch.cat([body_tok[i], lh_tok[i], rh_tok[i]]))

                unfinished_sequences = unfinished_sequences & (~just_finished)

                body_tok_for_embed = body_tok.squeeze(1) if body_tok.dim() == 2 else body_tok
                lh_tok_for_embed = lh_tok.squeeze(1) if lh_tok.dim() == 2 else lh_tok
                rh_tok_for_embed = rh_tok.squeeze(1) if rh_tok.dim() == 2 else rh_tok

                body_embed = self.body_projection(self.body_codebook(body_tok_for_embed))
                lh_embed = self.lh_projection(self.lh_codebook(lh_tok_for_embed))
                rh_embed = self.rh_projection(self.rh_codebook(rh_tok_for_embed))

                next_input_embed = (
                    (1 - 2 * self.fusion_lambda) * body_embed +
                    self.fusion_lambda * lh_embed +
                    self.fusion_lambda * rh_embed
                ).unsqueeze(1)

                outputs = self.language_model(
                    inputs_embeds=next_input_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values
                current_hidden_state = outputs.hidden_states[-1]

        final_motion_tensors = []
        for m_list in generated_motion_lists:
            if m_list:
                stacked_tensor = torch.stack(m_list, dim=0)
                flattened_tensor = stacked_tensor.flatten()
                final_motion_tensors.append(flattened_tensor)
                if flattened_tensor.shape[0]%3!=0:
                    print("error!",flattened_tensor.shape[0])
            else:
                final_motion_tensors.append(torch.empty(0, device=device, dtype=torch.long))

        return final_motion_tensors, [""]

    def generate_direct(self, task, task_prompts, motion_tokens, motion_lengths, caption_texts, max_new_tokens=128, do_sample=False, temperature=1.0, top_k=50):
        self.language_model.eval()
        device = self.language_model.device

        inputs_embeds, attention_mask = self._create_hybrid_inputs_multi_head(
            string_list=task_prompts, motion_tokens=motion_tokens,
            text_list=caption_texts, lengths=motion_lengths, device=device, add_eos=False, padding_side='left'
        )

        with torch.no_grad():
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            next_step_hidden_state = outputs.hidden_states[-1][:, -1:, :]

        if task == 't2m':
            return self._generate_motion_loop(next_step_hidden_state, past_key_values, max_new_tokens, do_sample, temperature, top_k)
        elif task == 'm2t':
            return self._generate_text_loop(next_step_hidden_state, past_key_values, max_new_tokens, do_sample, temperature, top_k)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _sample_logits(self, logits, do_sample, temperature, top_k):
        if do_sample:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)



    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
        if task == 't2m':
            inputs = ['<Caption_Placeholder>\n'] * len(lengths)
            outputs_tokens, cleaned_text = self.generate_direct(task,
                                                                task_prompts = inputs,
                                                                motion_tokens = None,
                                                                motion_lengths = None,
                                                                caption_texts = texts,
                                                                max_new_tokens=128,
                                                                do_sample=False,
                                                                temperature=0.8,
                                                                )
            return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            inputs = ['<Motion_Placeholder>\n'] * len(lengths)
            outputs_tokens, cleaned_text = self.generate_direct(
                task,
                task_prompts = inputs,
                motion_tokens = motion_tokens,
                motion_lengths = lengths,
                caption_texts = texts,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.8,
            )

            return cleaned_text

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str):

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'