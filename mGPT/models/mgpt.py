import numpy as np
import os
import random
import torch
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
from mGPT.archs.mgpt_vq import SplitBodyHandVQVAE, UnifiedSplitBodyHandVQVAE, VQVae

# 3 codebook
class SignGPT(BaseModel):
    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=False,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()
        self.count = 0
        self.number = 0

        self.vae = instantiate_from_config(motion_vae)

        if stage == 'vae':
            self.lm = None
        else:
            self.lm = instantiate_from_config(lm)

            self.lora = False
            self.frozen_init_textemb = False
            try:
                self.old_vocab_size = self.lm.old_vocabulary_size
                self.new_vocab_size = self.lm.total_token_num
                print("old_vocab_size: ", self.old_vocab_size)
                print("new_vocab_size: ", self.new_vocab_size)
            except:
                pass

            if self.lora:
                from peft import LoraConfig, TaskType, get_peft_model
                print("Applying LoRA to the language model...")
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=128,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.lm = get_peft_model(self.lm, lora_config)
                self.lm.print_trainable_parameters()
                self._setup_trainable_parameters(self.old_vocab_size)

            # Freeze VAE during LLM training
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def _setup_trainable_parameters(self, old_vocab_size):
        # 1. Custom projection layers - fully trainable
        custom_modules = ['body_projection', 'hand_projection', 'motion_projection', 'linear']
        for name, param in self.lm.named_parameters():
            if any(m in name for m in custom_modules):
                param.requires_grad = True
                print(f"✓ Trainable: {name} - {param.shape}")

        if self.frozen_init_textemb:
            # Correctly get the base MLM model, unwrapping it from the PEFT wrapper.
            # When PEFT is used, the original model is in `.base_model`.
            unwrapped_model = self.lm.base_model if hasattr(self.lm, 'base_model') else self.lm

            # 2. embed_tokens - is inside the '.model' attribute of LlamaForCausalLM
            # We must access it via `unwrapped_model.model.embed_tokens`
            unwrapped_model.model.model.embed_tokens.weight.requires_grad = True
            def hook_emb(grad, old_size=old_vocab_size):
                grad[:old_size] = 0
                return grad
            if self.hook_textemb:
                unwrapped_model.model.model.embed_tokens.weight.register_hook(hook_emb)
                print(f"✓ Partially trainable: embed_tokens.weight {unwrapped_model.model.model.embed_tokens.weight.shape} - new tokens only")

            # 3. lm_head - is a direct attribute of LlamaForCausalLM
            # We can access it directly via `unwrapped_model.lm_head`
            unwrapped_model.model.lm_head.weight.requires_grad = True
            def hook_head(grad, old_size=old_vocab_size):
                grad[:old_size] = 0
                return grad

            if self.hook_textemb:
                unwrapped_model.model.lm_head.weight.register_hook(hook_head)
                print(f"✓ Partially trainable: lm_head.weight {unwrapped_model.model.lm_head.weight.shape} - new tokens only")

            self._print_trainable_stats()

    def _print_trainable_stats(self):
        """Print trainable-parameter statistics."""
        total = sum(p.numel() for p in self.lm.parameters())
        trainable = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Trainable ratio: {100 * trainable / total:.2f}%")
        print("="*60 + "\n")


    def slc_forward(self, batch, task="t2m"):
        try:
            motion_tokens = batch["motion_tokens"]
            motion_lengths = batch["motion_lengths"]
            batch_size = len(motion_tokens)
        except:
            motion_tokens = None
            motion_lengths = None
        try:
            texts = batch["texts"]
            batch_size = len(texts)
        except:
            texts = None
        task_prompts = batch["task_prompts"]

        # Forward
        outputs, output_texts = self.lm.generate_direct(
                                            task,
                                            task_prompts = task_prompts,
                                            motion_tokens = motion_tokens,
                                            motion_lengths = motion_lengths,
                                            caption_texts = texts,
                                            max_new_tokens = 294 if task=='m2t' else 200,
                                            do_sample = True,
                                            temperature=0.9
                                            )

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0
        batch_body_codes = []
        batch_lh_codes = []
        batch_rh_codes = []

        if task in ["t2m"]:
            for i in range(batch_size):
                interleaved_codes = outputs[i]
                if interleaved_codes.dim() == 1:
                    interleaved_codes = interleaved_codes.unsqueeze(0)

                motion = self._decode_interleaved_motion(interleaved_codes, self.device)

                if isinstance(self.vae, SplitBodyHandVQVAE):
                    B, total_len = interleaved_codes.shape

                    # Ensure total_len is divisible by 3
                    if total_len % 3 != 0:
                        new_len = (total_len // 3) * 3
                        interleaved_codes = interleaved_codes[:, :new_len]
                        total_len = new_len

                    seq_len = total_len // 3
                    reshaped_codes = interleaved_codes.view(B, seq_len, 3)

                    body_codes = reshaped_codes[..., 0]
                    left_hand_codes = reshaped_codes[..., 1]
                    right_hand_codes = reshaped_codes[..., 2]

                    batch_body_codes.append(body_codes.squeeze())
                    batch_lh_codes.append(left_hand_codes.squeeze())
                    batch_rh_codes.append(right_hand_codes.squeeze())
                else:
                    batch_body_codes.append(torch.tensor([]))
                    batch_lh_codes.append(torch.tensor([]))
                    batch_rh_codes.append(torch.tensor([]))

                lengths.append(motion.shape[1])
                feats_rst_lst.append(motion)
                if motion.shape[1] > max_len:
                    max_len = motion.shape[1]

        if task=='t2m':
            feats_rst = torch.zeros(
                (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

            # padding and concat
            for i in range(len(feats_rst_lst)):
                feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

            # Recover joints for evaluation
            joints_rst = self.feats2joints(feats_rst)

            outputs = {
                "texts": output_texts,
                "feats": feats_rst,
                "joints": joints_rst,
                "length": lengths,
                "motion_tokens": outputs
            }
        else:
            outputs = {
                "texts": output_texts,
                "feats": None,
                "joints": None,
                "length": None
            }

        return outputs

    def forward(self, batch, task="t2m"):
        motion_feats = batch.get("motion_feat", None)
        texts = batch.get("init_texts", None)

        # Determine batch size safely
        if texts is not None:
            batch_size = len(texts)
        elif motion_feats is not None:
            batch_size = motion_feats.shape[0]
        else:
            raise ValueError("The batch must contain either 'init_texts' or 'motion_feat'.")

        # Initialize motion tokens and lengths to None
        motion_tokens = None
        motion_lengths = None

        # Tokenize input motion inside the model for motion-to-text inference.
        if task == 'm2t':
            task_prompts = ['<Motion_Placeholder>\n'] * batch_size
            if motion_feats is not None:
                # Encode the motion features into discrete tokens here
                with torch.no_grad():
                    motion_tokens, _ = self.vae.encode(motion_feats)

                # Calculate the length of the token sequence
                if motion_tokens is not None:
                    motion_lengths = [motion_tokens.shape[1]] * batch_size
        else:
            task_prompts = ['<Caption_Placeholder>'] * batch_size

        # Forward pass through the language model
        outputs, output_texts = self.lm.generate_direct(
                                            task,
                                            task_prompts=task_prompts,
                                            motion_tokens=motion_tokens,
                                            motion_lengths=motion_lengths,
                                            caption_texts=texts,
                                            max_new_tokens=294 if task == 'm2t' else 200)

        # Motion Decode (This part is for t2m and remains the same)
        feats_rst_lst = []
        lengths = []
        max_len = 0

        if task in ["t2m"]:
            for i in range(batch_size):
                interleaved_codes = outputs[i]
                if interleaved_codes.dim() == 1:
                    interleaved_codes = interleaved_codes.unsqueeze(0)

                B, total_len = interleaved_codes.shape

                if total_len == 0:
                    lengths.append(0)
                    zero_motion = torch.zeros((1, 0, self.feats2joints.input_feats_dim), device=self.device)
                    feats_rst_lst.append(zero_motion)
                    continue

                motion = self._decode_interleaved_motion(interleaved_codes, self.device)

                lengths.append(motion.shape[1])
                feats_rst_lst.append(motion)
                if motion.shape[1] > max_len:
                    max_len = motion.shape[1]

        if task == 't2m':
            # This check is important for cases where no motion was generated
            if not feats_rst_lst or max_len == 0:
                 feats_rst = torch.zeros((batch_size, 0, self.feats2joints.input_feats_dim), device=self.device)
                 joints_rst = torch.zeros((batch_size, 0, 73, 3), device=self.device)
            else:
                motion_feat_dim = feats_rst_lst[0].shape[-1]
                feats_rst = torch.zeros((len(feats_rst_lst), max_len, motion_feat_dim)).to(self.device)
                for i in range(len(feats_rst_lst)):
                    feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]
                joints_rst = self.feats2joints(feats_rst)

            return_outputs = {
                "texts": output_texts,
                "feats": feats_rst,
                "joints": joints_rst,
                "length": lengths
            }
        else:
            return_outputs = {
                "texts": output_texts,
                "feats": None,
                "joints": None,
                "length": None
            }

        return return_outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch["text"]

        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts = texts, motion_tokens = tokens_ref, lengths = lengths, tasks = tasks, istrain=True)

        return {'outputs': outputs}

    def _check_gradients(self):
        """Report gradients for frozen and newly added vocabulary rows."""
        try:
            unwrapped = self.lm.base_model.model if hasattr(self.lm, 'base_model') else self.lm
            old_size = self.old_vocab_size
            step = self.trainer.global_step

            print(f"\n{'='*60}")
            print(f"Gradient Check at Step {step}")
            print(f"{'='*60}")

            embed_layer = unwrapped.model.embed_tokens
            if embed_layer.weight.grad is not None:
                frozen_grad = embed_layer.weight.grad[:old_size].abs().sum().item()
                trainable_grad = embed_layer.weight.grad[old_size:].abs().sum().item()
                print(f"✓ embed_tokens:")
                print(f"  - Frozen   [:old]: {frozen_grad:12.6f} {'⚠️ NON-ZERO!' if frozen_grad > 1e-6 else '✓'}")
                print(f"  - Trainable[new]: {trainable_grad:12.6f}")

            head_layer = unwrapped.lm_head
            if head_layer.weight.grad is not None:
                frozen_grad = head_layer.weight.grad[:old_size].abs().sum().item()
                trainable_grad = head_layer.weight.grad[old_size:].abs().sum().item()
                print(f"✓ lm_head:")
                print(f"  - Frozen   [:old]: {frozen_grad:12.6f} {'⚠️ NON-ZERO!' if frozen_grad > 1e-6 else '✓'}")
                print(f"  - Trainable[new]: {trainable_grad:12.6f}")

            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Gradient check error: {e}")

    def _check_trainable_status(self):
        """Report the requires_grad state of partially frozen layers."""
        print(f"\n{'='*60}")
        print(f"Trainable Status Check at Step {self.trainer.global_step}")
        print(f"{'='*60}")

        for name in ['body_projection', 'hand_projection', 'motion_projection', 'linear']:
            if hasattr(self.lm, name):
                layer = getattr(self.lm, name)
                if isinstance(layer, torch.nn.Sequential):
                    layer = layer[0]

                if hasattr(layer, 'weight'):
                    print(f"✓ {name:20s} requires_grad: {layer.weight.requires_grad}")

        try:
            unwrapped = self.lm.base_model.model if hasattr(self.lm, 'base_model') else self.lm
            old_size = self.old_vocab_size

            embed_layer = unwrapped.model.embed_tokens
            print(f"✓ embed_tokens requires_grad: {embed_layer.weight.requires_grad}")
            print(f"  (Only indices {old_size}+ should update)")

            head_layer = unwrapped.lm_head
            print(f"✓ lm_head requires_grad: {head_layer.weight.requires_grad}")
            print(f"  (Only indices {old_size}+ should update)")

        except Exception as e:
            print(f"Error: {e}")

        print(f"{'='*60}\n")

    def _decode_interleaved_motion(self, interleaved_codes, device=None):
        """Decode motion tokens from interleaved or sequential format.
        Handles SplitBodyHandVQVAE (interleaved [b,l,r,b,l,r,...]),
        UnifiedSplitBodyHandVQVAE (sequential [body_all, lh_all, rh_all]),
        and the plain single-codebook VQVae (single stream, no split).
        """
        if device is None:
            device = self.device
        if isinstance(self.vae, UnifiedSplitBodyHandVQVAE):
            return self.vae.decode(interleaved_codes)
        elif isinstance(self.vae, SplitBodyHandVQVAE):
            if interleaved_codes.dim() == 1:
                interleaved_codes = interleaved_codes.unsqueeze(0)
            B, total_len = interleaved_codes.shape

            # Ensure total_len is divisible by 3 by trimming extra tokens
            # This handles cases where generation doesn't perfectly align with 3-part structure
            if total_len % 3 != 0:
                # Trim to the nearest multiple of 3
                new_len = (total_len // 3) * 3
                interleaved_codes = interleaved_codes[:, :new_len]
                total_len = new_len

            seq_len = total_len // 3
            reshaped_codes = interleaved_codes.view(B, seq_len, 3)
            body_codes = reshaped_codes[..., 0]
            lh_codes = reshaped_codes[..., 1]
            rh_codes = reshaped_codes[..., 2]
            clamped_body = torch.clamp(body_codes, 0, self.vae.body_quantizer.nb_code - 1).to(device)
            clamped_lh = torch.clamp(lh_codes, 0, self.vae.left_hand_quantizer.nb_code - 1).to(device)
            clamped_rh = torch.clamp(rh_codes, 0, self.vae.right_hand_quantizer.nb_code - 1).to(device)
            return self.vae.decode(clamped_body, clamped_lh, clamped_rh)
        else:
            # Plain single-codebook VQVae: decode the flat stream directly.
            if interleaved_codes.dim() == 1:
                interleaved_codes = interleaved_codes.unsqueeze(0)
            codes = torch.clamp(interleaved_codes, 0,
                                self.vae.quantizer.nb_code - 1).to(device)
            return self.vae.decode(codes)

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        file_names =  batch["file_name"]
        lengths = batch["length"]
        tasks = None

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        min_len = lengths.copy()
        nocut_min_len = lengths.copy()
        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               task="t2m",
                                               tasks=tasks)
        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        nocut_feats_rst = torch.zeros(feats_ref.shape[0], 100, feats_ref.shape[2],
                                       dtype=feats_ref.dtype,
                                       device=feats_ref.device)
        vae_device = next(self.vae.parameters()).device

        batch_body_codes = []
        batch_lh_codes = []
        batch_rh_codes = []
        gt_batch_body_codes = []
        gt_batch_lh_codes = []
        gt_batch_rh_codes = []
        # Single-codebook VQVae keeps the whole stream here for the PD/GT print.
        pred_codes_stream = []
        gt_codes_stream = []

        if outputs == None:
            outputs = [[i] for i in range(len(texts))]
        for i in range(len(texts)):
            if len(outputs[i]) > 1:
                interleaved_codes = outputs[i]
                if interleaved_codes.dim() == 1:
                    interleaved_codes = interleaved_codes.unsqueeze(0)

                motion = self._decode_interleaved_motion(interleaved_codes, vae_device)

                if isinstance(self.vae, SplitBodyHandVQVAE):
                    B, total_len = interleaved_codes.shape

                    # Ensure total_len is divisible by 3 (same check as in _decode_interleaved_motion)
                    if total_len % 3 != 0:
                        new_len = (total_len // 3) * 3
                        interleaved_codes = interleaved_codes[:, :new_len]
                        total_len = new_len

                    seq_len = total_len // 3
                    reshaped_codes = interleaved_codes.view(B, seq_len, 3)
                    body_codes = reshaped_codes[..., 0]
                    left_hand_codes = reshaped_codes[..., 1]
                    right_hand_codes = reshaped_codes[..., 2]
                    batch_body_codes.append(body_codes.squeeze())
                    batch_lh_codes.append(left_hand_codes.squeeze())
                    batch_rh_codes.append(right_hand_codes.squeeze())
                else:
                    pred_codes_stream.append(interleaved_codes.squeeze(0))
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...]).to(vae_device)

            min_len[i] = min(motion.shape[1], lengths[i])
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

            nocut_min_len[i] = min(motion.shape[1], 100)
            nocut_feats_rst[i:i + 1, :nocut_min_len[i], ...] = motion[:, :nocut_min_len[i]]

            t_feats_ref = feats_ref[i:i + 1,0:lengths[i]]
            gt_interleaved_codes, _ = self.vae.encode(t_feats_ref)
            if gt_interleaved_codes.dim() == 1:
                gt_interleaved_codes = gt_interleaved_codes.unsqueeze(0)
            B, gt_total_len = gt_interleaved_codes.shape
            if isinstance(self.vae, SplitBodyHandVQVAE):
                # Ensure gt_total_len is divisible by 3 (safety check)
                if gt_total_len % 3 != 0:
                    new_len = (gt_total_len // 3) * 3
                    gt_interleaved_codes = gt_interleaved_codes[:, :new_len]
                    gt_total_len = new_len

                seq_len = gt_total_len // 3
                gt_reshaped_codes = gt_interleaved_codes.view(B, seq_len, 3)
                gt_body_codes = gt_reshaped_codes[..., 0]
                gt_left_hand_codes = gt_reshaped_codes[..., 1]
                gt_right_hand_codes = gt_reshaped_codes[..., 2]
                gt_batch_body_codes.append(gt_body_codes.squeeze())
                gt_batch_lh_codes.append(gt_left_hand_codes.squeeze())
                gt_batch_rh_codes.append(gt_right_hand_codes.squeeze())
            else:
                gt_codes_stream.append(gt_interleaved_codes.squeeze(0))

        try:
            print("="*50)
            if isinstance(self.vae, SplitBodyHandVQVAE):
                print("PD: ", batch_lh_codes[0])
                print("GT: ", gt_batch_lh_codes[0])
            else:
                print("PD: ", pred_codes_stream[0])
                print("GT: ", gt_codes_stream[0])
        except:
            pass

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)


        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        part_joints_ref = joints_ref[0].cpu().numpy()
        cut_joints_rst = joints_rst[0].cpu().numpy()

        # Return reconstruction set (callers can save joints via TEST.SAVE_PREDICTIONS)
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len,
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        motion_token = batch["tokens"]
        texts = batch["text"]

        lengths = batch["length"]
        file_names =  batch["file_name"]

        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            t_file_names = file_names[i]
            t_feats_ref = feats_ref[i:i + 1,0:lengths[i]]
            motion_token, _ = self.vae.encode(t_feats_ref)
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')
        print("="*50)
        print("PD: ", outputs[0])
        print("GT: ", texts[0])

        rs_set = {
            "m_ref": feats_ref,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
        }

        return rs_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        gloss_embeddings_list = batch.get("GLOSS_emb", None)
        if gloss_embeddings_list and all(emb is not None for emb in gloss_embeddings_list):
            gloss_embeddings = torch.stack(gloss_embeddings_list, dim=0).to(feats_ref.device)
        else:
            gloss_embeddings = None

        feats_rst, loss_commit, perplexity, loss_semantic = self.vae(
                features=feats_ref,
                gloss_embeddings=gloss_embeddings,
                lengths=lengths,
            )

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "loss_semantic": loss_semantic,
            "length": lengths
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        feats_rst = torch.zeros_like(feats_ref)
        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx, dataloader_idx=0):
        loss = None
        metrics_obj = self.metrics_test if (split == "val" and dataloader_idx == 1) else self.metrics

        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict

                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(metrics_obj,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_pretrain", "lm_rl"
                        ]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(metrics_obj, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                            gt_texts=batch["text"],
                            joints_rst = rs_set["joints_rst"],
                            joints_ref = rs_set["joints_ref"]
                        )
                    elif metric == "UncondMetrics":
                        getattr(metrics_obj, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(metrics_obj,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(metrics_obj,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        getattr(metrics_obj,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_rl"
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
                for metric in metrics_dicts:
                    if metric == "M2TMetrics":
                        getattr(metrics_obj, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["text"],
                            lengths=rs_set['length'],
                            word_embs=batch["word_embs"],
                            pos_ohot=batch["pos_ohot"],
                            text_lengths=batch["text_len"],
                        )

        if split in ["test"]:
            if self.hparams.task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set[
                    "joints_ref"]
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"]

        return loss
