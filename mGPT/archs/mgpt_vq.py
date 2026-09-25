# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from mGPT.archs.tools.htsa_loss import compute_cosine_semantic_loss
from mGPT.archs.tools.quantize_cnn import QuantizeEMAReset
from mGPT.archs.tools.resnet import Resnet1D
from mGPT.archs.tools.tcn_layer import Temporal_Encoder, Temporal_Decoder


class SplitBodyHandVQVAE_MULTI(nn.Module):
    def __init__(self,
                 full_feature_dim: int = 230,
                 body_hand_feature_dim: int = 220,
                 root_dim: int = 4,
                 num_joints: int = 73,
                 vae_args_body: dict = {},
                 vae_args_hand: dict = {},
                 use_my_encdec: bool = False,
                 use_mlp: bool = False,
                 share_proj: bool = False,
                 gloss_loss: bool = False,
                 **kwargs):
        super().__init__()

        self.use_my_encdec = use_my_encdec
        self.use_mlp = use_mlp
        self.share_proj = share_proj
        self.gloss_loss = gloss_loss

        self.full_feature_dim = full_feature_dim
        self.body_hand_feature_dim = body_hand_feature_dim
        self.root_dim = root_dim
        self.num_joints = num_joints

        # The representation uses local positions without velocity features.
        self.local_feat_dim = (self.num_joints - 1) * 3  # 72 * 3 = 216

        # Body: root(4) + body_local(63+33=96) = 100
        self.body_dim = self.root_dim + 63 + 33  # 100

        # Hand: left_hand_local(45+15=60) + right_hand_local(45+15=60) = 120
        self.hand_dim = 120
        self.left_hand_dim = self.hand_dim // 2  # 60
        self.right_hand_dim = self.hand_dim // 2  # 60

        self.face_dim = full_feature_dim - body_hand_feature_dim  # 230-220=10
        self.body_face_dim = self.body_dim + self.face_dim  # 100+10=110

        # --- Instantiate the three VQ-VAEs ---
        self.body_vae = VQVae(nfeats=self.body_face_dim, **vae_args_body)
        self.left_hand_vae = VQVae(nfeats=self.left_hand_dim, **vae_args_hand)
        self.right_hand_vae = VQVae(nfeats=self.right_hand_dim, **vae_args_hand)

        # Indices within the 216-D local-feature segment.
        self.L_BODY1_END = 63
        self.L_HAND1_START = self.L_BODY1_END
        self.L_HAND1_MID = self.L_HAND1_START + 45  # 63+45=108
        self.L_HAND1_END = self.L_HAND1_START + 90  # 63+90=153

        self.L_BODY2_START = self.L_HAND1_END  # 153
        self.L_BODY2_END = self.L_BODY2_START + 33  # 153+33=186

        self.L_HAND2_START = self.L_BODY2_END  # 186
        self.L_HAND2_MID = self.L_HAND2_START + 15  # 186+15=201
        self.L_HAND2_END = self.L_HAND2_START + 30  # 186+30=216

    def _split_features(self, features: Tensor):
        """Splits the full feature tensor into body, left hand, right hand, and face parts."""

        features_bodyhand = features[..., :self.body_hand_feature_dim]  # [..., 220]
        features_face = features[..., self.body_hand_feature_dim:]  # [..., 10]

        root = features_bodyhand[..., :self.root_dim]  # [..., 4]
        local_features = features_bodyhand[..., self.root_dim:]  # [..., 216]

        # --- Extract Body Features (root + body local) ---
        body_features = torch.cat((
            root,  # 4
            local_features[..., :self.L_BODY1_END],  # 63
            local_features[..., self.L_BODY2_START:self.L_BODY2_END]  # 33
        ), dim=-1)  # Total: 100

        # --- Extract Left Hand Features ---
        left_hand_features = torch.cat((
            local_features[..., self.L_HAND1_START:self.L_HAND1_MID],  # 45
            local_features[..., self.L_HAND2_START:self.L_HAND2_MID]   # 15
        ), dim=-1)  # Total: 60

        # --- Extract Right Hand Features ---
        right_hand_features = torch.cat((
            local_features[..., self.L_HAND1_MID:self.L_HAND1_END],  # 45
            local_features[..., self.L_HAND2_MID:self.L_HAND2_END]   # 15
        ), dim=-1)  # Total: 60

        return body_features, left_hand_features, right_hand_features, features_face

    def _reassemble_features(self, body_rst: Tensor, left_hand_rst: Tensor,
                            right_hand_rst: Tensor, face_rst: Tensor) -> Tensor:
        """Reassembles the reconstructed parts into a full tensor."""

        batch_size, seq_len, _ = body_rst.shape
        rst_features = torch.zeros(batch_size, seq_len, self.full_feature_dim,
                                   device=body_rst.device)

        # --- Unpack Body Features ---
        root_rst = body_rst[..., :self.root_dim]  # 4
        body_local_rst1 = body_rst[..., self.root_dim:self.root_dim + 63]  # 63
        body_local_rst2 = body_rst[..., self.root_dim + 63:]  # 33

        # --- Unpack Left and Right Hand Features ---
        left_local1 = left_hand_rst[..., :45]   # 45
        left_local2 = left_hand_rst[..., 45:60]  # 15

        right_local1 = right_hand_rst[..., :45]   # 45
        right_local2 = right_hand_rst[..., 45:60]  # 15

        # --- Fill the reassembled tensor ---
        # Place root
        rst_features[..., :self.root_dim] = root_rst

        # Place local features (starting from index 4)
        local_start_idx = self.root_dim
        rst_features[..., local_start_idx:local_start_idx + self.L_BODY1_END] = body_local_rst1
        rst_features[..., local_start_idx + self.L_HAND1_START:local_start_idx + self.L_HAND1_MID] = left_local1
        rst_features[..., local_start_idx + self.L_HAND1_MID:local_start_idx + self.L_HAND1_END] = right_local1
        rst_features[..., local_start_idx + self.L_BODY2_START:local_start_idx + self.L_BODY2_END] = body_local_rst2
        rst_features[..., local_start_idx + self.L_HAND2_START:local_start_idx + self.L_HAND2_MID] = left_local2
        rst_features[..., local_start_idx + self.L_HAND2_MID:local_start_idx + self.L_HAND2_END] = right_local2

        # Place face features
        rst_features[..., self.body_hand_feature_dim:] = face_rst

        return rst_features

    def forward(self, features: Tensor, gloss_embeddings: torch.Tensor = None, lengths: list = None):
        gt_body, gt_left_hand, gt_right_hand, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        rst_body_face, loss_commit_body, perplexity_body = self.body_vae(gt_body_face)
        rst_left_hand, loss_commit_left, perplexity_left = self.left_hand_vae(gt_left_hand)
        rst_right_hand, loss_commit_right, perplexity_right = self.right_hand_vae(gt_right_hand)

        rst_body = rst_body_face[..., :self.body_dim]
        rst_face = rst_body_face[..., self.body_dim:]
        feats_rst = self._reassemble_features(rst_body, rst_left_hand, rst_right_hand, rst_face)

        loss_commit = loss_commit_body + loss_commit_left + loss_commit_right
        perplexity = (perplexity_body + perplexity_left + perplexity_right)
        loss_semantic = torch.tensor(0.0)

        return feats_rst, loss_commit, perplexity, loss_semantic

    def encode(self, features: Tensor):
        """Encodes features into separate body, left hand, and right hand code indices.

        Returns:
            interleaved_codes: Tensor of shape [1, seq_len*3], codes interleaved as [b0,l0,r0,b1,l1,r1,...]
            None: For compatibility
        """
        gt_body, gt_left_hand, gt_right_hand, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        N, T, _ = features.shape

        # Encode using the VQ-VAEs
        body_codes, _ = self.body_vae.encode(gt_body_face)
        left_hand_codes, _ = self.left_hand_vae.encode(gt_left_hand)
        right_hand_codes, _ = self.right_hand_vae.encode(gt_right_hand)

        # Squeeze to remove batch dimension if needed
        # Assuming codes are shape [N, seq_len] or [N, seq_len, ...], we want [seq_len]
        body_codes = body_codes.squeeze(0)  # [seq_len] or [seq_len, ...]
        left_hand_codes = left_hand_codes.squeeze(0)
        right_hand_codes = right_hand_codes.squeeze(0)

        # Handle multi-dimensional codes by flattening if necessary
        if body_codes.dim() > 1:
            # If codes have additional dimensions, take the first or flatten
            # Adjust based on your VQVae output structure
            body_codes = body_codes.reshape(body_codes.shape[0], -1)[:, 0]
        if left_hand_codes.dim() > 1:
            left_hand_codes = left_hand_codes.reshape(left_hand_codes.shape[0], -1)[:, 0]
        if right_hand_codes.dim() > 1:
            right_hand_codes = right_hand_codes.reshape(right_hand_codes.shape[0], -1)[:, 0]

        seq_len = body_codes.shape[0]

        # Interleave codes: [b0, l0, r0, b1, l1, r1, ...]
        interleaved_codes = []
        for i in range(seq_len):
            interleaved_codes.append(body_codes[i])       # bi
            interleaved_codes.append(left_hand_codes[i])  # li
            interleaved_codes.append(right_hand_codes[i]) # ri

        # Convert list to tensor, final shape [seq_len * 3]
        interleaved_codes_tensor = torch.stack(interleaved_codes, dim=0)

        return interleaved_codes_tensor.unsqueeze(0), None  # [1, seq_len*3]

    def decode(self, clamped_body_codes: Tensor, clamped_lh_codes: Tensor,
               clamped_rh_codes: Tensor) -> Tensor:

        rst_body_face = self.body_vae.decode(clamped_body_codes)
        rst_left_hand = self.left_hand_vae.decode(clamped_lh_codes)
        rst_right_hand = self.right_hand_vae.decode(clamped_rh_codes)

        rst_body = rst_body_face[..., :self.body_dim]
        rst_face = rst_body_face[..., self.body_dim:]

        feats_rst = self._reassemble_features(rst_body, rst_left_hand, rst_right_hand, rst_face)
        return feats_rst


class SplitBodyHandVQVAE(nn.Module):
    """Partitioned VQ-VAE for 4-D root, 216-D local, and 10-D face features."""
    def __init__(self,
                 full_feature_dim: int = 230,
                 body_hand_feature_dim: int = 220,
                 root_dim: int = 4,
                 num_joints: int = 73,
                 vae_args_body: dict = {},
                 vae_args_hand: dict = {},
                 hier_inject: str = 'hand2body',
                 use_my_encdec: bool = False,
                 use_mlp: bool = False,
                 share_proj: bool = False,
                 gloss_loss: bool = False,
                 **kwargs):
        super().__init__()

        assert hier_inject in ('hand2body', 'body2hand', 'none'), \
            f"hier_inject must be 'hand2body', 'body2hand', or 'none', got '{hier_inject}'"
        self.hier_inject = hier_inject
        self.use_my_encdec = use_my_encdec
        self.use_mlp = use_mlp
        self.share_proj = share_proj
        self.gloss_loss = gloss_loss

        self.full_feature_dim = full_feature_dim
        self.body_hand_feature_dim = body_hand_feature_dim
        self.root_dim = root_dim
        self.num_joints = num_joints

        # The representation uses local positions without velocity features.
        self.local_feat_dim = (self.num_joints - 1) * 3  # 72 * 3 = 216

        # Body: root(4) + body_local(63+33=96) = 100
        self.body_dim = self.root_dim + 63 + 33  # 100

        # Hand: left_hand_local(45+15=60) + right_hand_local(45+15=60) = 120
        self.hand_dim = 120
        self.left_hand_dim = self.hand_dim // 2  # 60
        self.right_hand_dim = self.hand_dim // 2  # 60

        self.face_dim = full_feature_dim - body_hand_feature_dim  # 230-220=10
        self.body_face_dim = self.body_dim + self.face_dim  # 100+10=110

        # Indices within the 216-D local-feature segment.
        self.L_BODY1_END = 63
        self.L_HAND1_START = self.L_BODY1_END
        self.L_HAND1_MID = self.L_HAND1_START + 45  # 63+45=108
        self.L_HAND1_END = self.L_HAND1_START + 90  # 63+90=153

        self.L_BODY2_START = self.L_HAND1_END  # 153
        self.L_BODY2_END = self.L_BODY2_START + 33  # 153+33=186

        self.L_HAND2_START = self.L_BODY2_END  # 186
        self.L_HAND2_MID = self.L_HAND2_START + 15  # 186+15=201
        self.L_HAND2_END = self.L_HAND2_START + 30  # 186+30=216

        # --- Encoders for each part ---

        if self.use_my_encdec:
            self.body_encoder = Temporal_Encoder(self.body_face_dim, **vae_args_body)
            self.left_hand_encoder = Temporal_Encoder(self.left_hand_dim, **vae_args_hand)
            self.right_hand_encoder = Temporal_Encoder(self.right_hand_dim, **vae_args_hand)
        else:
            self.body_encoder = Encoder(self.body_face_dim, **vae_args_body)
            self.left_hand_encoder = Encoder(self.left_hand_dim, **vae_args_hand)
            self.right_hand_encoder = Encoder(self.right_hand_dim, **vae_args_hand)

        # --- Quantizers for each part ---
        hand_code_dim = vae_args_hand.get('code_dim', 512)
        body_code_dim = vae_args_body.get('code_dim', 512)
        self.left_hand_quantizer = QuantizeEMAReset(vae_args_hand.get('code_num', 512), hand_code_dim, mu=0.99)
        self.right_hand_quantizer = QuantizeEMAReset(vae_args_hand.get('code_num', 512), hand_code_dim, mu=0.99)
        self.body_quantizer = QuantizeEMAReset(vae_args_body.get('code_num', 512), body_code_dim, mu=0.99)

        # --- Hierarchical Components ---
        if self.hier_inject == 'hand2body':
            # Inject quantized hand features before body quantization.
            self.hand_transform = nn.Conv1d(hand_code_dim * 2, hand_code_dim * 2, kernel_size=1)
            self.hierarchical_conv = nn.Conv1d(
                in_channels=(hand_code_dim * 2) + body_code_dim,
                out_channels=body_code_dim,
                kernel_size=3, padding=1
            )
        elif self.hier_inject == 'body2hand':
            # Inject quantized body features before hand quantization.
            self.body_transform = nn.Conv1d(body_code_dim, body_code_dim, kernel_size=1)
            self.hierarchical_conv_lh = nn.Conv1d(
                in_channels=body_code_dim + hand_code_dim,
                out_channels=hand_code_dim,
                kernel_size=3, padding=1
            )
            self.hierarchical_conv_rh = nn.Conv1d(
                in_channels=body_code_dim + hand_code_dim,
                out_channels=hand_code_dim,
                kernel_size=3, padding=1
            )
        else:  # 'none' — fully disentangled: body and hand quantize independently
            self.hand_transform = None
            self.hierarchical_conv = None
            self.body_transform = None
            self.hierarchical_conv_lh = None
            self.hierarchical_conv_rh = None

        # --- Unified Decoder ---
        total_code_dim = body_code_dim + hand_code_dim * 2
        decoder_width = vae_args_body.get('output_emb_width', 512)

        # A projection layer to create a unified latent space for the decoder
        self.decoder_input_proj = nn.Conv1d(total_code_dim, decoder_width, kernel_size=1)

        if self.use_my_encdec:
            self.decoder = Temporal_Decoder(self.full_feature_dim, **vae_args_body)
        else:
            self.decoder = Decoder(
                self.full_feature_dim,
                output_emb_width=decoder_width,
                down_t=vae_args_body.get('down_t', 3),
                stride_t=vae_args_body.get('stride_t', 2),
                width=vae_args_body.get('width', 512),
                depth=vae_args_body.get('depth', 3),
                dilation_growth_rate=vae_args_body.get('dilation_growth_rate', 3),
                activation=vae_args_body.get('activation', 'relu'),
                norm=vae_args_body.get('norm', None)
            )

        # Semantic bridge
        llm_hidden_dim = 2048
        codebook_dim = vae_args_body.get('code_dim', 512) # 1024

        if self.use_mlp:
            if self.share_proj:
                self.proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.proj_down = nn.Linear(llm_hidden_dim, codebook_dim)
            else:
                self.body_proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.lh_proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.rh_proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.body_proj_down = nn.Linear(llm_hidden_dim, codebook_dim)
                self.lh_proj_down = nn.Linear(llm_hidden_dim, codebook_dim)
                self.rh_proj_down = nn.Linear(llm_hidden_dim, codebook_dim)


    def _split_features(self, features: Tensor):
        """
        Splits the full feature tensor into body, left hand, right hand, and face parts.
        Input format: root(4) + local_features(216) + face(10) = 230
        """
        features_bodyhand = features[..., :self.body_hand_feature_dim]  # [..., 220]
        features_face = features[..., self.body_hand_feature_dim:]  # [..., 10]

        root = features_bodyhand[..., :self.root_dim]  # [..., 4]
        local_features = features_bodyhand[..., self.root_dim:]  # [..., 216]

        # --- Extract Body Features (root + body local) ---
        body_features = torch.cat((
            root,  # 4
            local_features[..., :self.L_BODY1_END],  # 63
            local_features[..., self.L_BODY2_START:self.L_BODY2_END]  # 33
        ), dim=-1)  # Total: 100

        # --- Extract Left Hand Features ---
        left_hand_features = torch.cat((
            local_features[..., self.L_HAND1_START:self.L_HAND1_MID],  # 45
            local_features[..., self.L_HAND2_START:self.L_HAND2_MID]   # 15
        ), dim=-1)  # Total: 60

        # --- Extract Right Hand Features ---
        right_hand_features = torch.cat((
            local_features[..., self.L_HAND1_MID:self.L_HAND1_END],  # 45
            local_features[..., self.L_HAND2_MID:self.L_HAND2_END]   # 15
        ), dim=-1)  # Total: 60

        return body_features, left_hand_features, right_hand_features, features_face

    def _reassemble_features(self, body_rst: Tensor, left_hand_rst: Tensor,
                            right_hand_rst: Tensor, face_rst: Tensor) -> Tensor:
        """Reassembles the reconstructed parts into a full tensor."""

        batch_size, seq_len, _ = body_rst.shape
        rst_features = torch.zeros(batch_size, seq_len, self.full_feature_dim,
                                   device=body_rst.device)

        # --- Unpack Body Features ---
        root_rst = body_rst[..., :self.root_dim]  # 4
        body_local_rst1 = body_rst[..., self.root_dim:self.root_dim + 63]  # 63
        body_local_rst2 = body_rst[..., self.root_dim + 63:]  # 33

        # --- Unpack Left and Right Hand Features ---
        left_local1 = left_hand_rst[..., :45]   # 45
        left_local2 = left_hand_rst[..., 45:60]  # 15

        right_local1 = right_hand_rst[..., :45]   # 45
        right_local2 = right_hand_rst[..., 45:60]  # 15

        # --- Fill the reassembled tensor ---
        # Place root
        rst_features[..., :self.root_dim] = root_rst

        # Place local features (starting from index 4)
        local_start_idx = self.root_dim
        rst_features[..., local_start_idx:local_start_idx + self.L_BODY1_END] = body_local_rst1
        rst_features[..., local_start_idx + self.L_HAND1_START:local_start_idx + self.L_HAND1_MID] = left_local1
        rst_features[..., local_start_idx + self.L_HAND1_MID:local_start_idx + self.L_HAND1_END] = right_local1
        rst_features[..., local_start_idx + self.L_BODY2_START:local_start_idx + self.L_BODY2_END] = body_local_rst2
        rst_features[..., local_start_idx + self.L_HAND2_START:local_start_idx + self.L_HAND2_MID] = left_local2
        rst_features[..., local_start_idx + self.L_HAND2_MID:local_start_idx + self.L_HAND2_END] = right_local2

        # Place face features
        rst_features[..., self.body_hand_feature_dim:] = face_rst

        return rst_features

    def preprocess(self, x: Tensor) -> Tensor:
        # (bs, T, Jx3) -> (bs, Jx3, T)
        return x.permute(0, 2, 1)

    def postprocess(self, x: Tensor) -> Tensor:
        # (bs, Jx3, T) -> (bs, T, Jx3)
        return x.permute(0, 2, 1)

    def forward(self, features: Tensor, gloss_embeddings: torch.Tensor = None, lengths: list = None):

        # 1. Split features into anatomical parts
        gt_body, gt_left_hand, gt_right_hand, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        # 2. Preprocess (permute) for 1D convolution
        body_face_in = self.preprocess(gt_body_face)
        left_hand_in = self.preprocess(gt_left_hand)
        right_hand_in = self.preprocess(gt_right_hand)

        # 3. Encode all parts to get latent representations
        z_body = self.body_encoder(body_face_in)
        z_lh = self.left_hand_encoder(left_hand_in)
        z_rh = self.right_hand_encoder(right_hand_in)

        # 4. Hierarchical quantization (direction depends on hier_inject)
        if self.hier_inject == 'hand2body':
            z_hat_lh, loss_commit_left, perplexity_left = self.left_hand_quantizer(z_lh)
            z_hat_rh, loss_commit_right, perplexity_right = self.right_hand_quantizer(z_rh)

            z_hat_hands_deq = torch.cat([z_hat_lh, z_hat_rh], dim=1)
            z_hands_transformed = self.hand_transform(z_hat_hands_deq)
            hier_input = torch.cat([z_hands_transformed, z_body], dim=1)
            hier_conv_out = self.hierarchical_conv(hier_input)
            z_hat_body, loss_commit_body, perplexity_body = self.body_quantizer(hier_conv_out)
        elif self.hier_inject == 'body2hand':
            z_hat_body, loss_commit_body, perplexity_body = self.body_quantizer(z_body)

            z_body_transformed = self.body_transform(z_hat_body)
            hier_input_lh = torch.cat([z_body_transformed, z_lh], dim=1)
            hier_input_rh = torch.cat([z_body_transformed, z_rh], dim=1)
            z_hat_lh, loss_commit_left, perplexity_left = self.left_hand_quantizer(self.hierarchical_conv_lh(hier_input_lh))
            z_hat_rh, loss_commit_right, perplexity_right = self.right_hand_quantizer(self.hierarchical_conv_rh(hier_input_rh))
        else:  # 'none' — body and hands quantize independently with no cross-injection
            z_hat_body, loss_commit_body, perplexity_body = self.body_quantizer(z_body)
            z_hat_lh, loss_commit_left, perplexity_left = self.left_hand_quantizer(z_lh)
            z_hat_rh, loss_commit_right, perplexity_right = self.right_hand_quantizer(z_rh)

        # 5. Unified Decode
        feats_rst, loss_semantic = self.decode_from_latents(
            z_hat_body, z_hat_lh, z_hat_rh, gloss_embeddings
        )

        # 5b. Length match: causal decoders may produce a temporal dim that
        # differs from the input (right-side padding/chomp asymmetry). Crop
        # or pad the output so it is bit-exact to ``features`` along T.
        if feats_rst.shape[1] != features.shape[1]:
            target_T = features.shape[1]
            if feats_rst.shape[1] >= target_T:
                feats_rst = feats_rst[:, :target_T, :]
            else:
                pad_T = target_T - feats_rst.shape[1]
                feats_rst = torch.cat(
                    [feats_rst, feats_rst[:, -1:, :].expand(-1, pad_T, -1)],
                    dim=1,
                )

        # 6. Combine losses and average perplexities
        loss_commit = loss_commit_body + loss_commit_left + loss_commit_right
        perplexity = (perplexity_body + perplexity_left + perplexity_right) / 3.0

        return feats_rst, loss_commit, perplexity, loss_semantic

    def decode_from_latents(self, z_hat_body: Tensor, z_hat_lh: Tensor, z_hat_rh: Tensor,
                           gloss_embeddings: torch.Tensor = None) -> Tensor:
        """Helper function to decode from quantized latent embeddings."""
        loss_semantic = torch.tensor(0.0, device=z_hat_body.device)

        if self.use_mlp:
            if self.share_proj:
                middle_z_hat_body = self.proj_up(z_hat_body.permute(0, 2, 1))
                middle_z_hat_lh = self.proj_up(z_hat_lh.permute(0, 2, 1))
                middle_z_hat_rh = self.proj_up(z_hat_rh.permute(0, 2, 1))
            else:
                middle_z_hat_body = self.body_proj_up(z_hat_body.permute(0, 2, 1))
                middle_z_hat_lh = self.lh_proj_up(z_hat_lh.permute(0, 2, 1))
                middle_z_hat_rh = self.rh_proj_up(z_hat_rh.permute(0, 2, 1))

            if gloss_embeddings is not None and self.gloss_loss:
                loss_body = compute_cosine_semantic_loss(middle_z_hat_body, gloss_embeddings)
                loss_lh = compute_cosine_semantic_loss(middle_z_hat_lh, gloss_embeddings)
                loss_rh = compute_cosine_semantic_loss(middle_z_hat_rh, gloss_embeddings)
                loss_semantic = (loss_body + loss_lh + loss_rh) / 3.0

            if self.share_proj:
                recover_z_hat_body = self.proj_down(middle_z_hat_body).permute(0, 2, 1)
                recover_z_hat_lh = self.proj_down(middle_z_hat_lh).permute(0, 2, 1)
                recover_z_hat_rh = self.proj_down(middle_z_hat_rh).permute(0, 2, 1)
            else:
                recover_z_hat_body = self.body_proj_down(middle_z_hat_body).permute(0, 2, 1)
                recover_z_hat_lh = self.lh_proj_down(middle_z_hat_lh).permute(0, 2, 1)
                recover_z_hat_rh = self.rh_proj_down(middle_z_hat_rh).permute(0, 2, 1)
        else:
            recover_z_hat_body = z_hat_body
            recover_z_hat_lh = z_hat_lh
            recover_z_hat_rh = z_hat_rh

        z_hat_all = torch.cat([recover_z_hat_body, recover_z_hat_lh, recover_z_hat_rh], dim=1)
        decoder_input = self.decoder_input_proj(z_hat_all)
        x_decoder = self.decoder(decoder_input)
        feats_rst = self.postprocess(x_decoder)
        return feats_rst, loss_semantic

    def encode(self, features: Tensor):
        """Encodes features into separate body, left hand, and right hand code indices."""
        gt_body, gt_left_hand, gt_right_hand, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        N, T, _ = features.shape

        body_face_in = self.preprocess(gt_body_face)
        left_hand_in = self.preprocess(gt_left_hand)
        right_hand_in = self.preprocess(gt_right_hand)
        z_body = self.body_encoder(body_face_in)
        z_lh = self.left_hand_encoder(left_hand_in)
        z_rh = self.right_hand_encoder(right_hand_in)

        if self.hier_inject == 'hand2body':
            z_hat_lh, _, _ = self.left_hand_quantizer(z_lh)
            z_hat_rh, _, _ = self.right_hand_quantizer(z_rh)

            left_hand_codes = self.left_hand_quantizer.quantize(
                z_lh.permute(0, 2, 1).contiguous().view(-1, z_lh.shape[1])
            ).view(N, -1)
            right_hand_codes = self.right_hand_quantizer.quantize(
                z_rh.permute(0, 2, 1).contiguous().view(-1, z_rh.shape[1])
            ).view(N, -1)

            z_hat_hands_deq = torch.cat([z_hat_lh, z_hat_rh], dim=1)
            z_hands_transformed = self.hand_transform(z_hat_hands_deq)
            hier_input = torch.cat([z_hands_transformed, z_body], dim=1)
            hier_conv_out = self.hierarchical_conv(hier_input)

            body_codes = self.body_quantizer.quantize(
                hier_conv_out.permute(0, 2, 1).contiguous().view(-1, hier_conv_out.shape[1])
            ).view(N, -1)
        elif self.hier_inject == 'body2hand':
            z_hat_body, _, _ = self.body_quantizer(z_body)

            body_codes = self.body_quantizer.quantize(
                z_body.permute(0, 2, 1).contiguous().view(-1, z_body.shape[1])
            ).view(N, -1)

            z_body_transformed = self.body_transform(z_hat_body)
            hier_input_lh = torch.cat([z_body_transformed, z_lh], dim=1)
            hier_input_rh = torch.cat([z_body_transformed, z_rh], dim=1)

            left_hand_codes = self.left_hand_quantizer.quantize(
                self.hierarchical_conv_lh(hier_input_lh).permute(0, 2, 1).contiguous().view(-1, z_lh.shape[1])
            ).view(N, -1)
            right_hand_codes = self.right_hand_quantizer.quantize(
                self.hierarchical_conv_rh(hier_input_rh).permute(0, 2, 1).contiguous().view(-1, z_rh.shape[1])
            ).view(N, -1)
        else:  # 'none' — body and hand codebooks are fully independent
            body_codes = self.body_quantizer.quantize(
                z_body.permute(0, 2, 1).contiguous().view(-1, z_body.shape[1])
            ).view(N, -1)
            left_hand_codes = self.left_hand_quantizer.quantize(
                z_lh.permute(0, 2, 1).contiguous().view(-1, z_lh.shape[1])
            ).view(N, -1)
            right_hand_codes = self.right_hand_quantizer.quantize(
                z_rh.permute(0, 2, 1).contiguous().view(-1, z_rh.shape[1])
            ).view(N, -1)

        body_codes = body_codes.squeeze(0)
        left_hand_codes = left_hand_codes.squeeze(0)
        right_hand_codes = right_hand_codes.squeeze(0)

        seq_len = body_codes.shape[0]
        interleaved_codes = []
        for i in range(seq_len):
            interleaved_codes.append(body_codes[i])
            interleaved_codes.append(left_hand_codes[i])
            interleaved_codes.append(right_hand_codes[i])

        interleaved_codes_tensor = torch.stack(interleaved_codes, dim=0)
        return interleaved_codes_tensor.unsqueeze(0), None  # [1, seq_len*3]

    def decode(self, clamped_body_codes: Tensor, clamped_lh_codes: Tensor,
               clamped_rh_codes: Tensor) -> Tensor:
        """Decodes from code indices back to features."""
        codes = {
            "body": clamped_body_codes,
            "left_hand": clamped_lh_codes,
            "right_hand": clamped_rh_codes
        }

        z_hat_body = self.body_quantizer.dequantize(codes["body"])
        z_hat_lh = self.left_hand_quantizer.dequantize(codes["left_hand"])
        z_hat_rh = self.right_hand_quantizer.dequantize(codes["right_hand"])

        # Reshape to (B, C, T) format for convolution
        B = codes["body"].shape[0]
        body_code_dim = self.body_quantizer.code_dim
        hand_code_dim = self.left_hand_quantizer.code_dim

        z_hat_body = z_hat_body.view(B, -1, body_code_dim).permute(0, 2, 1).contiguous()
        z_hat_lh = z_hat_lh.view(B, -1, hand_code_dim).permute(0, 2, 1).contiguous()
        z_hat_rh = z_hat_rh.view(B, -1, hand_code_dim).permute(0, 2, 1).contiguous()

        # Use the unified decoding pipeline
        feats_rst, _ = self.decode_from_latents(z_hat_body, z_hat_lh, z_hat_rh)
        return feats_rst


class SplitBodyHandVQVAE_2code(nn.Module):
    def __init__(self,
                 full_feature_dim: int = 230,
                 body_hand_feature_dim: int = 220,
                 root_dim: int = 4,
                 num_joints: int = 73,
                 vae_args_body: dict = {},
                 vae_args_hand: dict = {},
                 use_my_encdec: bool = False,
                 use_mlp: bool = False,
                 share_proj: bool = False,
                 gloss_loss: bool = False,
                 **kwargs):
        super().__init__()

        self.use_my_encdec = use_my_encdec
        self.use_mlp = use_mlp
        self.share_proj = share_proj
        self.gloss_loss = gloss_loss

        self.full_feature_dim = full_feature_dim
        self.body_hand_feature_dim = body_hand_feature_dim
        self.root_dim = root_dim
        self.num_joints = num_joints

        # The representation uses local positions without velocity features.
        self.local_feat_dim = (self.num_joints - 1) * 3  # 72 * 3 = 216

        # Body: root(4) + body_local(63+33=96) = 100
        self.body_dim = self.root_dim + 63 + 33  # 100

        # Hand: left_hand_local(60) + right_hand_local(60) = 120
        self.hand_dim = 120
        self.left_hand_dim = self.hand_dim // 2
        self.right_hand_dim = self.hand_dim // 2

        self.face_dim = full_feature_dim - body_hand_feature_dim  # 230-220=10
        self.body_face_dim = self.body_dim + self.face_dim  # 100+10=110

        # Indices within the 216-D local-feature segment.
        self.L_BODY1_END = 63
        self.L_HAND1_START = self.L_BODY1_END
        self.L_HAND1_MID = self.L_HAND1_START + 45
        self.L_HAND1_END = self.L_HAND1_START + 90
        self.L_BODY2_START = self.L_HAND1_END
        self.L_BODY2_END = self.L_BODY2_START + 33
        self.L_HAND2_START = self.L_BODY2_END
        self.L_HAND2_MID = self.L_HAND2_START + 15
        self.L_HAND2_END = self.L_HAND2_START + 30

        # Encoders for the body and unified hand streams.
        if self.use_my_encdec:
            self.body_encoder = Temporal_Encoder(self.body_face_dim, **vae_args_body)
            self.hand_encoder = Temporal_Encoder(self.hand_dim, **vae_args_hand)
        else:
            self.body_encoder = Encoder(self.body_face_dim, **vae_args_body)
            self.hand_encoder = Encoder(self.hand_dim, **vae_args_hand)
        # Quantizers for the body and hand streams.
        hand_code_dim = vae_args_hand.get('code_dim', 512)
        body_code_dim = vae_args_body.get('code_dim', 512)

        self.hand_quantizer = QuantizeEMAReset(vae_args_hand.get('code_num', 512), hand_code_dim, mu=0.99)
        self.body_quantizer = QuantizeEMAReset(vae_args_body.get('code_num', 512), body_code_dim, mu=0.99)
        # Hierarchical fusion of dequantized hand and body latents.
        self.hand_transform = nn.Conv1d(hand_code_dim, hand_code_dim, kernel_size=1)

        self.hierarchical_conv = nn.Conv1d(
            in_channels=hand_code_dim + body_code_dim,
            out_channels=body_code_dim,
            kernel_size=3,
            padding=1
        )

        # Unified decoder.
        total_code_dim = body_code_dim + hand_code_dim
        decoder_width = vae_args_body.get('output_emb_width', 512)

        self.decoder_input_proj = nn.Conv1d(total_code_dim, decoder_width, kernel_size=1)

        if self.use_my_encdec:
            self.decoder = Temporal_Decoder(self.full_feature_dim, **vae_args_body)
        else:
            self.decoder = Decoder(
                self.full_feature_dim,
                output_emb_width=decoder_width,
                down_t=vae_args_body.get('down_t', 3),
                stride_t=vae_args_body.get('stride_t', 2),
                width=vae_args_body.get('width', 512),
                depth=vae_args_body.get('depth', 3),
                dilation_growth_rate=vae_args_body.get('dilation_growth_rate', 3),
                activation=vae_args_body.get('activation', 'relu'),
                norm=vae_args_body.get('norm', None)
            )

        # Semantic bridge.
        llm_hidden_dim = 2048
        codebook_dim = vae_args_body.get('code_dim', 512)

        if self.use_mlp:
            hidden_dim = codebook_dim
            if self.share_proj:
                self.proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.proj_down = nn.Linear(llm_hidden_dim, codebook_dim)
            else:
                self.body_proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.hand_proj_up = nn.Linear(codebook_dim, llm_hidden_dim)
                self.body_proj_down = nn.Linear(llm_hidden_dim, codebook_dim)
                self.hand_proj_down = nn.Linear(llm_hidden_dim, codebook_dim)

    def _split_features(self, features: Tensor):
        """
        Split the full feature tensor into body, unified hands, and face parts.
        """
        features_bodyhand = features[..., :self.body_hand_feature_dim]
        features_face = features[..., self.body_hand_feature_dim:]

        root = features_bodyhand[..., :self.root_dim]
        local_features = features_bodyhand[..., self.root_dim:]

        body_features = torch.cat((
            root,
            local_features[..., :self.L_BODY1_END],
            local_features[..., self.L_BODY2_START:self.L_BODY2_END]
        ), dim=-1)

        left_hand_features = torch.cat((
            local_features[..., self.L_HAND1_START:self.L_HAND1_MID],
            local_features[..., self.L_HAND2_START:self.L_HAND2_MID]
        ), dim=-1)

        right_hand_features = torch.cat((
            local_features[..., self.L_HAND1_MID:self.L_HAND1_END],
            local_features[..., self.L_HAND2_MID:self.L_HAND2_END]
        ), dim=-1)

        # Concatenate left and right hand features into a single stream.
        hand_features = torch.cat((left_hand_features, right_hand_features), dim=-1)

        return body_features, hand_features, features_face

    def _reassemble_features(self, body_rst: Tensor, hand_rst: Tensor, face_rst: Tensor) -> Tensor:
        """
        Reassemble body, unified-hand, and face features.
        """
        batch_size, seq_len, _ = body_rst.shape
        rst_features = torch.zeros(batch_size, seq_len, self.full_feature_dim, device=body_rst.device)

        root_rst = body_rst[..., :self.root_dim]
        body_local_rst1 = body_rst[..., self.root_dim:self.root_dim + 63]
        body_local_rst2 = body_rst[..., self.root_dim + 63:]

        # Split the unified hand reconstruction back into left and right.
        left_hand_rst = hand_rst[..., :self.left_hand_dim]
        right_hand_rst = hand_rst[..., self.left_hand_dim:]

        left_local1 = left_hand_rst[..., :45]
        left_local2 = left_hand_rst[..., 45:60]
        right_local1 = right_hand_rst[..., :45]
        right_local2 = right_hand_rst[..., 45:60]

        local_start_idx = self.root_dim
        rst_features[..., :local_start_idx] = root_rst
        rst_features[..., local_start_idx:local_start_idx + self.L_BODY1_END] = body_local_rst1
        rst_features[..., local_start_idx + self.L_HAND1_START:local_start_idx + self.L_HAND1_MID] = left_local1
        rst_features[..., local_start_idx + self.L_HAND1_MID:local_start_idx + self.L_HAND1_END] = right_local1
        rst_features[..., local_start_idx + self.L_BODY2_START:local_start_idx + self.L_BODY2_END] = body_local_rst2
        rst_features[..., local_start_idx + self.L_HAND2_START:local_start_idx + self.L_HAND2_MID] = left_local2
        rst_features[..., local_start_idx + self.L_HAND2_MID:local_start_idx + self.L_HAND2_END] = right_local2
        rst_features[..., self.body_hand_feature_dim:] = face_rst

        return rst_features

    def preprocess(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)

    def postprocess(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)

    def forward(self, features: Tensor, gloss_embeddings: torch.Tensor = None, lengths: list = None):
        gt_body, gt_hands, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        body_face_in = self.preprocess(gt_body_face)
        hands_in = self.preprocess(gt_hands)

        z_body = self.body_encoder(body_face_in)
        z_hand = self.hand_encoder(hands_in)

        z_hat_hand, loss_commit_hand, perplexity_hand = self.hand_quantizer(z_hand)

        z_hand_transformed = self.hand_transform(z_hat_hand)
        hier_input = torch.cat([z_hand_transformed, z_body], dim=1)
        hier_conv_out = self.hierarchical_conv(hier_input)
        z_hat_body, loss_commit_body, perplexity_body = self.body_quantizer(hier_conv_out)

        feats_rst, loss_semantic = self.decode_from_latents(
            z_hat_body, z_hat_hand, gloss_embeddings
        )

        loss_commit = loss_commit_body + loss_commit_hand
        perplexity = (perplexity_body + perplexity_hand) / 2.0

        return feats_rst, loss_commit, perplexity, loss_semantic

    def decode_from_latents(self, z_hat_body: Tensor, z_hat_hand: Tensor,
                            gloss_embeddings: torch.Tensor = None) -> Tensor:
        """Decode body and unified-hand latents."""
        loss_semantic = torch.tensor(0.0, device=z_hat_body.device)

        if self.use_mlp:
            if self.share_proj:
                middle_z_hat_body = self.proj_up(z_hat_body.permute(0, 2, 1))
                middle_z_hat_hand = self.proj_up(z_hat_hand.permute(0, 2, 1))
            else:
                middle_z_hat_body = self.body_proj_up(z_hat_body.permute(0, 2, 1))
                middle_z_hat_hand = self.hand_proj_up(z_hat_hand.permute(0, 2, 1))

            if gloss_embeddings is not None and self.gloss_loss:
                loss_body = compute_cosine_semantic_loss(middle_z_hat_body, gloss_embeddings)
                loss_hand = compute_cosine_semantic_loss(middle_z_hat_hand, gloss_embeddings)
                loss_semantic = (loss_body + loss_hand) / 2.0

            if self.share_proj:
                recover_z_hat_body = self.proj_down(middle_z_hat_body).permute(0, 2, 1)
                recover_z_hat_hand = self.proj_down(middle_z_hat_hand).permute(0, 2, 1)
            else:
                recover_z_hat_body = self.body_proj_down(middle_z_hat_body).permute(0, 2, 1)
                recover_z_hat_hand = self.hand_proj_down(middle_z_hat_hand).permute(0, 2, 1)
        else:
            recover_z_hat_body = z_hat_body
            recover_z_hat_hand = z_hat_hand

        z_hat_all = torch.cat([recover_z_hat_body, recover_z_hat_hand], dim=1)
        decoder_input = self.decoder_input_proj(z_hat_all)
        x_decoder = self.decoder(decoder_input)
        feats_rst = self.postprocess(x_decoder)
        return feats_rst, loss_semantic

    def encode(self, features: Tensor):
        """Encode features into interleaved body and hand code indices."""
        gt_body, gt_hands, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        N, T, _ = features.shape

        body_face_in = self.preprocess(gt_body_face)
        hands_in = self.preprocess(gt_hands)
        z_body = self.body_encoder(body_face_in)
        z_hand = self.hand_encoder(hands_in)

        # Quantize hand to get latent for hierarchy
        z_hat_hand, _, _ = self.hand_quantizer(z_hand)

        # Get hand code indices
        hand_codes = self.hand_quantizer.quantize(
            z_hand.permute(0, 2, 1).contiguous().view(-1, z_hand.shape[1])
        ).view(N, -1)

        # Hierarchical body quantization
        z_hand_transformed = self.hand_transform(z_hat_hand)
        hier_input = torch.cat([z_hand_transformed, z_body], dim=1)
        hier_conv_out = self.hierarchical_conv(hier_input)

        # Get body code indices
        body_codes = self.body_quantizer.quantize(
            hier_conv_out.permute(0, 2, 1).contiguous().view(-1, hier_conv_out.shape[1])
        ).view(N, -1)

        body_codes = body_codes.squeeze(0)
        hand_codes = hand_codes.squeeze(0)

        # Interleave body and hand codes.
        seq_len = body_codes.shape[0]
        interleaved_codes = []
        for i in range(seq_len):
            interleaved_codes.append(body_codes[i])
            interleaved_codes.append(hand_codes[i])

        interleaved_codes_tensor = torch.stack(interleaved_codes, dim=0)
        return interleaved_codes_tensor.unsqueeze(0), None  # [1, seq_len*2]

    def decode(self, clamped_body_codes: Tensor, clamped_hand_codes: Tensor) -> Tensor:
        """Decode body and hand code indices back to features."""
        z_hat_body = self.body_quantizer.dequantize(clamped_body_codes)
        z_hat_hand = self.hand_quantizer.dequantize(clamped_hand_codes)

        B = clamped_body_codes.shape[0]
        body_code_dim = self.body_quantizer.code_dim
        hand_code_dim = self.hand_quantizer.code_dim

        z_hat_body = z_hat_body.view(B, -1, body_code_dim).permute(0, 2, 1).contiguous()
        z_hat_hand = z_hat_hand.view(B, -1, hand_code_dim).permute(0, 2, 1).contiguous()

        feats_rst, _ = self.decode_from_latents(z_hat_body, z_hat_hand)
        return feats_rst
# endregion


class VQVae(nn.Module):

    def __init__(self,
                 nfeats=230,
                 vae_args_body: dict = {},
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = vae_args_body.get('code_dim', 512)

        self.encoder = Encoder(nfeats,
                               vae_args_body.get('output_emb_width', 512),
                               vae_args_body.get('down_t', 3),
                               vae_args_body.get('stride_t', 2),
                               vae_args_body.get('width', 512),
                               vae_args_body.get('depth', 3),
                               vae_args_body.get('dilation_growth_rate', 3),
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                            vae_args_body.get('output_emb_width', 512),
                            vae_args_body.get('down_t', 3),
                            vae_args_body.get('stride_t', 2),
                            vae_args_body.get('width', 512),
                            vae_args_body.get('depth', 3),
                            vae_args_body.get('dilation_growth_rate', 3),
                            activation=activation,
                            norm=norm)

        self.quantizer = QuantizeEMAReset(vae_args_body.get('code_num', 512), vae_args_body.get('code_dim', 512), mu=0.99)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor, gloss_embeddings: torch.Tensor = None, lengths: list = None):
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity, torch.tensor(0.0)

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


from mGPT.archs.tools.tcn_layer_causal import CausalTemporal_Encoder, CausalTemporal_Decoder


class CausalSplitBodyHandVQVAE(SplitBodyHandVQVAE):
    """
    Ablation: Same as SplitBodyHandVQVAE but with causal TCN encoder/decoder.
    Each position only depends on current and past inputs (no future context).
    """
    def __init__(self, **kwargs):
        kwargs['use_my_encdec'] = True
        super().__init__(**kwargs)

        vae_args_body = kwargs.get('vae_args_body', {})
        vae_args_hand = kwargs.get('vae_args_hand', {})

        self.body_encoder = CausalTemporal_Encoder(self.body_face_dim, **vae_args_body)
        self.left_hand_encoder = CausalTemporal_Encoder(self.left_hand_dim, **vae_args_hand)
        self.right_hand_encoder = CausalTemporal_Encoder(self.right_hand_dim, **vae_args_hand)
        self.decoder = CausalTemporal_Decoder(self.full_feature_dim, **vae_args_body)


class UnifiedSplitBodyHandVQVAE(SplitBodyHandVQVAE):
    """
    Ablation: Same VQVAE but encodes tokens in sequential order
    [body_all, lh_all, rh_all] instead of interleaved [b0,l0,r0,b1,l1,r1,...].
    Used by the single-head unified decoding LLM (MLM_SingleHead).
    """
    def encode(self, features: Tensor):
        gt_body, gt_left_hand, gt_right_hand, gt_face = self._split_features(features)
        gt_body_face = torch.cat([gt_body, gt_face], dim=-1)

        N, T, _ = features.shape

        body_face_in = self.preprocess(gt_body_face)
        left_hand_in = self.preprocess(gt_left_hand)
        right_hand_in = self.preprocess(gt_right_hand)

        z_body = self.body_encoder(body_face_in)
        z_lh = self.left_hand_encoder(left_hand_in)
        z_rh = self.right_hand_encoder(right_hand_in)

        if self.hier_inject == 'hand2body':
            z_hat_lh, _, _ = self.left_hand_quantizer(z_lh)
            z_hat_rh, _, _ = self.right_hand_quantizer(z_rh)
            left_hand_codes = self.left_hand_quantizer.quantize(
                z_lh.permute(0, 2, 1).contiguous().view(-1, z_lh.shape[1])
            ).view(N, -1)
            right_hand_codes = self.right_hand_quantizer.quantize(
                z_rh.permute(0, 2, 1).contiguous().view(-1, z_rh.shape[1])
            ).view(N, -1)
            z_hat_hands_deq = torch.cat([z_hat_lh, z_hat_rh], dim=1)
            z_hands_transformed = self.hand_transform(z_hat_hands_deq)
            hier_input = torch.cat([z_hands_transformed, z_body], dim=1)
            hier_conv_out = self.hierarchical_conv(hier_input)
            body_codes = self.body_quantizer.quantize(
                hier_conv_out.permute(0, 2, 1).contiguous().view(-1, hier_conv_out.shape[1])
            ).view(N, -1)
        else:
            z_hat_body, _, _ = self.body_quantizer(z_body)
            body_codes = self.body_quantizer.quantize(
                z_body.permute(0, 2, 1).contiguous().view(-1, z_body.shape[1])
            ).view(N, -1)
            z_body_transformed = self.body_transform(z_hat_body)
            hier_input_lh = torch.cat([z_body_transformed, z_lh], dim=1)
            hier_input_rh = torch.cat([z_body_transformed, z_rh], dim=1)
            left_hand_codes = self.left_hand_quantizer.quantize(
                self.hierarchical_conv_lh(hier_input_lh).permute(0, 2, 1).contiguous().view(-1, z_lh.shape[1])
            ).view(N, -1)
            right_hand_codes = self.right_hand_quantizer.quantize(
                self.hierarchical_conv_rh(hier_input_rh).permute(0, 2, 1).contiguous().view(-1, z_rh.shape[1])
            ).view(N, -1)

        body_codes = body_codes.squeeze(0)
        left_hand_codes = left_hand_codes.squeeze(0)
        right_hand_codes = right_hand_codes.squeeze(0)

        # Sequential: [body_all, lh_all, rh_all]
        sequential_codes = torch.cat([body_codes, left_hand_codes, right_hand_codes], dim=0)
        return sequential_codes.unsqueeze(0), None

    def decode(self, unified_codes: Tensor) -> Tensor:
        """Decode from sequential unified codes: [body_all, lh_all, rh_all]."""
        if unified_codes.dim() == 1:
            unified_codes = unified_codes.unsqueeze(0)

        B, total_len = unified_codes.shape
        assert total_len % 3 == 0, f"Total length {total_len} not divisible by 3"
        seq_len = total_len // 3

        body_codes = unified_codes[:, :seq_len]
        lh_codes = unified_codes[:, seq_len:2*seq_len]
        rh_codes = unified_codes[:, 2*seq_len:]

        return super().decode(body_codes, lh_codes, rh_codes)


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 code_num=None,code_dim=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
