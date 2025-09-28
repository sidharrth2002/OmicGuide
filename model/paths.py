import os
import time
from matplotlib import pyplot as plt
import torch
from torch import nn
from typing import Dict, Tuple

from data_utils.patch_batch import PatchBatch
from .interface import Processor
from .aggregator import TransformerAggregator
import config as cfg
import utils
from model.transcriptomics_engine import get_num_transcriptomics_features
import torch.nn.functional as F
import wandb

import torch
import torch.nn as nn

NUM_BULK_OMICS = {
    "LUAD": 1555,  # LUAD has 1555 bulk omics features
    "KIRP": 1557,
    "BRCA": 1558
}

class CrossAttentionFusion(nn.Module):
    def __init__(self, patch_dim, transcriptomics_dim, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()

        # Project patch features to Q
        self.query_proj = nn.Linear(patch_dim, hidden_dim)
        
        # Project transcriptomics features to K and V
        self.key_proj = nn.Linear(transcriptomics_dim, hidden_dim)
        self.value_proj = nn.Linear(transcriptomics_dim, hidden_dim)

        # Multihead Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, patch_dim)

        # Optional layer norm
        self.ln = nn.LayerNorm(patch_dim)

    def forward(self, patch_feats, transcriptomics_feats):
        """
        patch_feats: Tensor of shape [B, N, D] (B=batch, N=patches, D=patch_dim)
        transcriptomics_feats: Tensor of shape [B, N, F] or [B, F] (if global)
        """

        B, N, D = patch_feats.shape

        if transcriptomics_feats.dim() == 2:
            transcriptomics_feats = transcriptomics_feats.unsqueeze(1).expand(-1, N, -1)

        Q = self.query_proj(patch_feats)
        K = self.key_proj(transcriptomics_feats)
        V = self.value_proj(transcriptomics_feats)

        attn_output, attn_weights = self.cross_attn(Q, K, V)  # [B, N, H]

        enriched = self.out_proj(attn_output)            # [B, N, D]
        enriched = self.ln(patch_feats + enriched)

        return enriched

class CombineTranscriptomicsPatchCtx(nn.Module):
    def __init__(self, feat_1_dim, feat_2_dim, hdim, dropout_p=0.1):
        super(CombineTranscriptomicsPatchCtx, self).__init__()
        # Define input and output dimensions
        input_dim = feat_1_dim + feat_2_dim
        output_dim = feat_1_dim

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, feat_1, feat_2):
        x = torch.cat((feat_1, feat_2), dim=-1)
        # x = torch.cat((feat_1, feat_2.clone().detach()), dim=-1)
        # print(f"feat_1 shape: {feat_1.shape}, feat_2 shape: {feat_2.shape}, x shape: {x.shape}")
        # x shape: [batch_size, input_dim]
        residual = self.residual_proj(x)

        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.dropout2(out)

        out = F.relu(out + residual)
        return out

class GatedTranscriptomicsFusion(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.enricher = CombineTranscriptomicsPatchCtx(
            feat_1_dim=feat1_dim,
            feat_2_dim=feat2_dim,
            hdim=hidden_dim,
            dropout_p=dropout_p
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat1_dim + feat2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feat1_dim + feat2_dim)

    def forward(self, feat_1, feat_2):
        # feat_1: [B,N,D], feat_2: [B,N,F]
        # 1) mask: which patches actually have transcriptomics
        print(f"In GatedTranscriptomicsFusion")
        print(f"feat_1 shape: {feat_1.shape}, feat_2 shape: {feat_2.shape}")
        
        # TODO: WARNING - VERY VERY IMPORTANT TO REMOVE IT
        # g = 0
        # if g == 0:
        #     return feat_1
        
        valid = (feat_2.abs().sum(dim=-1, keepdim=True) != 0).float()

        # 2) compute gate normally, then zero it where invalid
        # TODO: return back to normal if this becomes a problem
        g = self.gate_mlp(self.norm(torch.cat([feat_1, feat_2], dim=-1)))
        # g = self.gate_mlp(torch.cat([feat_1, feat_2], dim=-1))
        
        # set g to 1 of shape [B, N, 1]
        # g = torch.ones(feat_1.shape[0], feat_1.shape[1], 1).to(feat_1.device)
        # print(g)
        
        # TODO: put this back
        g = g * valid

        # 3) enrichment
        enriched = self.enricher(feat_1, feat_2)

        # wandb.log({
        #     "gated_enrichment": g.mean(),
        #     "g_hist": wandb.Histogram(g.detach().cpu().numpy())
        # })

        # 4) fuse
        return g * enriched + (1.0 - g) * feat_1

class ResidualTranscriptomicsFusion(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        input_dim = feat1_dim + feat2_dim
        output_dim = feat1_dim

        print(f"input dim: {input_dim}, output_dim: {output_dim}, hidden_dim: {hidden_dim}")

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, feat_1, feat_2):
        # feat_1: [B, N, D], feat_2: [B, N, F]
        fused_input = torch.cat([feat_1, feat_2], dim=-1)
        delta = self.proj(fused_input)

        # Zero out delta where transcriptomics is all zeros
        valid = (feat_2.abs().sum(dim=-1, keepdim=True) != 0).float()
        delta = delta * valid

        return feat_1 + delta

class AttentionWeightedSum(nn.Module):
    def __init__(self, feat_1_dim, feat_2_dim):
        super(AttentionWeightedSum, self).__init__()
        # TODO: finish this, refer here: https://github.com/sidharrth2002/text-scoring/blob/main/implementations/model/tabular_combiner.py
        pass

class PATHSProcessor(nn.Module, Processor):
    """
    Implementation of a processor for one magnification level, Pi, implementing the abstract class Processor.
    The full model is an `interface.RecursiveModel` containing several `PATHSProcessor`s.
    """

    def __init__(self, config, train_config, depth: int):
        super().__init__()
        train_config: cfg.Config
        config: cfg.PATHSProcessorConfig

        self.depth = depth

        # Output dimensionality
        num_logits = train_config.num_logits()

        self.config = config
        self.train_config = train_config

        if config.model_dim is None:
            self.proj_in = nn.Identity()
            self.dim = config.patch_embed_dim
        else:
            self.proj_in = nn.Linear(
                config.patch_embed_dim, config.model_dim, bias=False
            )
            self.dim = config.model_dim

        self.slide_ctx_dim = config.trans_dim

        # Slide context can either be concatenated or summed; in our paper we choose sum (mode="residual")
        if self.config.slide_ctx_mode == "concat":
            self.classification_layer = nn.Linear(
                self.slide_ctx_dim * (depth + 1), num_logits
            )
        else:
            self.classification_layer = nn.Linear(self.slide_ctx_dim, num_logits)

        # Per-patch MLP to produce importance values \alpha
        # if self.config.add_transcriptomics:
        #     self.importance_mlp = nn.Sequential(
        #         nn.Linear(self.dim + get_num_transcriptomics_features(), config.importance_mlp_hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(config.importance_mlp_hidden_dim, 1),
        #     )
        # else:
        
        if self.config.add_transcriptomics and self.config.transcriptomics_combine_method == "context-concat":
            self.importance_mlp = nn.Sequential(
                nn.Linear(self.dim + get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path), config.importance_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.importance_mlp_hidden_dim, 1),
            )
        else:
            self.importance_mlp = nn.Sequential(
                nn.Linear(self.dim, config.importance_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.importance_mlp_hidden_dim, 1),
            )
        
        # if self.config.add_transcriptomics:
        #     self.fuse_proj = (nn.Linear(self.dim + get_num_transcriptomics_features(config.transcriptomics_model_path), self.dim, bias=False))

        if config.lstm:
            self.hdim = config.hierarchical_ctx_mlp_hidden_dim
        else:
            # A simple RNN instead of the LSTM
            self.hctx_mlp = nn.Sequential(
                nn.Linear(self.dim, config.hierarchical_ctx_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hierarchical_ctx_mlp_hidden_dim, self.dim),
            )

        # Global aggregator
        if self.config.add_transcriptomics and self.config.transcriptomics_combine_method == "context-concat":
            self.global_agg = TransformerAggregator(
                input_dim=self.dim + get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path),
                model_dim=config.trans_dim,
                output_dim=self.dim,
                nhead=config.trans_heads,
                layers=config.trans_layers,
                dropout=config.dropout,
            )                
        else:    
            self.global_agg = TransformerAggregator(
                input_dim=self.dim,
                model_dim=config.trans_dim,
                output_dim=self.dim,
                nhead=config.trans_heads,
                layers=config.trans_layers,
                dropout=config.dropout,
            )

        # self.transcriptomics_projector = nn.Linear(get_num_transcriptomics_features(), self.dim + self.hdim)

        # make two-input MLP (one input is existing patch context, the other is transcriptomics, then output is the same size as the patch context)
        # self.combine_transcriptomics_patch_ctx = nn.Sequential(
        #     nn.Linear(self.dim + self.hdim + get_num_transcriptomics_features(), self.dim + self.hdim),
        #     nn.ReLU(),
        #     nn.Linear(self.dim + self.hdim, self.dim + self.hdim)
        # )
        if config.add_bulk_omics:
            self.combine_bulk_omics = ResidualTranscriptomicsFusion(
                feat1_dim=128,
                feat2_dim=NUM_BULK_OMICS[config.cohort],  # e.g. 1555 for LUAD
                hidden_dim=self.hdim,
                dropout_p=0.2
            )
        
        if config.add_transcriptomics:
            if config.transcriptomics_combine_method == "residual_enrichment":
                self.combine_transcriptomics_patch_ctx = CombineTranscriptomicsPatchCtx(
                    feat_1_dim=self.dim, 
                    feat_2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path), 
                    hdim=self.hdim,
                    dropout_p=0.2
                )
            elif config.transcriptomics_combine_method == "attention_weighted_sum":
                # print(f"feature 1 dim: {self.dim}, feature 2 dim: {get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path)}, hdim: {self.hdim}")
                self.combine_transcriptomics_patch_ctx = AttentionWeightedSum(
                    feat_1_dim=self.dim, 
                    feat_2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path), 
                    hidden_dim=self.hdim, 
                )
            elif config.transcriptomics_combine_method == "gated_enrichment":
                # feat1_dim = (self.dim + self.hdim) if config.lstm else self.dim
                # feat2_dim = get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path)
                self.combine_transcriptomics_patch_ctx = GatedTranscriptomicsFusion(
                    feat1_dim=self.dim + self.hdim,
                    feat2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path),
                    hidden_dim=self.hdim,
                    dropout_p=0.2
                )
                # self.enrich_aggregator = GatedTranscriptomicsFusion(
                #     feat1_dim=self.dim,
                #     feat2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path),
                #     hidden_dim=self.hdim,
                #     dropout_p=0.2
                # )
            elif config.transcriptomics_combine_method == "residual_fusion":
                # TODO: put back: very important
                print("initialising here")
                # self.combine_transcriptomics_patch_ctx = ResidualTranscriptomicsFusion(
                #     feat1_dim=self.dim + self.hdim,
                #     feat2_dim=get_num_transcriptomics_features(config.transcriptomics_model_path) + NUM_BULK_OMICS[config.cohort],
                #     hidden_dim=self.hdim
                # )
                self.combine_transcriptomics_patch_ctx = ResidualTranscriptomicsFusion(
                    feat1_dim=self.dim + self.hdim, 
                    feat2_dim=get_num_transcriptomics_features(config.transcriptomics_model_path),
                    hidden_dim=self.hdim
                )
                # for pre-concatenation
                # self.combine_transcriptomics_patch_ctx = ResidualTranscriptomicsFusion(
                #     feat1_dim=self.dim, 
                #     feat2_dim=get_num_transcriptomics_features(config.transcriptomics_model_path),
                #     hidden_dim=self.hdim
                # )
            elif config.transcriptomics_combine_method == "cross_attention":
                self.combine_transcriptomics_patch_ctx = CrossAttentionFusion(
                    patch_dim=self.dim + self.hdim if config.lstm else self.dim,
                    transcriptomics_dim=get_num_transcriptomics_features(config.transcriptomics_model_path),
                    hidden_dim=self.hdim,
                    n_heads=4,
                    dropout=0.2
                )
            elif config.transcriptomics_combine_method == "context-concat":
                pass
            else:
                raise ValueError(
                    f"Unknown transcriptomics combine method: {config.transcriptomics_combine_method}"
                )

    def process(self, data: PatchBatch, lstm=None) -> Dict:
        """
        Process a batch of slides to produce a batch of logits. The class PatchBatch is defined to manage the complexity
        of each slide having different length etc. (padding is required).
        """
        patch_features = data.fts
        transcriptomics = data.transcriptomics
        # print(f"patch_features shape before projection: {patch_features.shape}")
        patch_features = self.proj_in(patch_features)

        # print(f"patch_features shape: {patch_features.shape}")
        # print(f"transcriptomics shape: {transcriptomics.shape}")

        if self.config.add_transcriptomics and (transcriptomics is not None) and self.config.transcriptomics_combine_method == "context-concat":
            # concatenate transcriptomics features to patch features
            patch_features = torch.cat(
                (patch_features, transcriptomics), dim=-1
            )
            # patch_features = self.fuse_proj(patch_features)
        
        # print(f"patch_features shape after concatenating transcriptomics: {patch_features.shape}")

        ################# Apply LSTM
        if self.config.lstm:
            assert lstm is not None

            # Initialise LSTM state at top of hierarchy
            if self.depth == 0:
                if self.config.add_transcriptomics and self.config.transcriptomics_combine_method == "context-concat":
                    hs = torch.zeros(
                        data.batch_size, data.max_patches, self.dim + transcriptomics.shape[-1], device=data.device
                    )
                else:
                    hs = torch.zeros(
                        (data.batch_size, data.max_patches, self.dim), device=data.device
                    )
                cs = torch.zeros(
                    (data.batch_size, data.max_patches, self.hdim), device=data.device
                )

            # Otherwise, retrieve it
            else:
                lstm_state = data.ctx_patch[:, :, -1]
                assert (
                    lstm_state.shape[-1] == self.dim + self.hdim
                    or lstm_state.shape[-1]
                    == self.dim + self.hdim + transcriptomics.shape[-1]
                )
                if self.config.add_transcriptomics and self.config.transcriptomics_combine_method == "context-concat":
                    hs, cs = lstm_state[..., : self.dim + transcriptomics.shape[-1]], lstm_state[..., self.dim + transcriptomics.shape[-1] :]
                else:
                    hs, cs = lstm_state[..., : self.dim], lstm_state[..., self.dim :]

            # print(f"hs shape: {hs.shape}, cs shape: {cs.shape}")

            # if transcriptomics is not None and self.config.add_transcriptomics:
            #     # generate a random tensor of the same shape as the transcriptomics tensor
            #     # random_tensor = torch.rand(transcriptomics.shape).to(transcriptomics.device)
            #     # patch_ctx = self.combine_transcriptomics_patch_ctx(
            #     #     torch.cat((patch_ctx, random_tensor), dim=-1)
            #     # )

            #     # print("adding transcriptomics features to patch context")
            #     # TODO: check if this is correct
            #     if self.config.transcriptomics_combine_method == "residual_enrichment":
            #         valid_mask = transcriptomics.abs().sum(dim=-1, keepdim=True) != 0
            #         # print("mask ", valid_mask)
            #         enriched_ctx = self.combine_transcriptomics_patch_ctx(
            #             feat_1=hs,
            #             feat_2=transcriptomics,
            #         )
            #         # enriched_ctx = self.combine_transcriptomics_patch_ctx(
            #         #     torch.cat((patch_ctx, transcriptomics.clone().detach()), dim=-1)
            #         # )
            #         hs = torch.where(valid_mask, enriched_ctx, patch_ctx)
            #         # print how many patches are not zero
            #         print(
            #             f"Number of patches with non-zero transcriptomics features: {valid_mask.sum()} out of {valid_mask.numel()}"
            #         )
            #     elif self.config.transcriptomics_combine_method == "gated_enrichment":
            #         hs = self.combine_transcriptomics_patch_ctx(
            #             feat_1=hs,        # [B, N, D]
            #             feat_2=transcriptomics   # [B, N, F]
            #         )
            #     elif self.config.transcriptomics_combine_method == "residual_fusion":
            #         hs = self.combine_transcriptomics_patch_ctx(
            #             feat_1=hs,        # [B, N, D]
            #             feat_2=transcriptomics   # [B, N, F]
            #         )
            #     elif self.config.transcriptomics_combine_method == "cross_attention":
            #         hs = self.combine_transcriptomics_patch_ctx(
            #             patch_feats=hs,        # [B, N, D]
            #             transcriptomics_feats=transcriptomics   # [B, N, F]
            #         )

            hs, cs = lstm(patch_features, hs, cs)

            patch_features = patch_features + hs  # produce Y from X

            patch_ctx = torch.concat((hs, cs), dim=-1)
            # print('patch_ctx shape after concatenating hs and cs is ', patch_ctx.shape)
        ################# Get importance values \alpha
        # (this method ensures padding is assigned 0 importance: apply the MLP+sigmoid only to non-background patches)

        # TODO: put this back in later
        # if self.config.add_transcriptomics:
        #     importance = utils.apply_to_non_padded(
        #         lambda xs: torch.sigmoid(self.importance_mlp(torch.concat((xs["contextualised_features"], xs["transcriptomics"].clone().detach()), dim=-1))),
        #         patch_features,
        #         transcriptomics,
        #         data.valid_inds,
        #         1,
        #     )[..., 0]
        # else:
        importance = utils.apply_to_non_padded(
            lambda xs: torch.sigmoid(
                self.importance_mlp(xs["contextualised_features"])
            ),
            patch_features,
            # transcriptomics,
            data.valid_inds,
            1,
        )[..., 0]
        if self.config.importance_mode == "mul":
            # produce Z from Y
            patch_features = patch_features * importance[..., None]

        # If not using the LSTM, apply a RNN instead
        if not self.config.lstm:
            if self.depth > 0 and self.config.hierarchical_ctx:
                assert len(data.ctx_patch.shape) == 4
                hctx = data.ctx_patch[:, :, -1]  # B x MaxIms x D
                
                # hctx = hctx + transcriptomics
                
                hctx = utils.apply_to_non_padded(
                    self.hctx_mlp, hctx, data.valid_inds, self.dim
                )

                patch_features = patch_features + hctx
                # patch_features

            patch_ctx = patch_features

        # append transcriptomics features to patch context
        # if self.config.add_transcriptomics and (transcriptomics is not None):
        if transcriptomics is not None and self.config.add_transcriptomics:
            # generate a random tensor of the same shape as the transcriptomics tensor
            # random_tensor = torch.rand(transcriptomics.shape).to(transcriptomics.device)
            # patch_ctx = self.combine_transcriptomics_patch_ctx(
            #     torch.cat((patch_ctx, random_tensor), dim=-1)
            # )

            # print("adding transcriptomics features to patch context")
            # TODO: check if this is correct
            if self.config.transcriptomics_combine_method == "residual_enrichment":
                valid_mask = transcriptomics.abs().sum(dim=-1, keepdim=True) != 0
                # print("mask ", valid_mask)
                enriched_ctx = self.combine_transcriptomics_patch_ctx(
                    feat_1=patch_ctx,
                    feat_2=transcriptomics,
                )
                # enriched_ctx = self.combine_transcriptomics_patch_ctx(
                #     torch.cat((patch_ctx, transcriptomics.clone().detach()), dim=-1)
                # )
                patch_ctx = torch.where(valid_mask, enriched_ctx, patch_ctx)
                # print how many patches are not zero
                print(
                    f"Number of patches with non-zero transcriptomics features: {valid_mask.sum()} out of {valid_mask.numel()}"
                )
            elif self.config.transcriptomics_combine_method == "gated_enrichment":
                patch_ctx = self.combine_transcriptomics_patch_ctx(
                    feat_1=patch_ctx,        # [B, N, D]
                    feat_2=transcriptomics   # [B, N, F]
                )
            elif self.config.transcriptomics_combine_method == "residual_fusion":
                patch_ctx = self.combine_transcriptomics_patch_ctx(
                    feat_1=patch_ctx,        # [B, N, D]
                    feat_2=transcriptomics   # [B, N, F]
                )
                # print("appending bulk omics as well")
                # # print(len(data.bulk_omics))
                # # print(len(data.bulk_omics[0]))
                # # print(f"transcriptomics shape: {transcriptomics.shape}")
                # # print(type(data.bulk_omics))
                # # print(torch.Tensor(data.bulk_omics).shape)
                # # save data.bulk_omics to a pickle file
                # # with open("bulk_omics.pickle", "wb") as f:
                # #     import pickle
                # #     pickle.dump(data.bulk_omics, f)

                # if isinstance(data.bulk_omics, list):
                #     # convert list of tensors to a tensor
                #     catted_bulk_omics = torch.stack(data.bulk_omics).T
                #     print(f"catted_bulk_omics shape: {catted_bulk_omics.shape}")
                # else:
                #     catted_bulk_omics = data.bulk_omics
                #     print(f"data.bulk_omics.shape: {catted_bulk_omics.shape}")

                # # print(f"catted_bulk_omics shape: {catted_bulk_omics.shape}")
                # # print(f"transcriptomics shape: {transcriptomics.shape}")
                # catted_features = torch.cat((transcriptomics, catted_bulk_omics.to(dtype=torch.float32).unsqueeze(1).expand(-1, patch_ctx.shape[1], -1).to(transcriptomics.device)), dim=-1)
                # # make dtype double
                # catted_features = catted_features.to(dtype=torch.float32)
                # print(f"dtype of catted_features: {catted_features.dtype}")
                # # make catted_features float
                # print(f"catted_features shape: {catted_features.shape}")
                # patch_ctx = self.combine_transcriptomics_patch_ctx(
                #     feat_1=patch_ctx,        # [B, N, D]
                #     feat_2=catted_features   # [B, N, F + NUM_BULK_OMICS["LUAD"]]
                # )
            elif self.config.transcriptomics_combine_method == "cross_attention":
                patch_ctx = self.combine_transcriptomics_patch_ctx(
                    patch_feats=patch_ctx,        # [B, N, D]
                    transcriptomics_feats=transcriptomics   # [B, N, F]
                )

        ################# Global aggregation
        d = self.config.trans_dim

        # Unused conditional sequence for aggregation. We tried putting slide context here but it performed poorly
        #  compared to residual context.
        encoder_input = torch.zeros((data.batch_size, 0, d), device=data.device)

        # Positional encoding
        xs = patch_features
        
        # if self.config.add_transcriptomics and (transcriptomics is not None):
        #     # TODO: remove this later if doesn't help
        #     print(f"patch_features shape: {patch_features.shape}, transcriptomics shape: {transcriptomics.shape}")
        #     print(f"self.dim: {self.dim}, self.hdim: {self.hdim}")
        #     xs = self.enrich_aggregator(
        #         feat_1=patch_features,
        #         feat_2=transcriptomics
        #     )
        
        patch_locs = data.locs // self.config.patch_size  # pixel coords -> patch coords
        if self.config.pos_encoding_mode == "1d":
            xs = self.global_agg.pos_encode_1d(xs)
        elif self.config.pos_encoding_mode == "2d":
            xs = self.global_agg.pos_encode_2d(xs, patch_locs)

        # Aggregate
        slide_features = self.global_agg(encoder_input, xs, None, data.num_ims)

        # Apply residual connection
        if self.config.slide_ctx_mode == "residual" and data.ctx_depth > 0:
            slide_features = slide_features + data.ctx_slide[:, -1]

        print(f"slide_features shape: {slide_features.shape}")

        ################# Produce final logits
        if self.config.slide_ctx_mode == "concat":
            all_ctx = torch.flatten(
                data.ctx_slide, start_dim=1
            )  # B x K x D -> B x (K * D)
            ft = torch.cat((all_ctx, slide_features), dim=1)
            logits = self.classification_layer(ft)
        else:
            if self.depth == 4 and self.config.add_bulk_omics:
                if isinstance(data.bulk_omics, list):
                    # convert list of tensors to a tensor
                    catted_bulk_omics = torch.stack(data.bulk_omics).T
                    print(f"catted_bulk_omics shape: {catted_bulk_omics.shape}")
                else:
                    catted_bulk_omics = data.bulk_omics
                    print(f"data.bulk_omics.shape: {catted_bulk_omics.shape}")
                
                print(f"catted_bulk_omics shape: {catted_bulk_omics.shape}")
                print(f"adding bulk omics to slide features")
                slide_features = self.combine_bulk_omics(
                    feat_1=slide_features, 
                    feat_2=catted_bulk_omics.to(dtype=torch.float32)
                )
            logits = self.classification_layer(slide_features)

        return {
            "logits": logits,
            "ctx_slide": slide_features,
            "ctx_patch": patch_ctx,  # (actually RNN state)
            "importance": importance,
        }

    def ctx_dim(self) -> Tuple[int, int]:
        if self.config.lstm:
            if self.config.add_transcriptomics and self.config.transcriptomics_combine_method == "context-concat":
                return self.slide_ctx_dim, self.dim + self.hdim + get_num_transcriptomics_features(transcriptomics_model_path=self.config.transcriptomics_model_path)
            else:
                return self.slide_ctx_dim, self.dim + self.hdim
        return self.slide_ctx_dim, self.dim


def load(slide_id, power: float):
    assert root_dir is not None, f"set_preprocess_dir must be called before load!"
    path = join(root_dir, slide_id + f"_{power:.3f}.pt")
    assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)


if __name__ == "__main__":
    print("this runs")
