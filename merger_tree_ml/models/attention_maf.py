from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor

from . import modules
from . import flows
from . import transforms
from . import torchutils

class DataModule(modules.MAFModule):
    """
    DataModule for Attention-MAF model
    """
    arch_type = "AttentionMAF"
    def __init__(
            self, model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        super(DataModule, self).__init__(
            AttentionMAF, transforms.Preprocess, model_hparams,
            transform_hparams, optimizer_hparams)


class AttentionBlock(torch.nn.Module):
    """
    Self-attention block:
        Self-attention -> LayerNorm -> Linear -> ReLU -> LayerNorm
    """
    def __init__(
        self, embed_dim: int, num_heads: int, linear_dim: int) -> None:
        """
        Parameters
        ----------
            embed_dim: int
                Number of embedded dimension
            num_heads: int
                Number of attention heads.
                Embedded dim must be divisible by number of heads.
            linear_dim: int
                Number of linear dimension
        """
        super(AttentionBlock, self).__init__()

        self.sa_layer = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.linear_layer = torch.nn.Linear(embed_dim, linear_dim)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(linear_dim)
        self.activation = F.relu

    def forward(self, x: FloatTensor,
                attn_mask: FloatTensor) -> FloatTensor:
        x = self.norm1(
            x + self.sa_layer(
                x, x, x, attn_mask=attn_mask.cuda(), need_weights=False)[0])
        x = self.norm2(self.activation(self.linear_layer(x)))
        return x


class AttentionMAF(torch.nn.Module):
    """
    Neural network architecture for merger tree generation with self-attention
    mechanisms and masked autoregressive flows (MAF)
    """
    def __init__(
            self, in_channels: int, out_channels: int, hidden_features: int,
            num_blocks: int, num_heads: int, num_layers_flows: int,
            hidden_features_flows: int, num_blocks_flows: int
        ) -> None:
        """
        Parameters
        ----------
            in_channels: int
                Number of input channels
            out_channels: int
                Number of output channels
            hidden_features: int
                Number of hidden features in the input layer
            num_blocks: int
                Number of attention blocks
            num_heads: int
                Number of self-attention heads per attention block
            num_layers_flows: int
                Number of MAF transformations
            hidden_features_flows: int
                Number of hidden features in each MAF
            num_blocks_flows: int
                Number of MADE blocks in each MAF transformation
        """
        super(AttentionMAF, self).__init__()

        # input layer
        self.input_layer = torch.nn.Linear(in_channels, hidden_features)

        # self-attention blocks
        self.sa_blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.sa_blocks.append(
                AttentionBlock(hidden_features, num_heads, hidden_features))

        # MAF blocks
        self.maf_blocks = flows.build_maf(
            out_channels, hidden_features_flows, hidden_features,
            num_layers_flows, num_blocks_flows)

        # activation
        self.activation = F.relu

    def forward(
            self, x: FloatTensor, attn_mask: FloatTensor) -> FloatTensor:
        """ Forward pass """
        # pass input through input layer first
        x = self.activation(self.input_layer(x))

        # pass through self-attention blocks
        for i in range(len(self.sa_blocks)):
            x = self.sa_blocks[i](x, attn_mask)

        return x

    def log_prob(self, batch: Tuple[FloatTensor],
                 return_context: bool = False) -> FloatTensor:
        """ Calculate log-likelihood from batch """
        x, y, seq_len, mask = batch
        max_len = x.shape[1]

        # apply forward pass and return context for MAF
        context = self(x, attn_mask=torchutils.look_ahead_mask(max_len))

        # reshape tensor before passing through MAF blocks
        context = context.flatten(0, 1)
        y = y.flatten(0, 1)
        mask = mask.flatten()

        # calculate MAF log-likelihood P(y | context) and loss
        log_prob = self._log_prob_from_context(y[mask], context=context[mask])

        if return_context:
            return log_prob, context
        else:
            return log_prob

    def sample(self, x: FloatTensor, attn_mask: FloatTensor, num_samples: int,
               return_context: bool = False) -> FloatTensor:
        """ Sample from batch """
        # forward pass and get context
        context = self(x, attn_mask=attn_mask)
        context = context.flatten(0, 1)

        # sample and reshape
        y = self._sample_from_context(num_samples, context=context)
        return y

    def _sample_from_context(
            self, num_samples: int, context: Optional[FloatTensor] = None
        ) -> FloatTensor:
        """ Sample P(x | context) """
        return self.maf_blocks.sample(num_samples, context=context)


    def _log_prob_from_context(
            self, x: FloatTensor,
            context: Optional[FloatTensor] = None) -> FloatTensor:
        """ Return MAF log-likelihood P(x | context)"""
        return self.maf_blocks.log_prob(x, context=context)
