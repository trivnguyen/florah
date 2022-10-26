
import torch
import torch.nn.functional as F

from nflows import distributions, transforms, flows

def look_ahead_mask(seq_len: int) -> torch.FloatTensor:
    """  Return mask which prevents future data to be seen """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask[mask.bool()] = -torch.inf
    return mask

class AttentionBlock(torch.nn.Module):
    """
    Self-multi head attention block:
        Self-attention -> LayerNorm -> Linear -> ReLU -> LayerNorm
    """
    def __init__(
        self, in_channels: int, hidden_channels: int, num_heads: int):

        super(AttentionBlock, self).__init__()

        self.sa_layer = torch.nn.MultiheadAttention(
            in_channels, num_heads, batch_first=True)
        self.fc_layer = torch.nn.Linear(in_channels, hidden_channels)
        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.activation = F.relu

    def forward(self, x: torch.FloatTensor,
                attn_mask: torch.FloatTensor) -> torch.FloatTensor:
        x = self.norm1(
            x + self.sa_layer(x, x, x, attn_mask=attn_mask.cuda(), need_weights=False)[0])
        x = self.norm2(self.activation(self.fc_layer(x)))
        return x

class AttentionMAF(torch.nn.Module):
    """
    Neural network architecture for merger tree generation with self-attention mechanisms
    and masked autoregressive flows
    """
    def __init__(
        self, in_channels: int, out_channels: int, seq_len: int,
        num_blocks: int, num_heads: int, hidden_features: int,
        num_layers_flows: int, hidden_features_flows: int, num_blocks_flows: int):

        super(AttentionMAF, self).__init__()

        self.sa_blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.sa_blocks.append(
                AttentionBlock(n_in, n_out, num_heads))
        self.attn_mask = look_ahead_mask(seq_len)
        self.attn_mask = self.attn_mask.cuda() if torch.cuda.is_available() else self.attn_mask

        self.maf = create_maf_layers(
            out_channels, hidden_features_flows, hidden_features,
            num_layers_flows, num_blocks_flows)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """ Forward modeling """
        for i in range(len(self.sa_blocks)):
            x = self.sa_blocks[i](x, self.attn_mask)
        return x

