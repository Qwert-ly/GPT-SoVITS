import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
import GPT_SoVITS.module.commons as commons
from GPT_SoVITS.module.modules import LayerNorm, ResidualCouplingLayer as TransformerCouplingLayer
from functools import partial
from torch import Tensor
from typing import Optional, Tuple


# JIT optimization for frequently used functions
@torch.jit.script
def matmul_with_relative_keys(x: Tensor, y: Tensor) -> Tensor:
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))


@torch.jit.script
def matmul_with_relative_values(x: Tensor, y: Tensor) -> Tensor:
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    return torch.matmul(x, y.unsqueeze(0))


def relative_position_to_absolute_position(x: Tensor) -> Tensor:
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(
        x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
    )
    # Reshape and slice out the padded elements.
    return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]


def absolute_position_to_relative_position(x: Tensor) -> Tensor:
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(
        x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
    )
    x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]


@torch.jit.script
def attention_bias_proximal(length: int) -> Tensor:
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


def fused_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        k_channels: int = 64,
        key_relative_embeddings: Optional[Tensor] = None,
        value_relative_embeddings: Optional[Tensor] = None,
        proximal_bias: bool = False,
        block_length: Optional[int] = None,
        dropout_p: float = 0.0,
        training: bool = True
) -> Tuple[Tensor, Tensor]:
    # Calculate scores
    scores = torch.matmul(query / math.sqrt(k_channels), key.transpose(-2, -1))

    # Add relative position embeddings if provided
    t_s = key.size(2)
    if key_relative_embeddings is not None:
        rel_logits = matmul_with_relative_keys(query / math.sqrt(k_channels), key_relative_embeddings)
        scores_local = relative_position_to_absolute_position(rel_logits)
        scores += scores_local

    # Add proximal bias if requested
    if proximal_bias:
        scores += attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
        if block_length is not None:
            block_mask = torch.ones_like(scores).triu(-block_length).tril(block_length)
            scores = scores.masked_fill(block_mask == 0, -1e4)

    # Calculate attention weights and apply dropout
    p_attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        p_attn = F.dropout(p_attn, p=dropout_p)

    # Calculate output
    output = torch.matmul(p_attn, value)

    # Add relative values if provided
    if value_relative_embeddings is not None:
        relative_weights = absolute_position_to_relative_position(p_attn)
        output += matmul_with_relative_values(relative_weights, value_relative_embeddings)

    return output, p_attn


def get_relative_embeddings(relative_embeddings: Tensor, length: int, window_size: int) -> Tensor:
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (window_size + 1), 0)
    slice_start_position = max((window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
        padded_relative_embeddings = F.pad(
            relative_embeddings,
            commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
        )
    else:
        padded_relative_embeddings = relative_embeddings
    return padded_relative_embeddings[:, slice_start_position:slice_end_position]


def causal_padding(x: Tensor, kernel_size: int) -> Tensor:
    if kernel_size == 1:
        return x
    pad_l = kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    return F.pad(x, commons.convert_pad_shape(padding))


def same_padding(x: Tensor, kernel_size: int) -> Tensor:
    if kernel_size == 1:
        return x
    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    return F.pad(x, commons.convert_pad_shape(padding))


@torch.jit.script
def gelu_activation(x: Tensor) -> Tensor:
    return x * torch.sigmoid(1.702 * x)


class FFN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            filter_channels,
            kernel_size,
            p_dropout=0.0,
            activation=None,
            causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        # Use JIT compiled padding functions
        self.padding = causal_padding if causal else same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.zeros_(self.conv_1.bias)
        nn.init.zeros_(self.conv_2.bias)

    def forward(self, x, x_mask):
        x_padded = self.padding(x * x_mask, self.kernel_size)
        x = self.conv_1(x_padded)

        if self.activation == "gelu":
            x = gelu_activation(x)
        else:
            x = F.relu(x)

        x = self.drop(x)
        x_padded = self.padding(x * x_mask, self.kernel_size)
        x = self.conv_2(x_padded)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            n_heads,
            p_dropout=0.0,
            window_size=None,
            heads_share=True,
            block_length=None,
            proximal_bias=False,
            proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads

        # Use grouped convolutions for query/key/value projections for efficiency
        self.qkv_proj = nn.Conv1d(channels, channels * 3, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        # Initialize weights
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.conv_o.weight)
        nn.init.zeros_(self.conv_o.bias)

        if proximal_init:
            # Initialize with identity-like mapping
            with torch.no_grad():
                # Split the weights into three parts (q, k, v)
                weight_shape = self.qkv_proj.weight.shape
                q_weight = self.qkv_proj.weight[:weight_shape[0] // 3]
                k_weight = self.qkv_proj.weight[weight_shape[0] // 3:2 * weight_shape[0] // 3]
                # Make k_weight the same as q_weight for attention to focus on the identical position
                k_weight.copy_(q_weight)

    def forward(self, x, c, attn_mask=None):
        # Handle self-attention case efficiently
        if x is c:
            # Use single projection for q, k, v in self-attention
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=1)
        else:
            # In cross-attention, use different projections
            q = self.qkv_proj(x)[:, :self.channels, :]
            kv = self.qkv_proj(c)[:, self.channels:, :]
            k, v = torch.chunk(kv, 2, dim=1)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = *key.size(), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Prepare relative embeddings if needed
        key_relative_embeddings = None
        value_relative_embeddings = None

        if self.window_size is not None:
            assert (t_s == t_t), "Relative attention is only available for self-attention."
            key_relative_embeddings = get_relative_embeddings(self.emb_rel_k, t_s, self.window_size)
            value_relative_embeddings = get_relative_embeddings(self.emb_rel_v, t_s, self.window_size)

        # Use optimized fused attention forward
        output, p_attn = fused_attention_forward(
            query,
            key,
            value,
            mask=mask,
            k_channels=self.k_channels,
            key_relative_embeddings=key_relative_embeddings,
            value_relative_embeddings=value_relative_embeddings,
            proximal_bias=self.proximal_bias,
            block_length=self.block_length,
            dropout_p=self.p_dropout,
            training=self.training
        )

        # Reshape output back to original dimensions
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn


class Encoder(nn.Module):
    def __init__(
            self,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size=1,
            p_dropout=0.0,
            window_size=4,
            isflow=False,
            **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        # Create layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

        # Conditional layer for flow
        self.cond_layer = None
        self.cond_pre = None
        if isflow:
            cond_layer = torch.nn.Conv1d(
                kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1
            )
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
            self.gin_channels = kwargs["gin_channels"]

    def forward(self, x, x_mask, g=None):
        # Prepare attention mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        # Process conditional input if provided
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)

        # Apply encoder layers
        x = x * x_mask
        for i in range(self.n_layers):
            # Apply conditional input if available
            if g is not None and self.cond_pre is not None:
                x_cond = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset: cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(
                    x_cond, g_l, torch.IntTensor([self.hidden_channels])
                )

            # Self-attention
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # Feed-forward
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class Decoder(nn.Module):
    def __init__(
            self,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size=1,
            p_dropout=0.0,
            proximal_bias=False,
            proximal_init=True,
            **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        # Create layers
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        # Create masks
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        # Apply decoder layers
        x = x * x_mask
        for i in range(self.n_layers):
            # Self-attention
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            # Encoder-decoder attention
            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # Feed-forward
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.depth_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.depth_conv.weight)
        nn.init.xavier_uniform_(self.point_conv.weight)
        if bias:
            nn.init.zeros_(self.depth_conv.bias)
            nn.init.zeros_(self.point_conv.bias)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        self.depth_conv = remove_weight_norm(self.depth_conv, name="weight")
        self.point_conv = remove_weight_norm(self.point_conv, name="weight")


class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            bias=True,
            dilation=1,
            padding_mode="zeros",
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.depth_conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.depth_conv.weight)
        nn.init.xavier_uniform_(self.point_conv.weight)
        if bias:
            nn.init.zeros_(self.depth_conv.bias)
            nn.init.zeros_(self.point_conv.bias)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        remove_weight_norm(self.depth_conv, name="weight")
        remove_weight_norm(self.point_conv, name="weight")


def weight_norm_modules(module, name="weight", dim=0):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(
            module, Depthwise_Separable_TransposeConv1D
    ):
        module.weight_norm()
        return module
    else:
        return weight_norm(module, name, dim)


def remove_weight_norm_modules(module, name="weight"):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(
            module, Depthwise_Separable_TransposeConv1D
    ):
        module.remove_weight_norm()
    else:
        remove_weight_norm(module, name)


class FFT(nn.Module):
    def __init__(
            self,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers=1,
            kernel_size=1,
            p_dropout=0.0,
            proximal_bias=False,
            proximal_init=True,
            isflow=False,
            **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        # Conditional layer for flow
        self.cond_layer = None
        self.cond_pre = None
        if isflow:
            cond_layer = torch.nn.Conv1d(
                kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1
            )
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
            self.gin_channels = kwargs["gin_channels"]

        # Create layers
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()

        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """
        x: decoder input
        h: encoder output
        """
        # Create self-attention mask
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )

        # Process conditional input if provided
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)

        # Apply FFT layers
        x = x * x_mask
        for i in range(self.n_layers):
            # Apply conditional input if available
            if g is not None and self.cond_pre is not None:
                x_cond = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset: cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(
                    x_cond, g_l, torch.IntTensor([self.hidden_channels])
                )

            # Self-attention
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x


