"""
Improved Generator for DGMR

This module implements an improved generator architecture for the Deep
Generative Model of Radar (DGMR) with enhancements for high-intensity
precipitation nowcasting.

Key improvements:
1. Multi-scale feature fusion for better spatial representation
2. Self-attention mechanism for long-range dependencies
3. Residual connections for stable training
4. Convolutional LSTM for temporal consistency

Reference:
- Ravuri et al. (2021). Skilful precipitation nowcasting using deep
  generative models of radar. Nature, 599(7883), 681-687.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttention2D(nn.Module):
    """
    Self-Attention Mechanism for 2D Features

    Implements scaled dot-product attention for capturing long-range
    spatial dependencies in precipitation fields.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    key_channels : int, optional
        Number of channels for keys/queries (default: in_channels // 8)
    value_channels : int, optional
        Number of channels for values (default: in_channels)

    Examples
    --------
    >>> attn = SelfAttention2D(in_channels=128)
    >>> x = torch.randn(4, 128, 64, 64)  # [B, C, H, W]
    >>> out = attn(x)  # Same shape as input
    """

    def __init__(
        self,
        in_channels: int,
        key_channels: Optional[int] = None,
        value_channels: Optional[int] = None
    ):
        super().__init__()

        if key_channels is None:
            key_channels = in_channels // 8
        if value_channels is None:
            value_channels = in_channels

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # Projections for queries, keys, and values
        self.query = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, value_channels, kernel_size=1)

        # Output projection
        self.out = nn.Conv2d(value_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter (initialized to 0 for gradual incorporation)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W]

        Returns
        -------
        out : torch.Tensor
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute queries, keys, values
        proj_query = self.query(x).view(B, self.key_channels, -1)  # [B, K, HW]
        proj_key = self.key(x).view(B, self.key_channels, -1)  # [B, K, HW]
        proj_value = self.value(x).view(B, self.value_channels, -1)  # [B, V, HW]

        # Compute attention scores
        # [B, HW, HW]
        attention = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = F.softmax(attention / (self.key_channels ** 0.5), dim=-1)

        # Apply attention to values
        # [B, V, HW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, self.value_channels, H, W)

        # Output projection
        out = self.out(out)

        # Residual connection with learnable weight
        return self.gamma * out + x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell

    Combines convolutional operations with LSTM gating for capturing
    spatiotemporal dependencies.

    Parameters
    ----------
    input_channels : int
        Number of input channels
    hidden_channels : int
        Number of hidden channels
    kernel_size : int, default=3
        Size of convolutional kernel

    Reference
    ---------
    Shi et al. (2015). Convolutional LSTM Network: A Machine Learning
    Approach for Precipitation Nowcasting. NeurIPS.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()

        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Convolution for input-to-state and state-to-state
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # i, f, o, g gates
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W]
        hidden_state : tuple of torch.Tensor, optional
            Previous hidden state (h, c)

        Returns
        -------
        h : torch.Tensor
            Hidden state [B, hidden_channels, H, W]
        (h, c) : tuple
            Complete hidden state
        """
        B, _, H, W = x.shape

        # Initialize hidden state if needed
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.conv(combined)

        # Split into gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        # Apply gates
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Candidate cell state

        # Update cell state
        c = f * c + i * g

        # Update hidden state
        h = o * torch.tanh(c)

        return h, (h, c)


class MultiScaleConvBlock(nn.Module):
    """
    Multi-Scale Convolutional Block

    Applies convolutions at multiple scales and fuses the features.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_sizes : list of int, default=[3, 5, 7]
        Kernel sizes for different scales

    Examples
    --------
    >>> block = MultiScaleConvBlock(64, 128)
    >>> x = torch.randn(4, 64, 64, 64)
    >>> out = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [3, 5, 7]
    ):
        super().__init__()

        self.n_scales = len(kernel_sizes)
        channels_per_scale = out_channels // self.n_scales

        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels_per_scale,
                    kernel_size=k,
                    padding=k // 2
                ),
                nn.GroupNorm(8, channels_per_scale),
                nn.ReLU(inplace=True)
            )
            for k in kernel_sizes
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W]

        Returns
        -------
        out : torch.Tensor
            Output tensor [B, out_channels, H, W]
        """
        # Apply multi-scale convolutions
        multi_scale_features = [conv(x) for conv in self.convs]

        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(fused)

        return out


class ImprovedDGMRGenerator(nn.Module):
    """
    Improved DGMR Generator

    Enhanced generator architecture for precipitation nowcasting with
    better performance on high-intensity events.

    Key features:
    1. Multi-scale feature extraction
    2. Convolutional LSTM for temporal modeling
    3. Self-attention for spatial coherence
    4. Residual connections throughout

    Parameters
    ----------
    input_frames : int, default=12
        Number of input frames (historical observations)
    output_frames : int, default=24
        Number of output frames (forecast to generate)
    hidden_dim : int, default=128
        Number of hidden channels
    num_blocks : int, default=4
        Number of processing blocks
    use_attention : bool, default=True
        Whether to use self-attention
    use_convlstm : bool, default=True
        Whether to use ConvLSTM for temporal modeling

    Examples
    --------
    >>> model = ImprovedDGMRGenerator(input_frames=12, output_frames=24)
    >>> x = torch.randn(2, 12, 256, 256)  # 12 input frames
    >>> out = model(x)  # [2, 24, 256, 256]
    """

    def __init__(
        self,
        input_frames: int = 12,
        output_frames: int = 24,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        use_attention: bool = True,
        use_convlstm: bool = True
    ):
        super().__init__()

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_attention = use_attention
        self.use_convlstm = use_convlstm

        # Input encoding
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_frames, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        # Initial feature extraction
        self.init_block = MultiScaleConvBlock(64, hidden_dim)

        # Processing blocks
        self.processing_blocks = nn.ModuleList()
        self.attention_layers = nn.ModuleList() if use_attention else None

        for i in range(num_blocks):
            # Multi-scale processing block
            self.processing_blocks.append(
                MultiScaleConvBlock(hidden_dim, hidden_dim)
            )

            # Optional attention layer
            if use_attention:
                self.attention_layers.append(
                    SelfAttention2D(hidden_dim)
                )

        # ConvLSTM for temporal modeling (optional)
        if use_convlstm:
            self.convlstm = ConvLSTMCell(
                input_channels=hidden_dim,
                hidden_channels=hidden_dim,
                kernel_size=3
            )
        else:
            self.convlstm = None

        # Upsampling and output
        self.upsample = nn.Sequential(
            MultiScaleConvBlock(hidden_dim, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, output_frames, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input frames [B, input_frames, H, W]

        Returns
        -------
        out : torch.Tensor
            Generated frames [B, output_frames, H, W]
        """
        B, _, H, W = x.shape

        # Encode input
        x = self.input_encoder(x)

        # Initial feature extraction
        features = self.init_block(x)

        # Apply processing blocks with attention
        for i, block in enumerate(self.processing_blocks):
            # Multi-scale processing
            new_features = block(features)

            # Attention
            if self.use_attention:
                new_features = self.attention_layers[i](new_features)

            # Residual connection
            features = features + new_features

        # Temporal modeling with ConvLSTM (optional)
        if self.convlstm is not None:
            # Expand for temporal dimension
            features_expanded = features.unsqueeze(1).repeat(1, self.output_frames, 1, 1, 1)
            features_expanded = features_expanded.view(B * self.output_frames, self.hidden_dim, H, W)

            # Apply ConvLSTM
            lstm_out, _ = self.convlstm(features_expanded)
            features = lstm_out.view(B, self.output_frames, self.hidden_dim, H, W)

            # Take first time step for further processing
            features = features[:, 0]
        else:
            # For output frames, repeat features
            features = features.unsqueeze(1).repeat(1, self.output_frames, 1, 1, 1)
            features = features.view(B * self.output_frames, self.hidden_dim, H, W)

        # Upsample
        features = self.upsample(features)

        # Generate output
        output = self.output_layer(features)

        # Reshape to [B, output_frames, H, W]
        output = output.view(B, self.output_frames, H, W)

        return output

    def generate_sequence(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate sequence

        This method generates the output sequence autoregressively,
        using previously generated frames as input for subsequent steps.

        Parameters
        ----------
        x : torch.Tensor
            Initial input frames [B, input_frames, H, W]
        num_steps : int, optional
            Number of steps to generate (default: output_frames)

        Returns
        -------
        sequence : torch.Tensor
            Generated sequence [B, num_steps, H, W]
        """
        if num_steps is None:
            num_steps = self.output_frames

        generated = []

        # Initialize with input
        current_input = x

        for step in range(num_steps):
            # Generate next frame(s)
            output = self.forward(current_input)

            # Take the first predicted frame
            next_frame = output[:, 0:1]  # [B, 1, H, W]

            generated.append(next_frame)

            # Update input for next iteration
            # Remove oldest frame and add new frame
            current_input = torch.cat([current_input[:, 1:], next_frame], dim=1)

        # Concatenate generated frames
        sequence = torch.cat(generated, dim=1)  # [B, num_steps, H, W]

        return sequence


class DGMRDiscriminator(nn.Module):
    """
    Discriminator for DGMR

    Convolutional discriminator that distinguishes between real and
    generated precipitation sequences.

    Parameters
    ----------
    input_channels : int, default=24
        Number of input channels (number of frames)
    hidden_channels : list of int, default=[64, 128, 256, 512]
        Number of channels at each layer
    """

    def __init__(
        self,
        input_channels: int = 24,
        hidden_channels: list = [64, 128, 256, 512]
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for i, out_ch in enumerate(hidden_channels):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_ch = out_ch

        # Final output layer
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input sequence [B, T, H, W]

        Returns
        -------
        out : torch.Tensor
            Discriminator output [B, 1, H', W']
        """
        return self.model(x)
