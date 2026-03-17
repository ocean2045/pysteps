# DGMR (Deep Generative Model of Radar) Implementation
#
# This module implements the Improved DGMR for precipitation nowcasting
# with enhanced performance for high-intensity events.
#
# Reference:
# - Ravuri et al. (2021). Skilful precipitation nowcasting using deep
#   generative models of radar. Nature, 599(7883), 681-687.
# - Extended with balanced loss function for high-intensity events
#   (AMS AI for Earth Systems, 2024)

from .core.improved_loss import (
    ExtendedBalancedLoss,
    ImprovedDGMRLoss
)

from .core.improved_generator import (
    ImprovedDGMRGenerator,
    SelfAttention2D
)

__all__ = [
    'ExtendedBalancedLoss',
    'ImprovedDGMRLoss',
    'ImprovedDGMRGenerator',
    'SelfAttention2D'
]
