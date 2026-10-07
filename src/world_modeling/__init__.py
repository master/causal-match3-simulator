"""World models of match-3 gameplay: a next-board transformer, a stop-gradient JEPA, the same JEPA with an EMA target encoder, and an end-to-end LeWM. Every model maps (board, counters, action, level, tier, served difficulty) to the next board and counters and is trained and evaluated by the same script (logged-action k-step replay and engine-in-the-loop closed-loop rollouts)."""

from match3_simulator.world_modeling.decoder import StructuredDecoderConfig, StructuredStateDecoder
from match3_simulator.world_modeling.latent import LatentWorldModel, LatentWorldModelConfig
from match3_simulator.world_modeling.registry import build_world_model, config_from_dict, config_kind
from match3_simulator.world_modeling.transformer import TransitionTransformer, TransitionTransformerConfig

__all__ = [
    "LatentWorldModel",
    "LatentWorldModelConfig",
    "StructuredDecoderConfig",
    "StructuredStateDecoder",
    "TransitionTransformer",
    "TransitionTransformerConfig",
    "build_world_model",
    "config_from_dict",
    "config_kind",
]
