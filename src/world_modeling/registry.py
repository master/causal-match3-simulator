"""Build a world model from its configuration and rebuild configurations from checkpoints. The kind field written into every configuration selects the class."""

from __future__ import annotations

from dataclasses import fields

from torch import nn

from match3_simulator.world_modeling.latent import LatentWorldModel, LatentWorldModelConfig
from match3_simulator.world_modeling.transformer import TransitionTransformer, TransitionTransformerConfig

WorldModelConfig = TransitionTransformerConfig | LatentWorldModelConfig


def config_kind(config: WorldModelConfig) -> str:
    """Return the kind tag of a configuration.

    Inputs: a world-model configuration.
    Outputs: "transformer" or "latent".
    """
    return config.kind


def config_from_dict(raw: dict[str, object]) -> WorldModelConfig:
    """Rebuild a configuration from a checkpoint dictionary, ignoring unknown keys.

    Inputs: dict with a kind field and the configuration values.
    Outputs: the matching configuration dataclass.
    """
    kind = raw.get("kind")
    cls = {"transformer": TransitionTransformerConfig, "latent": LatentWorldModelConfig}.get(str(kind))
    if cls is None:
        raise ValueError(f"unknown world-model kind {kind!r}")
    names = {f.name for f in fields(cls)}
    values = {k: v for k, v in raw.items() if k in names}
    if cls is LatentWorldModelConfig and "target" not in values:  # checkpoints written before the target field
        values["target"] = "shared" if values.get("objective") == "lewm" else "stopgrad"
    return cls(**values)


def build_world_model(config: WorldModelConfig) -> nn.Module:
    """Instantiate the module for a configuration.

    Inputs: a world-model configuration.
    Outputs: a TransitionTransformer or LatentWorldModel.
    """
    if isinstance(config, TransitionTransformerConfig):
        return TransitionTransformer(config)
    if isinstance(config, LatentWorldModelConfig):
        return LatentWorldModel(config)
    raise TypeError(f"unsupported configuration {type(config).__name__}")


__all__ = ["WorldModelConfig", "build_world_model", "config_from_dict", "config_kind"]
