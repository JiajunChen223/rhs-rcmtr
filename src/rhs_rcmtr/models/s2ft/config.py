"""Configuration for the R7 S2FT architecture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class S2FTConfig:
    """Frozen architecture knobs for Sensor-Symmetric Foundation Transfer.

    Layer indices are zero based and refer to the shared DINO block sequence.
    The default (3, 7, 11) therefore corresponds to blocks 4, 8, and 12.
    """

    adapter_layers: tuple[int, ...] = (3, 7, 11)
    adapter_bottleneck: int = 64
    interaction_layers: tuple[int, ...] = (3, 7, 11)
    interaction_dim: int = 96
    interaction_window: int = 3
    max_offset_tokens: float = 1.5
    decoder_channels: int = 128
    structure_channels: int = 64
    use_structural_refinement: bool = True
    checkpoint_frozen_blocks: bool = True

    def validate(self) -> None:
        if not self.adapter_layers:
            raise ValueError("S2FT requires at least one sensor-adapter layer")
        if any(index < 0 for index in self.adapter_layers):
            raise ValueError("adapter layer indices must be non-negative")
        if any(index < 0 for index in self.interaction_layers):
            raise ValueError("interaction layer indices must be non-negative")
        if not set(self.interaction_layers).issubset(set(self.adapter_layers)):
            raise ValueError("each interaction layer must also host a sensor adapter")
        if self.adapter_bottleneck <= 0 or self.interaction_dim <= 0:
            raise ValueError("S2FT channel dimensions must be positive")
        if self.interaction_window != 3:
            raise ValueError("R7-S2FT v1 freezes the geo-local interaction window at 3x3")
        if self.max_offset_tokens <= 0.0:
            raise ValueError("max_offset_tokens must be positive")
        if self.decoder_channels <= 0 or self.structure_channels <= 0:
            raise ValueError("decoder and structure channels must be positive")
