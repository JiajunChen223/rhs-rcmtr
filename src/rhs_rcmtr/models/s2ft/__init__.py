"""R7 Sensor-Symmetric Foundation Transfer architecture."""

from .config import R7_S2FT_VARIANTS, S2FTConfig, s2ft_config_for_variant
from .model import S2FTSegmenter
from .patch_transplant import SarFoundationPatchEmbed
from .sensor_adapter import SensorAdapter, SensorAdapterPair
from .shared_foundation import SharedDinoSensorBackbone

__all__ = [
    "R7_S2FT_VARIANTS",
    "S2FTConfig",
    "S2FTSegmenter",
    "SarFoundationPatchEmbed",
    "SensorAdapter",
    "SensorAdapterPair",
    "SharedDinoSensorBackbone",
    "s2ft_config_for_variant",
]
