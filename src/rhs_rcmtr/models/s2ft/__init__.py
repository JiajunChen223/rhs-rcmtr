"""R7 Sensor-Symmetric Foundation Transfer architecture."""

from .config import S2FTConfig
from .model import S2FTSegmenter
from .patch_transplant import SarFoundationPatchEmbed
from .sensor_adapter import SensorAdapter, SensorAdapterPair
from .shared_foundation import SharedDinoSensorBackbone

__all__ = [
    "S2FTConfig",
    "S2FTSegmenter",
    "SarFoundationPatchEmbed",
    "SensorAdapter",
    "SensorAdapterPair",
    "SharedDinoSensorBackbone",
]
