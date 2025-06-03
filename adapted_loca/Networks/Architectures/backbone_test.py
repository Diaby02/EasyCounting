import torch
import timm
from anomalib.models.components.feature_extractors import (
    TimmFeatureExtractor
)
# Initialize feature extractor
extractor = TimmFeatureExtractor(
        backbone="tiny_vit_5m_224",
        layers=["stages.1", "stages.2", "stages.3"]
    )
# Extract features from input
print([info["module"] for info in extractor.feature_extractor.feature_info.info])
inputs = torch.randn(2, 3, 256, 256)
features = extractor(inputs)
for f in features.keys():
    print(features[f].shape)



    
