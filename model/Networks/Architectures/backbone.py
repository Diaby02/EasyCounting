import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from .anomalib_functions import (
    TimmFeatureExtractor
)
from torchvision.models._utils import IntermediateLayerGetter
import re

class ResNetBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str, reduction: int, kernel_dim: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #if not train_backbone:
                parameter.requires_grad_(False)
            # we freeze the first layer

        regex = r"(\d)"
        matches = re.findall(regex, return_layer, re.MULTILINE)
        self.num_last_layer = int(matches[0])

        self.reduction = reduction
        self.kernel_dim = kernel_dim

        return_layers = {}
        for i in range(2,self.num_last_layer+1):
            key = "layer" + str(i)
            return_layers[key] = str(i-2)
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        """supports both NestedTensor and torch.Tensor
        """
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        out = self.body(x)
        output_list = []
        for i in range(self.num_last_layer-1):
            output_list.append(out[str(i)])
        
        if size[0] < 24: # if we are in the exemplar case, we interpolate at 3x3
                x = torch.cat([
                F.interpolate(f, size=(self.kernel_dim,self.kernel_dim), mode='bilinear', align_corners=True)
                for f in output_list
            ], dim=1)

        else:

                x = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in output_list
            ], dim=1)

        return x


class ResNet(ResNetBase):

    def __init__(
        self,
        dilation: bool,
        reduction: int,
        requires_grad: bool,
        kernel_dim: int,
        last_layer= "layer4"
    ):

        resnet = getattr(models,'resnet50')(
            replace_stride_with_dilation=[False, False, dilation],
            weights="ResNet50_Weights.IMAGENET1K_V1", norm_layer=FrozenBatchNorm2d
        )

        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
            map_location="cpu"
        )
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        resnet.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        if last_layer == "layer4":
            num_channels = 3584
        else:
            num_channels = 1536

        for n, param in resnet.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)
        super().__init__(resnet, requires_grad, num_channels, last_layer, reduction,kernel_dim)

    """ def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = layer2 = self.backbone.layer2(x)
        x = layer3 = self.backbone.layer3(x)

        if self.last_layer == "layer4":
            x = layer4 = self.backbone.layer4(x)

            if size[0] < 24: # if we are in the exemplar case, we interpolate at 3x3
                x = torch.cat([
                    F.interpolate(f, size=(3,3), mode='bilinear', align_corners=True)
                    for f in [layer2, layer3, layer4]
                ], dim=1)

            else:

                x = torch.cat([
                    F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                    for f in [layer2, layer3, layer4]
                ], dim=1)


        else:

            if size[0] < 24:
                x = torch.cat([
                    F.interpolate(f, size=(3,3), mode='bilinear', align_corners=True)
                    for f in [layer2, layer3]
                ], dim=1)

            else:
                x = torch.cat([
                    F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                    for f in [layer2, layer3]
                ], dim=1)

        return x """
    

class MobileNetV3(nn.Module):

    def __init__(
        self,
        kernel_dim: int,
        reduction=8,
        require_grad=False
    ):

        super(MobileNetV3, self).__init__()
        # Initialize feature extractor
        self.extractor = TimmFeatureExtractor(
                backbone="mobilenetv3_large_100",
                layers=["blocks.2", "blocks.4", "blocks.6"],
                requires_grad=require_grad
            )
        
        """
        for the above function to work, you need to change the function in 
        anomalib/models/components/feature_extractors/timm.py by

        def _map_layer_to_idx(self, offset: int = 3) -> list[int]:
        
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:  # noqa: PERF203
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                logger.warning(msg)
                # Remove unfound key from layer dict
                self.layers.remove(layer)

        return idx
        """
        self.num_channels= 960 + 112 + 40 # 1112
        self.reduction = reduction
        self.kernel_dim = kernel_dim

        for n, param in self.extractor.feature_extractor.named_parameters():
            param.requires_grad_(require_grad)

        

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        features = self.extractor(x)

        features1 = features["blocks.2"]
        features2 = features["blocks.4"]
        features3 = features["blocks.6"]
        
        if size[0] < 24: # if we are in the exemplar case, we interpolate at 3x3
            x = torch.cat([
                F.interpolate(f, size=(self.kernel_dim,self.kernel_dim), mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        else:

            x = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        return x
    
class EfficientNet_b0(nn.Module):

    def __init__(
        self,
        kernel_dim: int,
        reduction=8,
        require_grad=False
    ):

        super(EfficientNet_b0, self).__init__()
        # Initialize feature extractor
        self.extractor = TimmFeatureExtractor(
                backbone="efficientnet_b0",
                layers=["blocks.2", "blocks.4", "blocks.6"]
            )
        self.num_channels= 320 + 112 + 40 # 472
        self.reduction = reduction
        self.kernel_dim = kernel_dim

        for n, param in self.extractor.feature_extractor.named_parameters():
            param.requires_grad_(require_grad)

        

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        features = self.extractor(x)

        features1 = features["blocks.2"]
        features2 = features["blocks.4"]
        features3 = features["blocks.6"]
        
        if size[0] < 24: # if we are in the exemplar case, we interpolate at 3x3
            x = torch.cat([
                F.interpolate(f, size=(self.kernel_dim,self.kernel_dim), mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        else:

            x = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        return x
    
class TinyViT(nn.Module):

    def __init__(
        self,
        kernel_dim: int,
        reduction=8,
        require_grad=False
    ):

        super(TinyViT, self).__init__()
        # Initialize feature extractor
        self.extractor = TimmFeatureExtractor(
                backbone="tiny_vit_5m_224",
                layers=["stages.1", "stages.2", "stages.3"]
            )
        self.num_channels= 320 + 160 + 128 # 608
        self.reduction = reduction
        self.kernel_dim = kernel_dim

        for n, param in self.extractor.feature_extractor.named_parameters():
            param.requires_grad_(require_grad)

        

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        features = self.extractor(x)

        features1 = features["stages.1"]
        features2 = features["stages.2"]
        features3 = features["stages.3"]
        
        if size[0] < 24: # if we are in the exemplar case, we interpolate at 3x3
            x = torch.cat([
                F.interpolate(f, size=(self.kernel_dim,self.kernel_dim), mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        else:

            x = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [features1, features2, features3]
            ], dim=1)

        return x



