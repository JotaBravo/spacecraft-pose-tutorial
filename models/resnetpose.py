import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .backbone.resnet import ResNetEncoder, BasicBlock, Bottleneck
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

class ResNetPose(nn.Module):
    def __init__(self, config: dict) -> None:
        
        super(ResNetPose, self).__init__()
        
        block_class, layers = resnet_spec[config["resnet_size"]]    
            
        self.backbone = ResNetEncoder(block_class, layers, config["im_channels"])
                
        self.keypoint_head = TopdownHeatmapSimpleHead(in_channels=self.backbone.inplanes, 
                                out_channels=config["num_keypoints"],
                                num_deconv_filters=config["num_deconv_filters"],
                                num_deconv_layers=config["num_deconv_layers"],
                                num_deconv_kernels=config["num_deconv_kernels"])
        
        
        # Try loading weights from PyTorch Hub (if available)
        model_url = None
        # Example URLs for common sizes (adjust based on available models)
        if config["resnet_size"] == 18:
        # ResNet-18 weights might not be available on PyTorch Hub
            model_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
            
        elif config["resnet_size"] == 34:
            # ResNet-34 weights might not be available on PyTorch Hub
            model_url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
            
        elif config["resnet_size"] == 50:
            model_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
            
        elif config["resnet_size"] == 101:
            model_url = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
            
        elif config["resnet_size"] == 152:
            model_url = "https://download.pytorch.org/models/resnet152-b121ed2d.pth"

        if model_url is not None:
        
            state_dict = load_state_dict_from_url(model_url)
            
            # the first layer is for RGB but our input is grayscale. Average the weights for the first layer
            # and set the weights for the first layer to the average of the weights
            if config["im_channels"] == 1:
                state_dict['conv1.weight'] = state_dict['conv1.weight'].mean(dim=1, keepdim=True)
            # remove the fc layer
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded pre-trained weights for Resnet-{config['resnet_size']} from PyTorch Hub")
 
            #except Exception as e:
            #    print(f"Error loading weights from PyTorch Hub for {config['resnet_size']}: {e}")

        self.keypoint_head.init_weights()
        
        
        
 
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.keypoint_head(self.backbone(x))




