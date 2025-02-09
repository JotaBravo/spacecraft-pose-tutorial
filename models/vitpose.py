import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


class ViTPose(nn.Module):
    def __init__(self, config: dict) -> None:
        super(ViTPose, self).__init__()
        

        
        self.blocks = ViT(img_size=config["target_size"],
                            in_chans=config["im_channels"],
                        patch_size=config["patch_size"],
                        embed_dim=config["embed_dim"],
                        depth=config["depth"],
                        num_heads=config["num_heads"],
                        ratio=config["ratio"],
                        use_checkpoint=config["use_checkpoint"],
                        mlp_ratio=config["mlp_ratio"],
                        qkv_bias=config["qkv_bias"],
                        drop_rate=config["drop_path_rate"])
                
        self.keypoint_head = TopdownHeatmapSimpleHead(in_channels=config["embed_dim"], 
                                out_channels=config["num_keypoints"],
                                num_deconv_filters=config["num_deconv_filters"],
                                num_deconv_layers=config["num_deconv_layers"],
                                num_deconv_kernels=config["num_deconv_kernels"])

        self.blocks.init_weights()
        weights = torch.load("models/pt_weights/mae_pretrain_vit_base.pth")["model"]
        # resize pos embedding if needed by interpolating
        pos_embed_pre = weights["pos_embed"]
        pos_embed_mod = self.blocks.pos_embed
        
        h = pos_embed_mod.shape[1]
        w = pos_embed_mod.shape[2]
        if pos_embed_pre.shape != pos_embed_mod.shape:

            pos_embed_pre = F.interpolate(pos_embed_pre.unsqueeze(1), size=(h, w), mode="bicubic")
            
        weights["pos_embed"] = pos_embed_pre.squeeze(1)
        
        self.blocks.load_state_dict(weights, strict=False)
        self.keypoint_head.init_weights()
        
        
    def forward_features(self, x):
        return self.blocks(x)
    
    def forward(self, x):
        return self.keypoint_head(self.blocks(x))