import torch.nn as nn
import torch.nn.functional as F
    
class MSELossKpts(nn.Module):

    def __init__(self, weight = 1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, output, target):
        
        # if the target has different size than the output, interpolate the target
        if output.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=output.shape[-2:], mode='bilinear', align_corners=True)
            
        loss_heatmap = self.criterion(output, self.weight*target)/self.weight
        
        return loss_heatmap
       