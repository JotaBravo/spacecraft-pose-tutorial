import kornia.augmentation as K
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import normalize_min_max
import torchvision.transforms as transforms


def get_transforms(config):
    
    
    if config["im_channels"] == 1:
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(config["target_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.445], std=[0.269]), # ImageNet grayscale normalization
        ])
    else:
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(config["target_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]), # ImageNet grayscale normalization
        ])        
        
        
    
    return transform


def get_augmentations():
    
    # Define geometric augmentation transformations
    augmentation_geometric = K.AugmentationSequential(
        K.RandomHorizontalFlip(),  # Flip horizontally with 50% probability
        K.RandomVerticalFlip(),    # Flip vertically with 50% probability
        #K.RandomAffine(degrees=30,scale=0.25),
        K.RandomCutMixV2(),        
        data_keys=["input"]       
    )
    
    # Define intensity augmentation transformationsq
    augmentation_intensity = K.AugmentationSequential(
        K.RandomContrast(contrast=(0.98, 1.00), p=1.0),
        K.RandomBrightness(brightness=(0.98,1.02), p=1.0),
        K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
        K.RandomErasing(scale=(0.002, 0.01), ratio=(0.3, 3.3), p=1.0),  
        #K.RandomSolarize(),
        data_keys=["input"]
    )

    augmentations = {"geometric": augmentation_geometric, "intensity": augmentation_intensity}
    
    return augmentations
    
def apply_augmentations(image, heatmap_gt, config, augmentations):
    
    
    augmentation_geometric = augmentations["geometric"]
    augmentation_intensity = augmentations["intensity"]
    
    
    if config["augmentation_intensity"]:
        image = augmentation_intensity(image)
        
    if config["normalize_minmax"]:
        image = normalize_min_max(image)
    
    if config["augmentation_geometric"]:
        image = augmentation_geometric(image)
        heatmap_gt = augmentation_geometric(heatmap_gt, params=augmentation_geometric._params)
    

    return image, heatmap_gt