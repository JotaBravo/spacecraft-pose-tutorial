import os
import json
from kornia.utils.grid import create_meshgrid

import torch
from torch.utils.data import DataLoader
from loader.speedplus import SPEEDPlus
from loader.swisscube import BOP_Dataset

import kornia

def load_json(path_to_file):
    with open(path_to_file) as f:
        config = json.load(f) 
    return config 

def save_json(file_path, data):
    with open(file_path, 'w') as f:  
        json.dump(data, f, indent=4)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def my_render_gaussian2d(
    mean, std, size, normalized_coordinates: bool = True
):
    r"""ADAPTED FROM KORNIA so the gaussian is not normalized
    Render the PDF of a 2D Gaussian distribution.

    Args:
        mean: the mean location of the Gaussian to render, :math:`(\mu_x, \mu_y)`. Shape: :math:`(*, 2)`.
        std: the standard deviation of the Gaussian to render, :math:`(\sigma_x, \sigma_y)`.
          Shape :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        size: the (height, width) of the output image.
        normalized_coordinates: whether ``mean`` and ``std`` are assumed to use coordinates normalized
          in the range of :math:`[-1, 1]`. Otherwise, coordinates are assumed to be in the range of the output shape.

    Returns:
        tensor including rendered points with shape :math:`(*, H, W)`.
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates, mean.device)
    grid = grid.to(mean.dtype)
    pos_x: torch.Tensor = grid[..., 0].view(height, width)
    pos_y: torch.Tensor = grid[..., 1].view(height, width)

    # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
    #              = exp(dists * ks),
    #                where dists = (x - \mu)^2 and ks = -1 / (2 \sigma^2)

    # dists <- (x - \mu)^2
    dist_x = (pos_x - mean[..., 0, None, None]) ** 2
    dist_y = (pos_y - mean[..., 1, None, None]) ** 2

    # ks <- -1 / (2 \sigma^2)
    k_x = -0.5 * torch.reciprocal(std[..., 0, None, None])
    k_y = -0.5 * torch.reciprocal(std[..., 1, None, None])

    # Assemble the 2D Gaussian.
    exps_x = torch.exp(dist_x * k_x)
    exps_y = torch.exp(dist_y * k_y)
    gauss = exps_x * exps_y

    return gauss

def render_heatmap(data, config):
    
    if config["fixed_std"] == True:
        std_heatmap = (torch.ones_like(data["kpts_std"])*config["std_val"])
    else:
        std_heatmap = data["kpts_std"]*config["scale_std"]
    
    heatmap = my_render_gaussian2d(mean=data["kpts_image"],
                                std=std_heatmap,
                                size=(config["target_size"][0], config["target_size"][1]),
                                normalized_coordinates=False).type(torch.float32)
    
    # those channels where data["kpts_vis"] is 0 should be zero

    ## remove kpts with 15% probability
    #removed = torch.rand_like(data["kpts_filtered"]) > 0.15
    
    #data["kpts_filtered"] = data["kpts_filtered"] * removed
    #heatmap = heatmap * data["kpts_filtered"][:, :, None, None]
    # The gaussian is not normalized, we want the center value to equal 1
    return heatmap

def dict_to_device(datadict, device):
    for key in datadict.keys():
        # check if its not list
        if not isinstance(datadict[key], list):
            datadict[key] = datadict[key].to(device)
    return datadict 


def get_dataset(split, config, transform):
    
    return SPEEDPlus(split, 
                    speed_root = config["dataset_path"], 
                    kpts_root = config["keypoints_path"],
                    transform_input=transform,
                    target_size = config["target_size"],
                    filter_margin=config["filter_margin"]
                    )
    

def get_loader(split, config, transform):
    
    sppeed_loader = get_dataset(split, config, transform)
    
    pytorch_loader = DataLoader(sppeed_loader, 
                              batch_size=config["batch_size"], 
                              shuffle=True, 
                              num_workers=config["num_workers"], 
                              drop_last=False)
    return pytorch_loader


def get_swisscube_dataset(split, config, transform):
    nsamples = 30356
    if split == "validation":
        nsamples = 4431
    print(split, nsamples)
    bop_dataset = BOP_Dataset(
        os.path.join(config["dataset_path"], split + ".txt"), 
        os.path.join(config["dataset_path"],"models/"), 
        os.path.join(config["dataset_path"],"swisscube_bbox.json"), 
        transform,
        config["target_size"],
        samples_count=nsamples,
    training = True)
        
    return bop_dataset
def get_swisscube_loader(split, config, transform):
    
    bop_dataset = get_swisscube_dataset(split, config, transform)
    
    pytorch_loader = DataLoader(bop_dataset, 
                              batch_size=config["batch_size"], 
                              shuffle=True, 
                              num_workers=config["num_workers"], 
                              drop_last=False)
    return pytorch_loader



import ast

def convert_dict_values(data):
    """
    Recursively converts string values in a dictionary to appropriate types.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            # If the value is another dictionary, recursively convert its values
            data[key] = convert_dict_values(value)
        elif isinstance(value, list):
            # If the value is a list, recursively convert each element
            data[key] = [convert_dict_values(item) if isinstance(item, (dict, list)) else convert_value(item) for item in value]
        else:
            # Convert the value to the appropriate type
            data[key] = convert_value(value)
    return data

def convert_value(value):
    """
    Converts a string value to an appropriate type.
    """
    try:
        # Try converting the value to an integer
        return int(value)
    except ValueError:
        try:
            # If conversion to an integer fails, try converting to a float
            return float(value)
        except ValueError:
            try:
                # If conversion to a float fails, check if it's a list representation and parse it
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # If it's not a valid list representation, leave it as a string
                return value
            
            
def heatmap_to_points(mask):
    return kornia.geometry.subpix.spatial_soft_argmax2d(mask,normalized_coordinates=False)


def argmax_2d(input):
  """
  Finds the location of the maximum in each NxW sub-matrix of a BxCxNxW tensor.

  Args:
      input: A BxCxNxW tensor.

  Returns:
      A BxCx2 tensor containing the x and y coordinates of the maximum 
      in each sub-matrix.
  """
  # Reshape to separate channels and spatial dimensions
  B, C, H, W = input.shape
  reshaped = input.view(B, C, H * W)

  # Find argmax along the spatial dimension (combined NxW)
  max_vals, max_indices = torch.max(reshaped, dim=2)

  # Unpack indices and convert to x, y coordinates
  x_coords = max_indices % W  # Get remainder for x-coordinate within sub-matrix
  y_coords = max_indices // W # Integer division for y-coordinate (row index)

  # Reshape back to BxCx2 with x and y coordinates
  output = torch.stack([x_coords, y_coords], dim=2)
  return output.view(B, C, 2).type(torch.double), max_vals