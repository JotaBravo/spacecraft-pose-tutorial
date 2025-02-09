import os
import json
import numpy as np
import cv2
from kornia.geometry import render_gaussian2d
from kornia.utils.grid import create_meshgrid

import torch

from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt


def load_json(path_to_file):
    with open(path_to_file) as f:
        config = json.load(f) 
    return config 

def save_json(file_path, data):
    with open(file_path, 'w') as f:  
        json.dump(data, f, indent=4)

def quat2dcm(q):

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def draw_keypoints(image, keypoints, diameter=8, color = (255, 0, 0)):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    return image


def tensor_to_cvmat(tensor, BGR = True):
    mat = tensor.cpu().data.numpy().squeeze()

    if (tensor.shape[0]) == 3:
        mat = np.transpose(mat, (1, 2, 0))

    min_mat = np.min(mat)
    max_mat = np.max(mat)    
    mat = (mat-min_mat)/(max_mat-min_mat)
    out = (255*mat).astype(np.uint8)
    if BGR == True:
        out = cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
    return out

def tensor_to_numpy(tensor):
    tensor = tensor.cpu().data.numpy().squeeze()
    return tensor


def dict_to_device(datadict, device):
    for key in datadict.keys():
        # check if its not list
        if not isinstance(datadict[key], list):
            datadict[key] = datadict[key].to(device)
    return datadict 



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
        std_heatmap = torch.ones_like(data["kpts_std"])*config["std_val"]
    else:
        std_heatmap = data["kpts_std"]*config["scale_std"]
    
    heatmap = my_render_gaussian2d(mean=data["kpts_pos"],
                                std=std_heatmap,
                                size=(config["target_size"][0], config["target_size"][1]),
                                normalized_coordinates=False).type(torch.float32)
    
    # those channels where data["kpts_vis"] is 0 should be zero
    heatmap = heatmap * data["kpts_vis"][:, :, None, None]
    # The gaussian is not normalized, we want the center value to equal 1
    return heatmap

def calculate_loss_heatmap(h_tgt, h_gt, flag):
    loss_heatmap = 0.0
    for batch_index in range(h_tgt.shape[0]):
        # Get visible key-points
        flag_vis = flag[batch_index].type(torch.bool)
        # Get the batch predictions and ground-truth
        pred_batch = h_tgt[batch_index, flag_vis, :, :]
        gt_batch = h_gt[batch_index, flag_vis, :, :]
        loss_heatmap += mse_loss(pred_batch, 1e6*gt_batch)/1000
    loss_heatmap = loss_heatmap / h_tgt.shape[0] 
    return loss_heatmap


def plot_loss(state):
    
    plt.figure()
    plt.plot(state["train_loss"], label='Training Loss')
    plt.plot(state["val_loss"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(state["path_plots"],"losses.png"))
    plt.close()



def get_max_coordinates(tensor):
  """
  This function takes a PyTorch tensor of shape Bx11xHxW and returns a tensor of shape Bx11x2,
  containing the (x, y) coordinates of the maximum value in each HxW matrix.

  Args:
      tensor: A PyTorch tensor of shape Bx11xHxW (where H and W can be different).

  Returns:
      A PyTorch tensor of shape Bx11x2 containing the (x, y) coordinates of the maximum value in each HxW matrix.
  """
  # Get the spatial dimensions (H and W) from the tensor's shape
  spatial_shape = tensor.size()[2:]  # Get all dimensions after the first two (batch and channel)

  # Reshape the tensor to Bx11x(H*W)
  flattened = tensor.view(tensor.size(0), tensor.size(1), -1)

  # Find the indices of the maximum value along the last dimension
  max_indices = torch.argmax(flattened, dim=2)

  # Unpack the indices to get separate x and y coordinates
  x_coords = torch.div(max_indices, spatial_shape[1], rounding_mode='floor')
  y_coords = max_indices % spatial_shape[0]  # Use height (H) for modulo

  # Reshape the coordinates back to Bx11x2
  coords = torch.stack([x_coords, y_coords], dim=2).double()

  return coords

import kornia
def heatmap_to_points(mask):
    return kornia.geometry.subpix.spatial_soft_argmax2d(mask,normalized_coordinates=False)




def rotation_vector_to_quaternion(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Extract the elements of the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Compute quaternion components
    qw = np.sqrt(1 + r11 + r22 + r33) / 2
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    return np.array([qw, qx, qy, qz])


def tensor_to_cvmat(tensor, BGR = True):
    mat = tensor.cpu().data.numpy().squeeze()

    if (tensor.shape[0]) == 3:
        mat = np.transpose(mat, (1, 2, 0))

    min_mat = np.min(mat)
    max_mat = np.max(mat)    
    mat = (mat-min_mat)/(max_mat-min_mat)
    out = (255*mat).astype(np.uint8)
    if BGR == True:
        out = cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
    return out


def save_heatmap_example(heatmap, heatmap_gt, image, state):
    
    heatmap = tensor_to_cvmat(heatmap[0].sum(0,True))
    heatmap_gt = tensor_to_cvmat(heatmap_gt[0].sum(0,True))
    # repeat third dimension for the image (tensor) to simulate RGB
    #image = image.repeat(1,3,1,1)
    #image = tensor_to_mat(image[0])*255
    #image =  np.transpose(image, (1, 2, 0))
    image = tensor_to_cvmat(image[0])
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    heatmap_gt_overlay = (0.5*heatmap_gt.astype(np.float64)+0.5*image.astype(np.float64)).astype(np.uint8)
    output = np.concatenate((heatmap, heatmap_gt_overlay, heatmap_gt), axis=1)
    
    cv2.imwrite(os.path.join(state["path_plots"], "heatmap_" + str(state["val_step"]) + ".png"), output)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_mat(tensor):
    tensor = tensor.cpu().data.numpy().squeeze()
    return tensor
