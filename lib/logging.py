import yaml
import mlflow
import cv2
import datetime
import os
import numpy as np

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def log_params_from_config(config):
    for key, value in config.items():
        mlflow.log_param(key, value)
        
def init_state_dict(config):
    
    # create an ouput path for plots with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path_plots = os.path.join(config["examples_path"], timestamp)
    # if path does not exist, create it
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
    
    state = {"epoch": 0, 
             "best_val": float('inf'),
             "prev_best_val": float('inf'),
             "train_loss": [], 
             "val_loss": [],
             "sunlamp_loss":[],
             "lightbox_loss":[],
             "global_step": 0,
             "val_step": 0,
             "path_plots": path_plots}
    
    return state

def flatten_config(config):
    """
    Flatten the nested configuration dictionary into a flat dictionary.
    
    Args:
        config (dict): Nested configuration dictionary.
        
    Returns:
        dict: Flattened dictionary.
    """
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_value = flatten_config(value)
            flat_config.update({f"{sub_key}": sub_value for sub_key, sub_value in flat_value.items()})
        else:
            flat_config[key] = value
    return flat_config


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


def save_heatmap_example(heatmap, heatmap_gt, image, state, val_dataset="validation"):
    
    heatmap = tensor_to_cvmat(heatmap[0].sum(0,True))
    heatmap_gt = tensor_to_cvmat(heatmap_gt[0].sum(0,True))
    image = tensor_to_cvmat(image[0])
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    heatmap_gt = cv2.resize(heatmap_gt, (heatmap.shape[1], heatmap.shape[0]))
    heatmap_gt_overlay = (0.5*heatmap_gt.astype(np.float64)+0.5*image.astype(np.float64)).astype(np.uint8)
    output = np.concatenate((heatmap, heatmap_gt_overlay, heatmap_gt), axis=1)
    
    if val_dataset == "test":
        cv2.imwrite(os.path.join("test.png"), output)
    else:
        cv2.imwrite(os.path.join(state["path_plots"], val_dataset + "_heatmap_" + str(state["val_step"]) + ".png"), output)




def log_into_mlflow(state, model, config):
    
    mlflow.log_metric("Training Loss", state["train_loss"][-1], step=state["epoch"])
    mlflow.log_metric("Validation Loss", state["val_loss"][-1], step=state["epoch"])
    mlflow.log_metric("Learning Rate", state["learning_rate"], step=state["epoch"])
    if config["dataset"] == "speedplus":
        mlflow.log_metric("Best Validation Loss", state["best_val"], step=state["epoch"])
        mlflow.log_metric("Sunlamp Validation Loss", state["sunlamp_loss"][-1], step=state["epoch"])
        mlflow.log_metric("Lightbox Validation Loss", state["lightbox_loss"][-1], step=state["epoch"])
    
    
    # add the images of the validation steps, go to the folder and add any file ended with png
    
    if config["save_val_examples"]:
        for file in os.listdir(state["path_plots"]):
            if file.endswith(".png"):
                mlflow.log_artifact(os.path.join(state["path_plots"], file))
    
    
    
    model_name = config["backbone"]
    if model_name == "resnet":
        model_name = f"{model_name}_{config['resnet_size']}"
        
    if state["best_val"] < state["prev_best_val"]:
        state["prev_best_val"] = state["best_val"]
        mlflow.pytorch.log_model(model, model_name + "_best")
        
    if state["epoch"] == config["num_epochs"]-1:
        mlflow.pytorch.log_model(model, model_name + "_final")



