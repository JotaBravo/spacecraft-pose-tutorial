
from tqdm import tqdm
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader

from kornia.geometry import undistort_image

from lib import logging, transformations, utils
from loader.speedplus import SPEEDPlus
from models.vitpose import ViTPose
from models.resnetpose import ResNetPose
from models.losses import MSELossKpts

import mlflow

def train(train_loader, model, criterion, optimizer, augmentation, config, state):

    model.train()
    
    total_loss = 0
  
    for i, data in enumerate(tqdm(train_loader)):

        with torch.no_grad():
            
            # move everything to the GPU
            data = utils.dict_to_device(data, config["device"])   

            data["image"] = undistort_image(data["image"], 
                                            data["intrinsics_scaled"], 
                                            data["d_coef"])
            
            heatmap_gt = utils.render_heatmap(data, config)
            
            data["image"], heatmap_gt = transformations.apply_augmentations(data["image"], 
                                                                            heatmap_gt, 
                                                                            config, 
                                                                            augmentation)

        # forward pass
        heatmap = model(data["image"])
        loss = criterion(heatmap, heatmap_gt)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % config["log_interval"] == 0:
            mlflow.log_metric("Running Training Loss", loss, step=state["global_step"])

        state["global_step"] += 1
        
        total_loss += loss.item()

    state["train_loss"].append(total_loss / len(train_loader))

@torch.no_grad
def eval(val_loader, model, criterion, config, augmentation, state, val_dataset = "validation"):

  
    model.eval()
    total_loss = 0
    state["val_step"] = 0
    for i, data in enumerate(tqdm(val_loader)):
        
        data = utils.dict_to_device(data, config["device"])
        
        data["image"] = undistort_image(data["image"], 
                                        data["intrinsics_scaled"], 
                                        data["d_coef"])

        heatmap_gt = utils.render_heatmap(data, config)
        if config["normalize_minmax"]:
            data["image"] = transformations.normalize_min_max(data["image"])
        # forward pass
        heatmap = model(data["image"])
            
        loss = criterion(heatmap, heatmap_gt)
            
        total_loss += loss.item()
        
        if config["save_val_examples"] and i%config["val_examples_interval"] == 0:
            logging.save_heatmap_example(heatmap, heatmap_gt, data["image"], state, val_dataset)    
        
        state["val_step"] +=1
    
    if val_dataset == "sunlamp":
        state["sunlamp_loss"].append(total_loss / len(val_loader))
    elif val_dataset == "lightbox":
        state["lightbox_loss"].append(total_loss / len(val_loader))
    else:
        state["val_loss"].append(total_loss / len(val_loader))    
        state["best_val"] = min(state["val_loss"])
    
def run_experiment(config):
    
    # ---------------------------------------------------------------------------------
    # Load config and transformations
    # ---------------------------------------------------------------------------------
    
    transform = transformations.get_transforms(config)
    augmentation = transformations.get_augmentations()

    # ---------------------------------------------------------------------------------
    # Get loaders
    # ---------------------------------------------------------------------------------
    
    train_loader = utils.get_loader("train", config, transform)
    
    val_loader = utils.get_loader("validation", config, transform)
    
    sunlamp_loader = utils.get_loader("sunlamp", config, transform)
    
    lightbox_loader = utils.get_loader("lightbox", config, transform)
    
    # ---------------------------------------------------------------------------------
    # Get model and loss
    # ---------------------------------------------------------------------------------

    criterion = MSELossKpts(weight=config["w_heatmap"])
    
    if config["backbone"] == "vit":
        model = ViTPose(config).to(config["device"])
    elif config["backbone"] == "resnet":
        model = ResNetPose(config).to(config["device"])
    else:
        raise ValueError("Unknown backbone")

    nparam = utils.count_parameters(model)
    print("Loaded model {} with {} parameters".format(config["backbone"], nparam))
    
    # ---------------------------------------------------------------------------------
    # Get optimizer and schedulers
    # ---------------------------------------------------------------------------------    
    
    # Optimizer
    optimizer = Adam(model.parameters(), 
                      lr=config["learning_rate"], 
                      betas=config["betas"], 
                      weight_decay=config["weight_decay"])    
    
    
    # ---------------------------------------------------------------------------------
    # Start training
    # --------------------------------------------------------------------------------- 
    
    state = logging.init_state_dict(config)

    for epoch in range(config["num_epochs"]):
        
        state["epoch"] = epoch
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        train(train_loader, 
            model,
            criterion, 
            optimizer,
            augmentation,
            config, 
            state)
    
        eval(val_loader, 
            model,
            criterion, 
            config, 
            augmentation,
            state)

        eval(sunlamp_loader,
             model,
             criterion,
             config,
             augmentation,
             state,
             val_dataset = "sunlamp")

        eval(lightbox_loader,
             model,
             criterion,
             config,
             augmentation,
             state,
             val_dataset = "lightbox")
        
        logging.log_into_mlflow(state, model, config)
