import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.geometry import undistort_image
import kornia
import mlflow.pytorch as mlflow_pytorch
from lib import logging, transformations, speedscore
from loader.speedplus import SPEEDPlus
from models.vitpose import ViTPose
from models.resnetpose import ResNetPose

import mlflow, yaml

import numpy as np
import cv2
import os
from lib import utils

# fix all random seeds
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test(dataset, loader, model, config, split):
    

    model.eval()
    predictions = []
    gt_quaternions = []
    gt_translations = []
    print("Evaluating model...")
    for data in tqdm(loader):
        
        # move everything to the GPU
        data = utils.dict_to_device(data, config["device"])
        
        # undistort and augment the image
        img = data["image"]
        #img = undistort_image(img, data["intrinsics_scaled"], data["d_coef"])
        
        if config["normalize_minmax"]:
            img = transformations.normalize_min_max(img     )
        
        heatmap = model(img)

        coords = utils.argmax_2d(heatmap)
  
        predictions.extend(logging.tensor_to_numpy(coords))
        gt_quaternions.extend(logging.tensor_to_numpy(data["quaternion"]))
        gt_translations.extend(logging.tensor_to_numpy(data["translation"]))

        
    print("Computing PnP...")

    results = {"translation_errors": [], "speed_r": [], "speed_t":[], "speed": []}
    for i in tqdm(range(0, len(predictions))):
        

        try:
            flag, rotation, translation, inliers= cv2.solvePnPRansac(dataset.kpts.T,  # 3D kpts
                                                                    predictions[i],   # 2D kpts
                                                                    dataset.k_mat_im, # K mat (at gt resolution)
                                                                    distCoeffs=None,  # we undistorted the img
                                                                    confidence=0.999,
                                                                    reprojectionError=4.0,
                                                                    flags=cv2.SOLVEPNP_EPNP)


            if translation[0] < -1:
                translation[0] = -1
            if translation[0] > 1:
                translation[0] = 1

            if translation[1] < -2:
                translation[1] = -2
            if translation[1] > 2:
                translation[1] = 2

            if translation[2] > 10:
                translation[2] = 10
            if translation[2] < 2:
                translation[2] = 2   

            rotation = torch.from_numpy(np.squeeze(rotation)).unsqueeze(0).to(config["device"])
            rotation = kornia.geometry.axis_angle_to_quaternion(rotation).unsqueeze(0)
            rotation = logging.tensor_to_numpy(rotation[0])

            score = speedscore.speed_score(translation, rotation, gt_translations[i], gt_quaternions[i])

            translation = np.squeeze(translation)
            translation_error = (np.abs(translation - gt_translations[i])).tolist()
            results["translation_errors"].append(translation_error)
            results["speed_r"].append(score[2])
            results["speed_t"].append(score[1])
            
            results["speed"].append(score[0])

            # compute the rotation error per axis

            
        except Exception as e:
            print(e)
            print("Error in PnP")
            continue
        
        # results["translation_errors"] is a list of lists, make it a np array. then compute the error per axis
    results["translation_errors"] = np.array(results["translation_errors"])


    # print the keys of the configuration parameters as the title of the column
    # the next row shall include the values. Make it all latex friendly

    
    print("LR & Int. Aug & Geom. Aug & MinMax Aug & Fixed Std & Scale Std & W Heatmap & Mean X & Mean Y & Mean Z & S_r & S_t & S \\\\")
    
    
    
    with open(split + "_results.txt", "a") as file:
        file.write(f"{config['learning_rate']} & {config['augmentation_intensity']} & {config['augmentation_geometric']} & {config['normalize_minmax']} & {config['fixed_std']} & {config['std_val']} & {config['w_heatmap']} & {np.round(np.mean(results['translation_errors'][:, 0]), 3)} & {np.round(np.mean(results['translation_errors'][:, 1]), 3)} & {np.round(np.mean(results['translation_errors'][:, 2]), 3)} & {np.round(np.mean(results['speed_r']), 3)} & {np.round(np.mean(results['speed_t']), 3)} & {np.round(np.mean(results['speed']), 3)} \\\\\n")



    print("Translation errors X: ", np.mean(results["translation_errors"][:, 0]))
    print("Translation errors Y:",  np.mean(results["translation_errors"][:, 1]))
    print("Translation errors Z:",  np.mean(results["translation_errors"][:, 2]))
        
def main():
    
    # ---------------------------------------------------------------------------------
    # Parse arguments
    # ---------------------------------------------------------------------------------
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Run name")
    args = parser.parse_args()
    
    # ---------------------------------------------------------------------------------
    # Load config and transformations
    # ---------------------------------------------------------------------------------
    
    # Load the associated MLflow run's metadata
    run = mlflow.get_run(args.run)
    config = utils.convert_dict_values(run.data.params)
    

    transform = transformations.get_transforms(config)

    # ---------------------------------------------------------------------------------
    # Get loaders
    # ---------------------------------------------------------------------------------
    
    validation_dataset = utils.get_dataset("validation", config, transform)
    validation_loader = utils.get_loader("validation", config, transform)
    
    sunlamp_dataset = utils.get_dataset("sunlamp", config, transform)
    sunlamp_loader = utils.get_loader("sunlamp", config, transform)
    
    lightbox_dataset = utils.get_dataset("lightbox", config, transform)
    lightbox_loader = utils.get_loader("lightbox", config, transform)
    
    # ---------------------------------------------------------------------------------
    # Get model and loss
    # ---------------------------------------------------------------------------------
    
    if config["backbone"] == "vit":
        model = ViTPose(config).to(config["device"])
    elif config["backbone"] == "resnet":
        model = ResNetPose(config).to(config["device"])
    else:
        raise ValueError("Unknown backbone")

    ## ---------------------------------------------------------------------------------
    ## Get models and optimizer
    ## ---------------------------------------------------------------------------------
    
    #if config["backbone"] == "vit":
    #    model = ViTPose(config).to(config["device"])
    #elif config["backbone"] == "resnet":
    #    model = ResNetPose(config).to(config["device"])
    #else:
    #    raise ValueError("Unknown backbone")

    # Load the model from the MLflow run
    model_uri = "runs:/" + args.run + "/resnet_18"
    model = mlflow.pytorch.load_model(model_uri)
    
    
    
    ## ---------------------------------------------------------------------------------
    ## Training loop
    ## ---------------------------------------------------------------------------------

    with torch.no_grad():
        print("Testing on validation")
        test(validation_dataset, validation_loader, model, config, split="validation")
#
        print("Testing on sunlamp")
        test(sunlamp_dataset, sunlamp_loader, model, config, split="sunlamp")
        
        print("Testing on lightbox")
        test(lightbox_dataset, lightbox_loader, model, config, split="lightbox")
        


if __name__ == "__main__":
    main()
