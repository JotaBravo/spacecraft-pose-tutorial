import os
import json
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .swisscubeutils.boxlist import BoxList
from .swisscubeutils.poses import PoseAnnot

from .swisscubeutils.utils import (
    load_bop_meshes,
    load_bbox_3d,
    get_single_bop_annotation
)

class BOP_Dataset(Dataset):
    
    def __init__(self, image_list_file, mesh_dir, bbox_json, transform, target_size, samples_count=0, training=True):
        
        # file list and data should be in the same directory
        dataDir = os.path.split(image_list_file)[0]
        with open(image_list_file, 'r') as f:
            self.img_files = f.readlines()
            self.img_files = [dataDir + '/' + x.strip() for x in self.img_files]
         
        rawSampleCount = len(self.img_files)
        if training and samples_count > 0:
            self.img_files = random.choices(self.img_files, k = samples_count)

        if training:
            random.shuffle(self.img_files)

        print("Number of samples: %d / %d" % (len(self.img_files), rawSampleCount))
        
        self.meshes, self.objID_2_clsID= load_bop_meshes(mesh_dir)
        
        self.bbox_3d = load_bbox_3d(bbox_json)
        

        self.transformer = transform
        self.training = training
        
        self.rows = target_size[0]
        self.cols = target_size[1]
        
        self.MAX_DIST = 17.84
        self.MIN_DIST = 1.78



    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
            
        img_path = self.img_files[index]

        img = Image.open(img_path, 'r')
    
        K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation(img_path, self.objID_2_clsID)

        # extract widht and height from PIL image
        K = self.scale_intrinsics(K, self.cols, self.rows)
        
        
        target = PoseAnnot(self.bbox_3d, K, merged_mask, class_ids, rotations, translations, self.cols, self.rows)

        kpts_2d, kpts_3d = target.compute_keypoint_positions()
        
        # compute the standard deviation

        distance = kpts_3d[-1, :]/100
  
        kpts_std = self.compute_std_distance(distance)
        
        sample = {
            "image": self.transformer(img),
                
            "quaternion": np.squeeze(np.array(rotations,dtype=np.float64)),
            "rotation_mat": np.squeeze(np.array(rotations,dtype=np.float64)),
            "translation": np.squeeze(np.array(translations,dtype=np.float64)),
                
            "kpts_image": np.squeeze(kpts_2d.astype(np.float64)),
            "kpts_3d": kpts_3d.T,
            "kpts_std": kpts_std,
            "kpts_filtered": np.ones((len(kpts_2d)), dtype=np.float64),
                
            "intrinsics_scaled": K.astype(np.float64),
            "d_coef": False
        }
            

        return sample

    def scale_intrinsics(self, k_input, cols, rows):
        """
            Scale the intrinsic parameters of a camera matrix.

            Parameters:
                self (object): Instance of the class.
                k_input (numpy.ndarray): Input camera matrix.
                cols (int): Number of columns in the image.
                rows (int): Number of rows in the image.

            Returns:
                numpy.ndarray: Scaled camera matrix.
        """
        
        k_output = k_input.copy()
        k_output[0, :] *= ((cols)/(k_input[0,2]*2))
        k_output[1, :] *= ((rows)/(k_input[1,2]*2))
        
        return k_output
    
    def compute_std_distance(self, distance):
        """
            Compute the standard deviation of keypoints based on distance.

            Parameters:
                self (object): Instance of the class.
                distance (numpy.ndarray): Array of distances from keypoints to the camera.

            Returns:
                numpy.ndarray: Standard deviation of keypoints based on distance.
        """    
        scaled_dist = np.expand_dims((self.MAX_DIST-distance)+self.MIN_DIST,axis=1)
        
        # circles, same std for x and y        
        std_kpt  = np.repeat(scaled_dist, 2, axis=1) 
        
        return std_kpt    