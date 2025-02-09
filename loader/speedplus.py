import os
import json
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

"""
speedplus.py

This file contains the implementation of the Camera and data loader classes for the SPEED+ dataset.
The code provided here is intended to be self-contained and can be used independently. Th is is, you
should be able to port this to your own repository or code without much hustle.
"""

class Camera:
    
    def __init__(self,speed_root):

        """" Utility class for accessing camera parameters. """
        with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
            camera_params = json.load(f)
            
        self.fx = camera_params['fx'] # focal length[m]
        self.fy = camera_params['fy'] # focal length[m]
        self.nu = camera_params['Nu'] # number of horizontal[pixels]
        self.nv = camera_params['Nv'] # number of vertical[pixels]
        self.ppx = camera_params['ppx'] # horizontal pixel pitch[m / pixel]
        self.ppy = camera_params['ppy'] # vertical pixel pitch[m / pixel]
        self.fpx = self.fx / self.ppx  # horizontal focal length[pixels]
        self.fpy = self.fy / self.ppy  # vertical focal length[pixels]
        self.k = camera_params['cameraMatrix']
        self.K = np.array(self.k, dtype=np.float32) # cameraMatrix
        self.dcoef = camera_params['distCoeffs']
        self.ccx = camera_params['ccx'] # principal point x[pixels]
        self.ccy = camera_params['ccy']

class SPEEDPlus(Dataset):

    """ Data loader for the SPEED+ dataset https://arxiv.org/abs/2110.03101 
    
        Parameters:
            split (str): Split of the dataset (e.g., 'train', 'val').
            speed_root (str): Root directory of the dataset.
            path_kpts (str): Path to keypoints.
            transform_input (function): Function to transform input data.
            target_size (tuple): Target size of the input data.
            filter_margin (int): Margin for filtering keypoints.
            
            
        In each iteration, the loader returns a dictionary with the following keys
            - image: Image data.
            - quaternion: Ground truth quaternion.
            - rotation_mat: Ground truth rotation matrix.
            - translation: Ground truth translation vector.
            - kpts_image: Keypoints in the image plane.
            - kpts_3d: Keypoints in 3D space.
            - kpts_std: Standard deviation of keypoints based on distance.
            - kpts_filtered: Filtered keypoints.
            - intrinsics_scaled: Scaled intrinsic camera matrix.
            - d_coef: Camera distortion coefficients.
    """
    
    def __init__(self, split, speed_root, kpts_root, transform_input, target_size, filter_margin):
     
        # ----------------------------------------------------------------------
        #                       Parse inputs
        # ----------------------------------------------------------------------
        
        self.speed_root = speed_root
        self.transform_input = transform_input
        self.filter_margin = filter_margin  
        
        # ----------------------------------------------------------------------
        #                       Load the json ground-truth files
        # ----------------------------------------------------------------------
        
        if split not in {'train', 'validation', 'sunlamp', 'lightbox'}:
            raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

        if split in {'train', 'validation'}:

            self.image_root = os.path.join(speed_root, 'synthetic', 'images')

            with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                label_list = json.load(f)

        else:
            self.image_root = os.path.join(speed_root, split, 'images')
            with open(os.path.join(speed_root, split, "test.json"), 'r') as f:
                label_list = json.load(f)
        
        self.sample_ids = [label['filename'] for label in label_list]
        self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}
        
        # ----------------------------------------------------------------------
        #                       Load the auxiliary data
        # ----------------------------------------------------------------------
                
        # undo sizes
        self.rows = target_size[0]
        self.cols = target_size[1]

        # world spacecraft keypoints (3xN)
        with open(os.path.join(kpts_root), 'r') as f:
            kpts = json.load(f)
            
        self.kpts = np.array(kpts) 
        
        # keypoints in homogeneous coordinates (4xN)
        self.kpts_h = np.vstack((self.kpts, np.ones(self.kpts.shape[1])))

        # load the camera info
        self.camera = Camera(self.speed_root)
        
        # given the target size, we need to scale the intrinsics
        self.k_mat_im = self.scale_intrinsics(self.camera.K,
                                              self.cols, 
                                              self.rows)
        # retrieve the camera coefficients 
        self.d_coef = np.array(self.camera.dcoef, dtype=np.float32)
        
        # precompute max-min distances (for scale-aware heatmaps)
        self.MAX_DIST, self.MIN_DIST = self.find_max_min_distances()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):

        # ------------------------------------------------------- #
        # Read the image and the ground-truth
        # ------------------------------------------------------- #
        
        # sample the image
        sample_id = self.sample_ids[idx]

        img = Image.open(os.path.join(self.image_root, sample_id), 'r')
        
        quaternion = np.array(self.labels[sample_id]['q'], dtype=np.double)
        
        rotation_mat = self.quat2dcm(quaternion)
        translation_vec = np.array(self.labels[sample_id]['r'], dtype=np.double)
        translation_vec = np.expand_dims(translation_vec,axis=1)
        
        # ------------------------------------------------------- #
        # Compute keypoint locations in the image plane
        # ------------------------------------------------------- #

        kpts_image, kpts_3d = self.compute_keypoint_coordinates(rotation_mat, 
                                                                translation_vec, 
                                                                self.k_mat_im)
        
        # compute the standard deviation
        distance = kpts_3d[-1, :]
        kpts_std = self.compute_std_distance(distance)

        # filter the keypoints that are not visible or too close to the image borders
        filtered = self.filter_points(kpts_image, margin=self.filter_margin)

        # ----------------------------------------------------- #
        # Create the output dictionary
        # ----------------------------------------------------- #
        
        sample = {
            "image": self.transform_input(img),
            
            "quaternion": quaternion,
            "rotation_mat": rotation_mat,
            "translation": translation_vec,
            
            "kpts_image": kpts_image.T,
            "kpts_3d": kpts_3d.T,
            "kpts_std": kpts_std,
            "kpts_filtered": filtered,
            
            "intrinsics_scaled": self.k_mat_im,
            "d_coef": self.d_coef
        }
        
        return sample
    
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
    
    def normalize_depth_safe(self, points, eps):
        """
           Normalize depth values safely to avoid division by zero errors.
           Adapted from Kornia: https://github.com/kornia/kornia
           Parameters:
               self (object): Instance of the class.
               points (numpy.ndarray): Array of points with depth values.
               eps (float): Small value to avoid division by zero.
           Returns:
               numpy.ndarray: Normalized points.
        """        
        # get depth
        z_vec = points[ -1:,...]

        # mask for non-zero/near-zero values
        mask = np.abs(z_vec) > eps
        
        # scale factor with division by zero handling
        scale = np.where(mask, 1.0 / (z_vec + eps), 1.0)
        
        return scale * points[:-1, ...]

    def apply_intrinsics(self, points, intrinsics):
        """
            Apply intrinsic camera parameters to a set of 3D points to obtain 2D pixel coordinates.

            Extracted from: kornia/kornia/geometry/conversions/denormalize_points_with_intrinsics.py
            https://github.com/kornia/kornia

            Parameters:
                self (object): Instance of the class.
                points (numpy.ndarray): Array of 3D points with shape (N, 3).
                intrinsics (numpy.ndarray): Intrinsic camera parameters matrix with shape (3, 3).

            Returns:
                numpy.ndarray: Array of 2D pixel coordinates with shape (N, 2).
        """
        
        # projection eq. [u, v, w]' = K * [x y z 1]'
        # u = fx * X + cx
        # v = fy * Y + cy    
            
        # Unpack coordinates
        x_coord = points[0, :]
        y_coord = points[1, :]

        # Unpack intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Apply intrinsics
        u_coord = x_coord * fx + cx
        v_coord = y_coord * fy + cy

        # Stack coordinates into a single array 2xN
        uv_coords = np.stack((u_coord, v_coord), axis=0)

        return uv_coords        
        
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
        k_output[0, :] *= ((cols)/(self.camera.ccx*2))
        k_output[1, :] *= ((rows)/(self.camera.ccy*2))
        
        return k_output
    
    def compute_keypoint_coordinates(self, rotation_mat, translation_vec, k_mat_im):
        """
            Compute the pixel coordinates of keypoints after applying a given rotation, translation, and camera matrix.

            Parameters:
                self (object): Instance of the class.
                rotation_mat (numpy.ndarray): Rotation matrix.
                translation_vec (numpy.ndarray): Translation vector.
                k_mat_im (numpy.ndarray): Intrinsic camera matrix.

            Returns:
                numpy.ndarray: Pixel coordinates of keypoints in the image.
                numpy.ndarray: 3Dcoordinates of keypoints in front of the camera.
        """     
        # compose the 3x4 transformation matrix
        rotation_mat = np.transpose(rotation_mat) 
        transform_mat = np.concatenate((rotation_mat, translation_vec), axis=1)
        
        # transform points to camera frame (3x4 * 4xN = 3xN)
        camera_frame_points = np.dot(transform_mat, self.kpts_h)
        
        # safely normalize by the distance (z coordinate)
        
        camera_frame_points = self.normalize_depth_safe(camera_frame_points, eps=1e-8)

        # project the points with the camera matrix
        kpts_image = self.apply_intrinsics(camera_frame_points, k_mat_im)    

        return kpts_image, camera_frame_points
    
    def filter_points(self, points, margin=0):
        """
            Filter points based on their positions within a margin from the image boundaries.

            Parameters:
                self (object): Instance of the class.
                points (numpy.ndarray): Array of points with shape (N, 2).
                margin (int): Margin distance from the image boundaries. Default is 0.

            Returns:
                numpy.ndarray: Boolean array indicating whether points are inside the margin or not.
        """
        flag_inside_x = np.bitwise_and(points[0,:] > margin, points[0,:] < (self.cols-margin))
        flag_inside_y = np.bitwise_and(points[1,:] > margin, points[1,:] < (self.rows-margin))
        flag_inside = np.bitwise_and(flag_inside_x, flag_inside_y)

        return flag_inside
    
    def quat2dcm(self, q):
        """
        Convert a quaternion to a direction cosine matrix (DCM).
        Extracted from: https://github.com/tpark94/speedplusbaseline

        Parameters:
            q (numpy.ndarray): Quaternion to be converted.

        Returns:
            numpy.ndarray: Direction cosine matrix (DCM) representation of the quaternion.
        """

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
    
    def find_max_min_distances(self):
        """
            Find the maximum and minimum distance for each keypoint based on ground truth data.

            Returns:
                Tuple[float, float]: A tuple containing the minimum and maximum distances of keypoints.
        """

        with open(os.path.join(self.speed_root, "synthetic", "train" + '.json'), 'r') as f:
            train_gt = json.load(f)
        with open(os.path.join(self.speed_root, "synthetic", "validation" + '.json'), 'r') as f:
            val_gt = json.load(f)
            
        # merge the train and val ground-truth
        train_gt.extend(val_gt)

        for value in train_gt:
            min_dist = float('inf')
            max_dist = -float('inf')

            # get the rotation and translation
            rot = np.expand_dims(value["q_vbs2tango_true"],axis=1)
            rot = self.quat2dcm(rot) 
            pos = np.expand_dims(value["r_Vo2To_vbs_true"],axis=1)
            pos = rot@pos


            kpts_cam = ((rot.T@(self.kpts+pos)).T)
            kpts_dist = kpts_cam[:,2]

            # find the maximum and minimum distance
            min_dist = min(min_dist, np.min(kpts_dist))
            max_dist = max(max_dist, np.max(kpts_dist))

        return min_dist, max_dist