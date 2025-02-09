# Spacecraft Pose Estimation Tutorial

## Overview
Monocular spacecraft pose estimation involves determining the position and orientation of a target spacecraft from a single monocular image. This process relies on identifying keypoints, which are specific 3D locations on the spacecraft with known spatial coordinates (x, y, z) relative to the spacecraft’s coordinate center.

The method works by finding the pixel coordinates of each 3D keypoint in the image. Essentially, given an image of the spacecraft, the goal is to locate the 2D coordinates of these keypoints. Once the correspondences between the 2D and 3D keypoints are established, a Perspective-n-Point solver or a similar method can be used to derive the spacecraft’s pose.

There are many datasets for spacecraft pose estimation. However, in most cases, the only ground-truth information available is the pose and camera data (see Chapter \ref{chapter:SPIN}). Assuming a predefined set of 3D keypoints (which will be discussed later), this section explains how to obtain the ground-truth keypoint image coordinates needed to train your model, how to generate the heatmaps, and other important considerations.


### Computing Keypoint Locations

The ground-truth pose expresses the transformation between the camera coordinate system and the target spacecraft coordinate system. The ground-truth 3D spacecraft keypoints are expressed with respect to the target coordinate system. Hence, our first task is to transform the 3D spacecraft points to the camera coordinate system (step 1 in Figure 1.). 

To do so, we are going to build the transformation matrix $T = [R|t] \in \mathbb{R}^{3 \times 4}$ by appending the ground-truth translation vector $t \in \mathbb{R}^3$ to the ground-truth rotation matrix $R \in \mathbb{R}^{3\times 3}$. 

Please note that in the code accompanying this tutorial, which deals with the SPEED+ dataset, $R$ transforms from the camera frame to the target frame, which is why we invert it (this might be the case in other datasets). In addition, you will notice from other implementations that it is common to make $T$ homogeneous, i.e., append a row vector $v = (0,0,0,1)$ to the transformation matrix to make it square ($T \in \mathbb{R}^{4 \times 4}$). This step is useful when dealing with many transformations that have to be concatenated one after the other or if you want your software to be as general as possible. However, for this particular tutorial, we are going to stick with $T \in \mathbb{R}^{3 \times 4}$. 

The next step is to express the 3D spacecraft keypoints in homogeneous coordinates, i.e., $P^i = (x,y,z,w)$ with $w=1$, which effectively means appending a row of ones to the points. We denote these points by $P \in \mathbb{R}^{4 \times N}$. Then, the points in the camera frame are obtained by

![keypoint-projection (1)-1](https://github.com/user-attachments/assets/7afbe94a-db41-4bf6-9537-39abbd0c59dd)
Figure 1. Representation of the keypoint location computation pipeline. The first step is to transform the 3D keypoints, expressed in the target frame, to the camera frame. The next step involves to project them to the image space to obtain their final locations.


Once the points are expressed in camera coordinates, the final step is to project them into the image plane (step 2 in Figure 1.). This is accomplished by first normalising the points by the depth ($z$ component), with $\bar{P}^i_{c} = (x/z, y/z, 1)$ and then applying the camera intrinsics $K$ to obtain the image coordinates $(u,v)^i$ of the $i$-th keypoint. The camera intrinsics $K$ is of the form

$$
K = \begin{pmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{pmatrix}
$$

where:
- $f_x$ and $f_y$ are the focal lengths in the x and y directions, respectively.
- $c_x$ and $c_y$ are the coordinates of the principal point (usually the center of the image).

Finally, the pixel coordinates of the 3D keypoint locations can be obtained via:

$$
u^i = f_x \bar{P}^i_c(x) + c_x
$$

$$
v^i = f_y \bar{P}^i_c(y) + c_y
$$

**Handling Image Resizing**: the camera intrinsics allow us to project 3D points into 2D images (and vice versa if the depth of the keypoint is known). The intrinsics provided in datasets correspond to the original resolution of the image (e.g., 1900×1200 px), which is often too large to efficiently train a model. When rescaling images, the intrinsics must also be rescaled to ensure that the keypoint coordinates remain accurate.  

Let the original image dimensions be $w$ (width) and $h$ (height), and the new image dimensions be $w'$ and $h'$. The scaling factors are:

$$
s_x = \frac{w'}{w}, \quad s_y = \frac{h'}{h}
$$

The scaled camera intrinsics matrix $K'$ is then:

$$
K' = \begin{pmatrix}
s_x f_x & 0 & s_x c_x \\
0 & s_y f_y & s_y c_y \\
0 & 0 & 1
\end{pmatrix}
$$

**Handling Image Distortion:** The image distortion is represented by distortion coefficients. The common practice is to undistort the image using these coefficients to ensure that it adheres to the pinhole camera model. This approach greatly simplifies any further transformations applied to the keypoints.  An alternative, though less recommended, is to apply distortion directly to the computed keypoint locations.


### Heatmap Generation

A heatmap encodes the probability of a keypoint being in a given 2D position. A heatmap $H \in \mathbb{R}^{N \times M \times C}$ is generated by placing circular 2D Gaussians with a fixed standard deviation centered around the image ground-truth keypoint positions. $N$ and $M$ represent the spatial coordinates. $C$ represents the number of channels—defined by the number of keypoints—with each channel containing one unique keypoint.

**Standard Deviation**: Most papers in the literature employ a standard deviation of 1 pixel. This choice seems to be motivated by the original work from [Tompson et al.](https://github.com/max-andr/joint-cnn-mrf), where the authors use heatmaps with a resolution of 90×60 pixels and a standard deviation of 1 pixel. Although this specific value is not explicitly stated in the paper, it is referenced in the associated GitHub implementation and described in subsequent work by [Newell et al.](https://arxiv.org/abs/1603.06937). 

Unlike the 90×60 pixel heatmaps, in the following sections, we employ significantly larger heatmaps at 512×512 pixels. To ensure that our scaled heatmaps maintain a similar coverage area with respect to the 90×60 pixel heatmaps at 1 pixel standard deviation, we use a value of 7 pixels for the standard deviation. This can be derived from the average scaling factor between heatmap sizes:

$$
\text{scale factor} = 0.5 \times \left(\frac{512}{90} + \frac{512}{60} \right) = 0.5 \times (5.68 + 8.53) = 0.5 \times 14.21 = 7
$$

Following the literature, we keep the standard deviation value of the heatmaps constant for every pose, independent of factors such as ambient illumination. An interesting research direction would be to scale the heatmap standard deviation with the distance to the spacecraft. This approach would ensure consistent keypoint coverage regardless of distance.

**Scaling**: Heatmaps represent the probability of a keypoint appearing at a given image coordinate. Ideally, the heatmaps should be scaled so that the Gaussian sums to one. However, this often leads to numerical instability and is not suitable for training. The common convention in pose estimation is to scale the Gaussian so that the maximum value is 1. However, this value might not be suitable for some datasets, particularly those with large black backgrounds, which may require larger values around 100.

**Practical considerations**: How the heatmap is generated affects both the speed of the training process and the numerical resolution of the heatmap. Below, we outline a trade-off between three different approaches to generating heatmaps:

- **Generating heatmaps online in the data loader**: This is the classical approach, offering high flexibility as any changes in image dimensions are automatically captured. However, this method may be slow if you lack many CPU cores or if you use large batch sizes or image resolutions.

- **Pre-computing and storing heatmaps in memory**: Each storage format has its own advantages and limitations:
  - **PNG files** offer improved processing speed compared to generating heatmaps online. However, PNG is limited to only three channels (or four with an alpha channel), requiring $\lceil k/3 \rceil$ images per training sample, where $k$ is the number of keypoints.
  - Additionally, PNG files support only 256 intensity values per channel, leading to a loss of numerical resolution in the heatmap. To overcome this, one could store a single heatmap per PNG file, allowing 24-bit encoding, but at the cost of increased storage and loading times.
  - **NumPy arrays** preserve full precision, but they may demand more storage space and longer encoding-decoding times.

- **Computing heatmaps online in the GPU**: This approach, used in this tutorial, provides fast generation times while maintaining the flexibility to modify heatmap parameters in real time. It also enables the integration of specialized loss functions and end-to-end optimization by allowing gradient propagation through the heatmap generation process.

## Training

### Data Augmentation

When applying augmentations for spacecraft pose estimation, caution is necessary depending on the type of augmentation. Augmentations can be broadly divided into two types: **intensity-based** and **geometric**. 

- **Intensity-based augmentations** modify the intensity values of the image while preserving its geometric information. These can be applied without issues as they only affect pixel values.
- **Geometric augmentations** alter the image's geometry, such as through rotation, cropping, and perspective projection. These require caution because they can affect the ground-truth pose. For example, if an image is rotated, the keypoint locations or heatmap must be rotated accordingly. Additionally, if the ground-truth pose is used as part of the loss function, it is important to account for changes due to geometric transformations.

### Models and Losses

The models for regressing keypoint heatmap locations follow an **encoder-decoder** architecture. In particular, this tutorial provides two types of encoders:  
- A **ResNet-based encoder**  
- A **Visual Transformer (ViT)-based encoder**  

These models serve as strong baselines that are relatively easy to train and understand.

A common objective function is the **Mean Squared Error (MSE)** between the ground-truth heatmap $H$ and each predicted heatmap $\hat{H}$:

$$
\ell_{MSE} = \frac{1}{2NMC} \Vert \hat{H} - \beta H \rVert_F^2,
$$

where $\beta$ is a parameter that scales the ground-truth Gaussian heatmap.

## Testing

To estimate the pose, there are three main steps:

1. **Keypoint Prediction**: Pass an image through the model to obtain a heatmap prediction.
2. **Keypoint Localization**: Extract the keypoint locations, typically by sampling the positions with maximum values from the heatmap.
3. **Pose Estimation**: Use a **Perspective-n-Point (PnP)** algorithm to compute the pose by matching 2D keypoints with their corresponding 3D locations.

Optionally, the keypoints can be filtered based on their response, which can be beneficial in cases where the model struggles due to domain gaps.


