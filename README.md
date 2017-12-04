# 3D Morphable Models as Spatial Transformer Networks

#### Update: A simple gradient descent method is added to show how the layers work. Please see the [demo.m](https://github.com/anilbas/3DMMasSTN/blob/master/demo.m).

This page shows how to use a 3D morphable model as a spatial transformer within a convolutional neural network (CNN). It is an extension of the original spatial transformer network in that we are able to interpret and normalise 3D pose changes and self-occlusions. The network (specifically, the localiser part of the network) learns to fit a 3D morphable model to a single 2D image without needing labelled examples of fitted models.

<p align="center">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/elon_musk_34.jpg" 
       alt="Elon Musk (34)" title="Elon Musk (34)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/christian_bale_51.jpg" 
       alt="Christian Bale (51)" title="Christian Bale (51)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/elisha_cuthbert_53.jpg" 
       alt="Elisha Cuthbert (53)" title="Elisha Cuthbert (53)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/clint_eastwood_62.jpg" 
       alt="Clint Eastwood (62)" title="Clint Eastwood (62)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/emma_watson_73.jpg" 
       alt="Emma Watson (73)" title="Emma Watson (73)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/chuck_palahniuk_48.jpg" 
       alt="Chuck Palahniuk (48)" title="Chuck Palahniuk (48)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/nelson_mandela_52.jpg"
       alt="Nelson Mandela (52)" title="Nelson Mandela (52)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/kim_jong-un_60.jpg" 
       alt="Kim Jong-un (60)" title="Kim Jong-un (60)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/ben_affleck_66.jpg"
       alt="Ben Affleck (66)" title="Ben Affleck (66)" width="19.4%">
  <img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/average/courteney_cox_127.jpg" 
       alt="Courteney Cox (127)" title="Courteney Cox (127)" width="19.4%">
</p>
<p align="center">
A set of mean flattened images that are obtained by applying the 3DMM-STN to multiple images of the same person from the <a href="http://www.umdfaces.io">UMDFaces Dataset</a>.<br><i>(Please hover over the image to see the subject's name and the number of images used for averaging)</i> 
</p>

The proposed architecture is based on a purely geometric approach in which only the shape component of a 3DMM is used to geometrically normalise an image. Our method can be trained in an unsupervised fashion, and thus does not depend on synthetic training data or the fitting results of an existing algorithm.

In contrast to all previous 3DMM fitting networks, the output of our 3DMM-STN is a 2D resampling of the original image which contains all of the high frequency, discriminating detail in a face rather than a model-based reconstruction which only captures the gross, low frequency aspects of appearance that can be explained by a 3DMM.

## Citation

Please cite the [following paper](http://arxiv.org/abs/1708.07199) if you use this work in your research:

A. Bas, P. Huber, W. A. P. Smith, M. Awais and J. Kittler. "3D Morphable Models as Spatial Transformer Networks". In Proc. ICCVW on Geometry Meets Deep Learning, to appear, 2017.

## Usage & Training

We train our network using the [MatConvNet](http://www.vlfeat.org/matconvnet/) library. Plese refer to the [installation page](http://www.vlfeat.org/matconvnet/install/) for the instructions.

In order to start the training, you need to create the resampled expression model first. To do that, you need (1) [Basel Face Model](http://faces.cs.unibas.ch/bfm), `01_MorphableModel.mat` and (2) [3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip), `Model_Expression.mat`. You can set the paths accordingly and run the `prepareExpressionBFM` function in the prepareModel folder to build a resampled expression model.

Finally, run the `dagnn_3dmmasstn.m` script to start the training.

<img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig1.png" alt="Overview of the 3DMM-STN" width="50%"><img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig2.png" alt="The grid generator network within a 3DMM-STN" width="50%">

#### Localiser Network

The localiser network is a CNN that takes an image as input and regresses the pose and shape parameters, theta (*θ* = **r**, **t**, *logs*, **α**). For our localiser network, we use the pre-trained [VGGFaces](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) architecture, delete the classification layer and add a new fully connected layer with 6 + *D* outputs. The pre-trained models can be downloaded from MatConvNet [model repository](http://www.vlfeat.org/matconvnet/pretrained/).

#### Grid Generator Network
Our grid generator combines a linear statistical model with a scaled orthographic projection. We apply a 3D transformation and projection to a 3D mesh that comes from the morphable model. The intensities sampled from the source image are then assigned to the corresponding points in a flattened 2D grid.

## UV texture space embedding for Basel Face Model
The output of our 3DMM-STN is a resampled image in a flattened 2D texture space in which the images are in dense, pixel-wise correspondence. In other words, the output grid is a texture space flattening of the 3DMM mesh. Specifically, we compute a Tutte embedding using conformal Laplacian weights and with the mesh boundary mapped to a square. To ensure a symmetric embedding we map the symmetry line to the symmetry line of the square, flatten only one side of the mesh and obtain the flattening of the other half by reflection. 

You can find the UV coordinates as [BFM_UV.mat file](https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat) in the util folder.
<p align="center">
<img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/UV.png" alt="The output grid visualisation using the mean texture" width="25%"><img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/geometry.png" alt="The mean shape as a geometry image" width="25%">
</p>

## Customised Layers

In this section, we summarise our customised layers and loss functions. Please refer to the [paper](http://arxiv.org/abs/1708.07199) for more details.

* **3D morphable model layer** generates a shape **X**, comprising *N* 3D vertices by taking a linear combination of principal components stored in the matrix and the mean shape, according to shape parameters **α**.
* **Axis-angle to rotation matrix layer** converts an axis-angle representation of a rotation, **r**, into a rotation matrix **R**.
* **3D rotation layer** takes as input a rotation matrix **R** and *N* 3D points **X**, and applies the rotation.
* **Orthographic projection layer** takes as input a set of *N* 3D points **X'** and outputs *N* 2D points **Y** by applying an orthographic projection along the *z* axis.
* **Scaling layers** scale the 2D points *Y* based on scale *s*, after the log scale *logs* transformed to scale *s*.
* **Translation layer** generates the 2D sample points by adding a 2D translation **t** to each of the scaled points.
* **Grid layer** takes as input 2x*N* points and produces 2x*H'W'* grid using re-sampled 3DMM which has *N=H'W'* vertices and each vertex *i*, has an associated UV coordinate. To understand how to compute the re-sampled model over [a uniform grid in the UV space](#uv-texture-space-embedding-for-basel-face-model), please refer to the `resampleModel` function and the sampling section of the paper.
* **Bilinear sampler** is a layer that is exactly as in the original STN.
* **Visibility (self-occlusions) layer** takes as input the rotation matrix **R** and the shape parameters **α** and outputs a binary occlusion mask **M**.
* **Masking layer** combines the sampled image and the visibility map via pixel-wise products.

#### Geometric Loss Functions

* **Bilateral symmetry loss** measures asymmetry of the sampled face texture over visible pixels.
* **Siamese multi-view fitting loss** penalises differences between multiple images of the same face in different poses.
* **Landmark loss** minimises the Euclidean distance between observed and predicted 2D points.
* **Statistical prior loss** minimises an appearance error, regularising the statistical shape prior (We scale the shape basis vectors such that the shape parameters follow a standard multivariate normal distribution).

## Dependencies

- **map_tddfa_to_basel.mat** file is supplied by James Booth. 

- **Basel Face Model** is freely available upon signing a license agreement via the [website](http://faces.cs.unibas.ch/bfm) of [Graphics and Vision Research Group, University of Basel](http://gravis.dmi.unibas.ch).

- **The expression model** is using the correspondence to the Basel Model provided by [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). The components originally come from [FaceWarehouse](http://gaps-zju.org/facewarehouse/).





