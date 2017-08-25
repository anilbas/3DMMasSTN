# 3D Morphable Models as Spatial Transformer Networks

This page shows how to use a 3D morphable model as a spatial transformer within a convolutional neural network (CNN). It is an extension of the original spatial transformer network in that we are able to interpret and normalise 3D pose changes and self-occlusions. The network (specifically, the localiser part of the network) learns to fit a 3D morphable model to a single 2D image without needing labelled examples of fitted models.

<img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig1.png" alt="Overview of the 3DMM-STN" width="50%"><img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig2.png" alt="The grid generator network within a 3DMM-STN" width="50%">

The proposed architecture is based on a purely geometric approach in which only the shape component of a 3DMM is used to geometrically normalise an image. Our method can be trained in an unsupervised fashion, and thus does not depend on synthetic training data or the fitting results of an existing algorithm.

In contrast to all previous 3DMM fitting networks, the output of our 3DMM-STN is a 2D resampling of the original image which contains all of the high frequency, discriminating detail in a face rather than a model-based reconstruction which only captures the gross, low frequency aspects of appearance that can be explained by a 3DMM.

## Citation

Please cite the [following paper](http://arxiv.org/abs/1708.07199) if you use this work in your research:

A. Bas, P. Huber, W. A. P. Smith, M. Awais and J. Kittler. "3D Morphable Models as Spatial Transformer Networks". In Proc. ICCVW on Geometry Meets Deep Learning, to appear, 2017.

## Usage & Training

We train our network using the [MatConvNet](http://www.vlfeat.org/matconvnet/) library. Plese refer to the [installation page](http://www.vlfeat.org/matconvnet/install/) for the instructions.

In order to start the training, you need to create the resampled expression model first. To do that, you need (1) [Basel Face Model](http://faces.cs.unibas.ch/bfm), `01_MorphableModel.mat` and (2) [3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip), `Model_Expression.mat`. You can set the paths accordingly and run the prepareExpressionBFM function in the prepareModel folder to build a resampled expression model.

Finally, run the dagnn_3dmmasstn.m script to start the training.

#### Localiser Network

The localiser network is a CNN that takes an image as input and regresses the pose and shape parameters, theta (theta = R, t, logs, alpha). For our localiser network, we use the pre-trained [VGGFaces](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) architecture, delete the classification layer and add a new fully connected layer with 6 + D outputs. The pre-trained models can be downloaded from MatConvNet [model repository](http://www.vlfeat.org/matconvnet/pretrained/).

#### Grid Generator Network
Our grid generator combines a linear statistical model with a scaled orthographic projection. We apply a 3D transformation and projection to a 3D mesh that comes from the morphable model. The intensities sampled from the source image are then assigned to the corresponding points in a flattened 2D grid.

## UV texture space embedding for Basel Face Model
The output of our 3DMM-STN is a resampled image in a flattened 2D texture space in which the images are in dense, pixel-wise correspondence. In other words, the output grid is a texture space flattening of the 3DMM mesh. Specifically, we compute a Tutte embedding using conformal Laplacian weights and with the mesh boundary mapped to a square. To ensure a symmetric embedding we map the symmetry line to the symmetry line of the square, flatten only one side of the mesh and obtain the flattening of the other half by reflection. 

You can find the UV coordinates as [BFM_UV.mat file](#) in the util folder.
<p align="center">
<img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/UV.png" alt="The output grid visualisation using the mean texture" width="25%"><img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/geometry.png" alt="The mean shape as a geometry image" width="25%">
</p>

## Customised Layers

In this section, we summarise our customised layers and loss functions. Please refer to the [paper](http://arxiv.org/abs/1708.07199) for more details.

## Dependencies

- **map_tddfa_to_basel.mat** file is supplied by James Booth. 

- **Basel Face Model** is freely available upon signing a license agreement via the [website](http://faces.cs.unibas.ch/bfm) of [Graphics and Vision Research Group, University of Basel](http://gravis.dmi.unibas.ch).

- **The expression model** is using the correspondence to the Basel Model provided by [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). The components originally come from [FaceWarehouse](http://gaps-zju.org/facewarehouse/).





