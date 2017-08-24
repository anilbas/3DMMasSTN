# 3D Morphable Models as Spatial Transformer Networks

This page shows how to use a 3D morphable model as a spatial transformer within a convolutional neural network (CNN). It is an extension of the original spatial transformer network in that we are able to interpret and normalise 3D pose changes and self-occlusions. The network (specifically, the localiser part of the network) learns to fit a 3D morphable model to a single 2D image without needing labelled examples of fitted models.

<img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig1.png" alt="Overview of the 3DMM-STN" width="50%"><img src="https://github.com/anilbas/3DMMasSTN/blob/master/img/fig2.png" alt="The grid generator network within a 3DMM-STN" width="50%">

The proposed architecture is based on a purely geometric approach in which only the shape component of a 3DMM is used to geometrically normalise an image. Our method can be trained in an unsupervised fashion, and thus does not depend on synthetic training data or the fitting results of an existing algorithm.

In contrast to all previous 3DMM fitting networks, the output of our 3DMM-STN is a 2D resampling of the original image which contains all of the high frequency, discriminating detail in a face rather than a model-based reconstruction which only captures the gross, low frequency aspects of appearance that can be explained by a 3DMM.
