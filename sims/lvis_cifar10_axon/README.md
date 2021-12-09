# LVis CIFAR-10 TE16deg

This is a "standard" version of the LVis model with a 16 degree receptive field width for the highest TE level neurons, implemented using the `axon` spiking activation algorithm, applied to the standard CIFAR-10 image dataset: https://www.cs.toronto.edu/~kriz/cifar.html

The cemer `lvix_fix8.proj` version has "blob" color filters in addition to the monochrome gabor filters, which are also used here.

# CIFAR-10 Image handling

The `cifar_images.go` file manages access to the original binary (`.bin`) files, loading them whole into memory and providing lists of train and test items sorted by category as well as a flat list used in the environment.

The `ImagesEnv` environment in `images_env.go`  applies V1-like gabor filtering to make the pixel images much sparser and non-overlapping in ways that reflect the way the visual system actually works.  It can add in-plane affine transformations (translation, scale, rotation) (now known as "data augmentation").  By default these are not used.

# Benchmark Results

Old (circa 2015) results: https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html -- ranged from roughly 20% error for early models, 10% for basic early DCNN's, and roughly 4% error for the best as of 2015.

A manual analysis of the dataset: http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/ suggested that the human performance level might be about 6% error, and noted that the dataset is quite varied in terms of images within each category.  Karpathy estimated that a 20% error level was "reasonable" given how inconsistent the images are.  It seems likely given these analyses that models are learning pure pattern discriminations, not actually recognizing the shapes, and relying on things like texture -- very similar overall to the issues with ImageNet.

In contrast, the CU3D100 dataset of rendered 3D objects on uniform backgrounds really tests the systematic representation of shape per se, and the same model used here achieves high levels of training and generalization.


# Axon Results

Current results are not very good overall, only getting roughly 50% correct on training and testing (very similar performance between the two), which improves to roughly 25% if you count the top two responses.  Need to do more repeat testing to determine noise effects -- the raw testing performance does exhibit significant variability.



