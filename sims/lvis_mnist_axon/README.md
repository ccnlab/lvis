# LVis MNIST TE16deg

This is a "standard" version of the LVis model with a 16 degree receptive field width for the highest TE level neurons, implemented using the `axon` spiking activation algorithm, applied to the standard MNIST digit dataset: http://yann.lecun.com/exdb/mnist/

# MNIST Image handling

The `mnist_images.go` file manages access to the original gzipped (`.gz`) files, loading them whole into memory and providing lists of train and test items sorted by category (digit) as well as a flat list used in the environment.

The `ImagesEnv` environment in `images_env.go` applies V1-like gabor filtering to make the pixel images much sparser and non-overlapping in ways that reflect the way the visual system actually works.  It can add in-plane affine transformations (translation, scale, rotation) (now known as "data augmentation").  By default these are not used.

# Benchmark Results

Old (circa 2015) results at http://yann.lecun.com/exdb/mnist/ -- .23% generalization error is the best there, and anything under around 1% looks pretty reasonable.

A basic Leabra model on the raw pixel images does very poorly, with roughly 20-30% training and generalization error, due to the highly overlapping nature of the stimuli.

# Axon Results

Preliminary data shows the basic LVis axon model architecture learns to about 5% training and testing error.  It remains unclear how much of this residual error is due to noisy nature of spiking: using the top-2 criterion drops that down to about 1%.  Need to do more repeat testing to determine noise effects -- the raw testing performance does exhibit significant variability.

It looks like having the more focused 8 degree field of view in addition to the full 16 degree provides slightly better learning and generalization.

