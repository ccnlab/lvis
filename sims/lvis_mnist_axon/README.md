# LVis MNIST TE16deg

This is a "standard" version of the LVis model, implemented using the `axon` spiking activation algorithm, applied to the standard MNIST digit dataset: https://www.cs.toronto.edu/~kriz/cifar.html

The cemer `lvix_fix8.proj` version has "blob" color filters in addition to the monochrome gabor filters, which are also used here.

# CIFAR-10 Image handling

The `ImagesEnv` environment in `images_env.go` adds in-plane affine transformations (translation, scale, rotation) (now known as "data augmentation"), with the standard case being scaling in the range .5 - 1.1, rotation +/- 8 degrees, and translation using gaussian normal distribution with sigma of .15 and a max of .3.  The previous transforms had some kind of automatic border generating mechanism -- now this is handled by preserving the existing border, by using a smaller range of scales: .4 - 1.0.

# Benchmark Results

Old (circa 2015) results at https://www.cs.toronto.edu/~kriz/cifar.html

# Cemer standard performance

With standard params, the cemer versions of the model would learn to about 80% percent correct on the *training set*, with generalization accuracy of about 50%.  The inability of the `leabra` algorithm to continue to improve any further on the training set, despite many attempts to fix this problem, was an indication of some significant limitations in the scaling behavior.  Also, this large network is highly susceptible to the hogging problem, where a small subset of neurons lock in and take over the representational space, due to positive feedback loops present in bidirectionally connected networks.

The translation params have a big impact: .3 uniform is fairly difficult, and the gaussian distribution is helpful.

# Compute Speed

This is a good benchmark for performance.  On the `blanca` cluster, with 4 threads per MPI node and 16 nodes, it takes about 40 secs per 504 trials = 80 msec per trial (i.e., 80% of real time for 100 msec alpha cycle ;)

The Go leabra replication is about the same speed on the nominally faster hpc2 cluster, with 2 threads and 16 MPI nodes.

Spiking takes about 120 msec for 200 cycles so it is significantly faster per cycle as ex6pected.

# Params

This model is generally highly sensitive to parameters, and is an excellent platform for testing different parameters.

The cemer versions used fairly standard params (because they determined these params!), except:

* `XX1.Gain` = 80 instead of 100
* `Gbar.L` = .2 instead of .1
* `Inhib.Layer.FB` = 0 instead of 1, for layers with pool inhibition.

# TODO:

* no subpool
* no inhib
* no f8
* no cross between f8, f16
* no te

