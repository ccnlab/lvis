# LVis CU3D100 TE16deg

This is the "standard" version of the LVis model, implemented using the `axon` spiking activation algorithm, with the architecture tracing back to the `cemer` C++ versions that were originally developed (`lvis_te16deg.proj` and `lvis_fix8.proj`).

The `lvix_fix8.proj` version has "blob" color filters in addition to the monochrome gabor filters, and has the capacity to fixate on different regions in the image, but this was never fully utilized.  This Go implementation has the blob color filters, but no specified fixation -- just random 2D planar transforms.

# Images: CU3D100

This [google drive folder](https://drive.google.com/drive/folders/13Mi9aUlF1A3sx3JaofX-qzKlxGoViT86?usp=sharing) has .png input files for use with this model.

`CU3D_100_plus_renders.tar.gz` is a set of 30,240 rendered images from 100 3D object categories, with 14.45 average different instances per category, used for much of our object recognition work, including the above paper and originally: O'Reilly, R.C., Wyatte, D., Herd, S., Mingus, B. & Jilk, D.J. (2013). Recurrent Processing during Object Recognition. *Frontiers in Psychology, 4,* 124. [PDF](https://ccnlab.org/papers/OReillyWyatteHerdEtAl13.pdf) | [URL](http://www.ncbi.nlm.nih.gov/pubmed/23554596)

The image specs are: 320x320 color images. 100 object classes, 20 images per exemplar. Rendered with 40° depth rotation about y-axis (plus horizontal flip), 20° tilt rotation about x-axis, 80° overhead lighting rotation.

The `ImagesEnv` environment in `images_env.go` adds in-plane affine transformations (translation, scale, rotation) (now known as "data augmentation"), with the standard case being scaling in the range .5 - 1.1, rotation +/- 8 degrees, and translation using gaussian normal distribution with sigma of .15 and a max of .3.  The previous transforms had some kind of automatic border generating mechanism -- now this is handled by preserving the existing border, by using a smaller range of scales: .4 - 1.0.

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

