# About
VeryTinyCnn is a forward-only convolutional neural network implementation with multi-thread support powered by standard C++ 11
and optional AVX 2. And it's also a header-only library. All these makes it suitable for evaluating
neural networks with minimized dependency.

The main motivation for me to write VeryTinyCnn is that I can use it as a trick in my homework. It might not be
permitted to use non-standard libraries or too many lines of copied codes in my homework. So I decided to write
my own implementation for evaluating neural networks.

**Please note: VeryTinyCnn doesn't has the ability to train a nerual network. The training step should be finished with other
modern deep learning frameworks.**

**Also, it may be much slower than these frameworks, due to no GPU support, poor SIMD
instruction support, poor cache optimization and poor algorithm.**

# Feature
* Pure C++ 11 despite the optional AVX 2 code
* Multi-thread support: *It has really a vast speed boost depending on number of CPU cores. With many many CPU cores, it can
even run a little faster compared to some modern deep learning frameworks when tested out of box.*
* AVX 2 support **(disabled by default)**: *It only has a nearly 2x speed boost, due to my poor programming skill.*

## Layers
* 2-dimension convoltuional layer (AVX optimized)
* 2-dimension max pool layer
* Linear layer (AVX optimized)
* ReLU layer (AVX optimized)

# Compile and Run
`feature.cpp` is an example application of VeryTinyCnn. It uses Alexnet to extract feature and PCA to reduce feature dimension.
Besides VeryTinyCnn, it only depends on `CImg.h`. However, generating the modals and analyizing the feature require some other
Python libraries:

* `pytorch`: generating Alexnet data
* `sklearn`: generating PCA data and calculating tSNE with extracted features
* `matplotlib`: plotting tSNE result

To extract features from images, type

    make feature data/alexnet data/pca/nn-<feature-num>.dat
    ./feature -a data/alexnet -p data/pca/nn-<feature-num>.dat -v -o <output> <images>...

To plot extracted features (feature number set in Makefile) with tSNE, type

    make

Here is the help message for `feature`

    Usage: feature [DATA_OPTIONS]... [OPTION]... FILE...
    Data options:
      -a, --alexnet=FILE        binary Alexnet data
      -p, --pca=FILE            binary PCA data
    Options:
      -a, --alexnet=FILE        binary Alexnet data
      -p, --pca=FILE            binary PCA data
      -t, --threads=NUM         create NUM worker threads
      -s, --batch=NUM           set forward batch size
      -o, --output=FILE         set output file
      -b, --binary              set output mode to binary
      -v, --verbose             enable verbose mode
      -h, --help                print this help message
    Forward flow:
                    Alexnet        PCA
                 X ---------> Y ---------> Z

    At least one data option should be present to run this program. And forward
    flow is changed according to data options. Batch size and extra files is
    ignored in "Y -> Z" mode.
