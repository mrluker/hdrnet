# Deep Bilateral Learning for Real-Time Image Enhancements
Siggraph 2017

Visit our [Project Page](https://groups.csail.mit.edu/graphics/hdrnet/).

[Michael Gharbi](https://mgharbi.com)
Jiawen Chen
Jonathan T. Barron
Samuel W. Hasinoff
Fredo Durand

Maintained by Michael Gharbi (<gharbi@mit.edu>)

Tested on Python 2.7, Ubuntu 14.0, gcc-4.8.

## Disclaimer

This is not an official Google product.

## How to setup on a clean install of Ubuntu 16.04 with an NVIDIA pascal gpu

# Install Dependencies First

### Install Pip if not currently installed:

    sudo apt install python-pip
    sudo pip install --upgrade pip
    
### Install setproctitle:

    sudo pip install setproctitle
    
### Install CMake:
Download here: https://cmake.org/download/ and extract it to the "home" folder

    $ cd cmake-3.9.3
    $ ./configure
    $ make
    $ cd ..
    
### Install Fortran

    $ sudo apt-get install gfortran
    
### Install BLAS & LAPACK:
*(you have to install BLAS before LAPACK, because LAPACK needs it)

    $ git clone https://github.com/Reference-LAPACK/lapack-release.git
    $ cd lapack-release
    $ cp make.inc.example make.inc
    $ make FC=make.inc
    $ cd ..
    
### Install GLEW https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.tgz/download
extract it to the "home "
    
    $ sudo apt-get install build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev
    $ cd glew-2.1.0
    
Using GNU Make
    
    $ make
    $ sudo make install
    $ make clean
    $ cd ..
    
### Install Open CV

    $ pip install opencv-python
    
### Install FFMPEG and LIBAV

    $ sudo add-apt-repository ppa:djcj/hybrid
    $ sudo apt-get update
    $ sudo apt-get install ffmpeg -y

    
### NumPy Install:
    $ sudo pip install numpy
    $ sudo pip install numpy --upgrade
    
double check some things 

    $ sudo apt-get install gcc gfortran python-dev libblas-dev liblapack-dev cython
    
### pyglib Install:

    sudo pip install pyglib

### scikit_image & skimage
    $ sudo apt-get install python-skimage
    $ sudo pip install -U scikit-image
### Install Tensor Flow
*Install Distributions needed for Tensor Flow:

    $ pip install wheel
    $ pip wheel --wheel-dir=/tmp/wheelhouse pyramid
    
    $ sudo pip install html5lib
    
    $ sudo pip install bleach
    
    $ sudo pip install six
    
*Download ProtoBuf https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-python-3.4.1.tar.gz
and extract it to the "Home" folder
    
    $ sudo pip install prodobuf
    
    $ sudo pip install mock
    
    $ sudo pip install backports.weakref
    
    $ sudo pip install tensorboard
    
    $ sudo pip install tensorflow_gpu
    
### Install python_gflag

    $ git clone https://github.com/google/python-gflags.git
    $ cd python-gflags
    $ python setup.py build
    
### Install python_magic
    $ sudo pip install python_magic
    
### NVIDIA
    $ sudo apt-get install libcupti-dev
    $ export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
    $ sudo apt-get update
    $ sudo apt-get upgrade
    $ sudo apt install nvidia-cuda-toolkit
    
### Download HDRNET

    $ git clone https://github.com/google/hdrnet.git

    
    
    cd hdrnet
    pip install -r requirements.txt

# Build

Our network requires a custom Tensorflow operator to "slice" in the bilateral grid.
To build it, run:

    $ cd hdrnet/hdrnet
    $ make

To build the benchmarking code, run:

    cd benchmark
    make

Note that the benchmarking code requires a frozen and optimized model. Use
`hdrnet/bin/scripts/optimize_graph.sh` and `hdrnet/bin/freeze.py to produce these`.

To build the Android demo, see dedicated section below.

### Test

Run the test suite to make sure the BilateralSlice operator works correctly:

    cd hdrnet
    py.test test

### Download pretrained models

We provide a set of pretrained models. One of these is included in the repo
(see `pretrained_models/local_laplacian_sample`). To download the rest of them
run:

    cd pretrained_models
    ./download.py

## Usage

To train a model, run the following command:

    ./hdrnet/bin/train.py <checkpoint_dir> <path/to_training_data/filelist.txt>

Look at `sample_data/identity/` for a typical structure of the training data folder.

You can monitor the training process using Tensorboard:

    tensorboard --logdir <checkpoint_dir>

To run a trained model on a novel image (or set of images), use:

    ./hdrnet/bin/run.py <checkpoint_dir> <path/to_eval_data> <output_dir>

To prepare a model for use on mobile, freeze the graph, and optimize the network:

    ./hdrnet/bin/freeze_graph.py <checkpoint_dir>
    ./hdrnet/bin/scripts/optimize_graph.sh <checkpoint_dir>

You will need to change the `${TF_BASE}` environment variable in `./hdrnet/bin/scripts/optimize_graph.sh`
and compile the necessary tensorflow command line tools for this (automated in the script).


## Android prototype

We will add it to this repo soon.

## Known issues and limitations

* The BilateralSliceApply operation is GPU only at this point. We do not plan on releasing a CPU implementation.
* The provided pre-trained models were updated from an older version and might slightly differ from the models used for evaluation in the paper.
* The pre-trained HDR+ model expects as input a specially formatted 16-bit linear input. In summary, starting from Bayer RAW:
  1. Subtract black level.
  2. Apply white balance channel gains.
  3. Demosaic to RGB.
  4. Apply lens shading correction (aka vignetting correction).

  Our Android demo approximates this by undoing the RGB->YUV conversion and
  white balance, and tone mapping performed by the Qualcomm SOC. It results in slightly different colors than that on the test set. If you run our HDR+ model on an sRGB input, it may produce uncanny colors.
