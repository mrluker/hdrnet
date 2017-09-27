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

## Setup

### Dependencies

Install Pip if not currently installed:

    sudo apt install python-pip
    sudo pip install --upgrade pip
    
Install setproctitle:

    sudo pip install setproctitle
    
# NumPy Install:

    git clone https://github.com/numpy/numpy.git numpy
    
    sudo apt-get install gcc gfortran python-dev libblas-dev liblapack-dev cython
    
    cd numpy
    
    python setup.py build
    
Before building, you will also need to install packages BLAS and LAPACK that NumPy depends on:
you have to install BLAS before LAPACK, because LAPACK needs it

Download BLAS (Basic Linear Algebra Subprograms) http://www.netlib.org/blas/#_software

    
    cd BLAS-3.7.1
    make
    mv blas_LINUX.a libblas.a
    sudo cp libblas.a /usr/local/lib/
    
    
    
    
    cd hdrnet
    pip install -r requirements.txt

### Build

Our network requires a custom Tensorflow operator to "slice" in the bilateral grid.
To build it, run:

    cd hdrnet
    make

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
