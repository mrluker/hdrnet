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
I've trashed a lot of Ubuntu installs and had to start over again. Here are the steps:

#### Install the following dependencies:

### Install Git if not currently installed:
    $ sudo apt install git

### Install Pip if not currently installed:

    $ sudo apt install python-pip
    $ sudo pip install --upgrade pip
    
### Install setproctitle:

    $ sudo pip install setproctitle
    
### Install CMake:
Download here: https://cmake.org/download/ and extract it to the "home" folder. Then run the following:

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
    
### Install GLEW
download: https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.tgz/download
Extract it to the "home" folder. Then run the following

    
    $ sudo apt-get install build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev
    $ cd glew-2.1.0
    
Using GNU Make
    
    $ make
    $ sudo make install
    $ make clean
    $ cd ..
    
### Install Open CV

    $ sudo apt-get install build-essential
    $ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    $ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
    
    $ git clone https://github.com/opencv/opencv.git

    $ pip install opencv-python
    $ apt-get install libopencv-dev
    
### Install FFMPEG and LIBAV

    $ sudo add-apt-repository ppa:djcj/hybrid
    $ sudo apt-get update
    $ sudo apt-get install ffmpeg -y
# Install CUDA 8 and cuDNN
All the steps are below and is based off the following guide found here: https://medium.com/@RotekSong/tutorial-installing-cuda-8-cudnn-5-1-fc3ac216ead6
### Install Cuda 8
download the cuda deb[local] file here: https://developer.nvidia.com/cuda-80-ga2-download-archive
follow the instructions here: (they are also listed below) http://developer2.download.nvidia.com/compute/cuda/8.0/secure/Prod2/docs/sidebar/CUDA_Quick_Start_Guide.pdf?TK1-dsE5eqmaDkfXsf0_PvH6BKAzETQvh06qeuxMIR2j77oYH6_YFQRuE_7ml4sxrZrz7S3P-i_OIQBIPB64C59dIMe8oB2dPSEORTb0FJ7oV0uxN8u76TyecNQnPVfZfJDNQWl8BIP5b6kInmL4JkswUoVpziEByVQWKW1AzQGDTSU0

    $ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    $ sudo apt-get update
    $ sudo apt-get install cuda
Reboot, then
set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables:

    $ export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
    $ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
                         
Install a writable copy of the samples then build and run the nbody sample:

    $ cuda-install-samples-8.0.sh ~
    $ cd ~/NVIDIA_CUDA-8.0_Samples/5_Simulations/nbody
    $ make
    $ ./nbody
    
### Install cuDNN

step 2.1: Download & "save" cuDNN to the "home" folder https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.3/prod/8.0_20170926/cudnn-8.0-linux-x64-v7-tgz 
You have to register a Nvidia developer account.

    $ tar -zxvf cudnn-8.0-linux-x64-v7.tgz
    $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    $ sudo apt-get -f install
 
### Install HDF5
download HDF5 here: https://support.hdfgroup.org/ftp/HDF5/current/src/CMake-hdf5-1.10.1.tar.gz
    
    $ mkdir hdf5stuff
Extract the tar.gz to hdf5stuff. Then run the following:

    $ cd "hdf5stuff/CMake-hdf5-1.10.1"
    $ ctest -S HDF5config.cmake,BUILD_GENERATOR=Unix -C Release -VV -O hdf5.log
    $ chmod +x build-unix.sh
    $ ./build-unix.sh
Navigate to the root directory & create two symbolics to ensure later steps can find HDF5:

    $ cd /usr/lib/x86_64-linux-gnu
    $ sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so
    $ sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so 
    
### Install Boost

    $ sudo apt-get install libboost-all-dev
    
### Install Caffe

    $ sudo apt-get update
    $ sudo apt-get upgrade
save everything you have open
    
    $ reboot
Run this first:

    $ sudo apt-get install -y build-essential cmake git pkg-config
    $ sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
    $ sudo apt-get install -y libatlas-base-dev 
    $ sudo apt-get install -y --no-install-recommends libboost-all-dev
    $ sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
Download Caffe: https://github.com/BVLC/caffe/archive/rc5.zip
Extract it to the home folder
    
    $ cd caffe-rc5
    $ cp Makefile.config.example Makefile.config
    $ sudo apt-get install kate
    $ kate ./Makefile.config
    
In the Katy editor Ctrl+f to find this in Makefile.config..

    CUDA_DIR := /usr/local/cuda
    
and replace it with this:

    CUDA_DIR := /usr/local/cuda-8.0
    
Save the document

    $ cd python
    $ for req in $(cat requirements.txt); do sudo -H pip install $req; done
    $ cd ..
    
Edit the Makefile

    kate ./Makefile
    
Add this line before the line BUILD_DIR_LINK := $(BUILD_DIR):

    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/

Search for and replace this line:

    NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)

with the following line

    NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
    
Save and close

Also, open the file CMakeLists.txt 

    kate ./CMakeLists.txt
and add the following line:

    # ---[ Includes
    set(${CMAKE_CXX_FLAGS} "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")

Save and close (See the discussion at: https://github.com/BVLC/caffe/issues/4046)

When compiling with OpenCV 3.0 or errors show imread,imencode,imdecode or VideoCapture open your Makefile 

    kate ./Makefile

Ctrl+f "LIBRARIES += glog" and replace what is there with the following code:

    LIBRARIES += glog gflags protobuf leveldb snappy boost_system boost_filesystem m hdf5_hl hdf5 opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio
    
Save and close



Now we can install. I have added -j $(($(nproc) + 1)) to make the installs multi-threaded and speed them up.

    $ make all -j $(($(nproc) + 1))
    $ make test -j $(($(nproc) + 1))
    $ make runtest -j $(($(nproc) + 1))
    $ make pycaffe
    $ make distribute

Paste the following code at the bottom of the document
    
  
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
