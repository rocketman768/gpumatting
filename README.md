# GPUMatting

The purpose of this software is to provide CUDA-accelerated matting algorithms
related to the research of Philip G. Lee. Natural image matting is the problem
of extracting an implied alpha blending layer from a single natural image.
It is an extremely hard problem of image processing and computer vision, due
to the extremely large size of the underlying linear systems. For this reason,
current algorithms tend to be far to slow to be practical. We develop algorithms
that are scalable and parallelizable, making new exciting applications possible.
This code will demonstrate these algorithms.

## Requirements

* cmake >= 2.8
* cuda-5
* gcc

## Compiling

    $ mkdir build # This MUST be outside the source code folder 'gpumatting'
    $ cd build
    $ cmake ../gpumatting
    $ make

## Running the Example

    ./bin/matting grad 10000 ../gpumatting/data/05.ppm ../gpumatting/data/05_scribs.pgm ../gpumatting/data/05_gt.pgm

You will find the computed alpha matte in alpha.pgm in the working directory.
You can find more example data at [alphamatting.com](http://www.alphamatting.com/datasets.php)
The explanation of the options and parameters is given if you run without any
arguments.
