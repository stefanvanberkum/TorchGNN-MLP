# TorchGNN-MLP

Integrated example of MLP functionality in ROOT's TorchGNN.

## Dependencies

- A Python virtual environment with:
    - PyTorch
    - ROOT
- C++
    - LibTorch: Can be downloaded from the [PyTorch homepage](https://pytorch.org/). We used version TorchLib 2.0.1
      (cxx11 ABI).
    - BLAS: We used [OpenBLAS](https://www.openblas.net/).

## How to run

The model can be trained and parsed by running the Python code ```MLP_generator.py```.

The C++ benchmarking code ```main.cxx``` can be compiled and run in the following way:

```
cd code_directory
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="path/to/libtorch" -DCMAKE_BUILD_TYPE="Release" ..
make
export OMP_NUM_THREADS=1
./TorchGNN
```

Alternatively, one can use the provided bash script ```collect_statistics.sh```, to collect all statistics mentioned in
the report. Make sure to change the CMAKE_PREFIX_PATH variable and activate the virtual environment before calling the bash script.
