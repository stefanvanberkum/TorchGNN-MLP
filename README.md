# TorchGNN-MLP
Integrated example of MLP functionality in ROOT's TorchGNN.

The code requires LibTorch, which can be downloaded from the [PyTorch homepage](https://pytorch.org/). We used version TorchLib 2.0.1 (cxx11 ABI).

The code can be compiled and run in the following way:
```
cd code_directory
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="path/to/libtorch" -DCMAKE_BUILD_TYPE="Release" ..
make
export OMP_NUM_THREADS=1
./TorchGNN
```
