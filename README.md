# 2D BMC
2 dimensional Backward Monte Carlo scheme example, parallelized with MPI and openMP

A forward Monte Carlo scheme is included for benchmark

## Compile

`make clean && make`

`make clean && ENABLEMPI=1 make` (MPI)

## Run

`./bmc`

`mpiexec -n 2 bmc` (MPI)

The output of the last timestep is stored in `out.txt`, with format:

[X] [V] [i] [Probability]

where `X` and `V` are the space and velocity mesh indices and `i = X + Nx * V`

## TODO

- Modify code and data structures according to ASCOT code
- Generalize the code to 6D
- Comparison with existing simulations
- Add SIMD vectorization
- optimize openMP
