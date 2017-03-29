# CUDA GPU TLB Benchmarks

Micro-Benchmarks for Discovering TLB Cache Level Hierarchies.

## Requirements

- cmake 2.8+
- C++ compiler (tested with gcc5.3.0)
- CUDA (7.5, 8.0 or newer)

## Build

Go to the cuda-gpu-tlb directory (created by git clone ...):

```
mkdir release && cd release
cmake ..
make -j 2
```

It generates device codes for common Nvidia GPU architectures starting with Kepler architecture. If you know your architecture, just edit src/CMakeLists.txt.

## tlb-bench Usage

```
$ ./tlb-bench 
usage: ./tlb-bench data_from_MB data_to_MB stride_from_KB stride_to_KB Device_No=0
```

### Test for the K80 - generates a CSV file 

```
./tlb-bench 1 5 64 256
./tlb-bench 48 300 1024 4096
./tlb-bench 1500 5000 1024 4096
```

### Plotting needs R - generates a pdf file

```
cd supplemental/
chmod u+x tlb-benchmark-plot.r
./tlb-benchmark-plot.r TLB-Test-1-5-64-256.csv
./tlb-benchmark-plot.r TLB-Test-48-300-1024-4096.csv
./tlb-benchmark-plot.r TLB-Test-1500-5000-1024-4096.csv
```

or just use the provided Makefile.


## Known issues

Kepler and Pascal GPUs seem to work fine but we had some issues getting good results on Maxwell GPUs. This is future work.


# TLB Sharing

## tlb-sharing usage

```
./tlb-sharing 
usage: ./tlb-sharing stride_KB iterations device_No=0
```

try:

- stride_KB = page_size  and 
- iteration = #entries
- choose the parameters that (otherwise you wont see the wanted effects):
```
stride_kb * iterations < TLB but 2 * stride_kb * iterations > TLB
```

output should be something like this:

```
./sharing 2048 65
#Tesla K80: cuda 3.7
#----------- absolute values ---------------
# 0 1 2 3 4 5 6 7 8 9 10 11 12 
0 342 286 285 286 285 343 285 286 285 287 285 286 285 
1 293 346 287 287 287 286 346 287 288 287 287 286 287 
2 285 281 336 281 280 281 280 337 280 281 336 281 280 
3 286 280 281 337 281 280 281 280 338 280 282 337 281 
4 286 282 281 282 336 282 281 282 281 337 281 282 336 
5 348 290 291 291 291 347 291 290 291 290 291 290 291 
6 297 352 292 293 291 293 351 293 291 293 291 293 291 
7 290 284 341 284 285 284 285 340 285 284 341 284 285 
8 290 285 284 342 284 285 284 285 341 285 284 342 284 
9 291 285 286 285 341 285 286 285 286 340 286 285 341 
10 294 290 345 290 289 290 289 346 289 290 345 290 289 
11 296 289 290 346 290 289 290 289 347 289 290 346 290 
12 295 291 290 291 345 291 290 291 290 346 290 291 345 
#----------- which SMs interfere ---------------
# 0 1 2 3 4 5 6 7 8 9 10 11 12 
0 .X .. .. .. .. .X .. .. .. .. .. .. .. 
1 .. .X .. .. .. .. .X .. .. .. .. .. .. 
2 .. .. .X .. .. .. .. .X .. .. .X .. .. 
3 .. .. .. .X .. .. .. .. .X .. .. .X .. 
4 .. .. .. .. .X .. .. .. .. .X .. .. .X 
5 .X .. .. .. .. .X .. .. .. .. .. .. .. 
6 .. .X .. .. .. .. .X .. .. .. .. .. .. 
7 .. .. .X .. .. .. .. .X .. .. .X .. .. 
8 .. .. .. .X .. .. .. .. .X .. .. .X .. 
9 .. .. .. .. .X .. .. .. .. .X .. .. .X 
10 .. .. .X .. .. .. .. .X .. .. .X .. .. 
11 .. .. .. .X .. .. .. .. .X .. .. .X .. 
12 .. .. .. .. .X .. .. .. .. .X .. .. .X 
```

# Publication

... will be referenced here soon ...
