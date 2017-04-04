/*-
 * Copyright (c) 2017 Tomas Karnagel and Matthias Werner
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 

using namespace std;



// --------------------------------- GPU Kernel ------------------------------
static __global__ void TLBtester(unsigned int * data, unsigned int iterations)
{

    unsigned long start = clock64();
    unsigned long stop = clock64();
    
    unsigned long sum = 0;
    unsigned int pos = 0;
    unsigned int posSum = 0;

    // warmup
    for (unsigned int i = 0; i < iterations; i++)
        pos = data[pos];
    if (pos != 0) pos = 0;
    
    
    
    for (unsigned int i = 0; i < iterations; i++){
        start = clock64();
        pos = data[pos];
        // this is done to add a data dependency, for better clock measurments
        posSum += pos;
        stop = clock64();
        sum += (stop-start);
    }

    // get last page
    if (pos != 0) pos = 0;
    for (unsigned int i = 0; i < iterations-1; i++)
        pos = data[pos];
    
    // write output here, that we do not access another page
    data[(pos+1)] = (unsigned int)((unsigned int)sum / (iterations));
    
    // if I don't write that, the compiler will optimize all computation away
    if (pos == 0) data[(pos+2)] = posSum;
}

// --------------------------------- support functions ------------------------------

// check Errors
#define checkCuda(x) { gpuAssert((x), __LINE__); }
inline void gpuAssert(cudaError_t code, int line)
{   
    if (code != cudaSuccess) {
      cerr << "CUDA ERROR: " << cudaGetErrorString(code) << " in Line " << line << endl;;
       exit(code);
       }
}

// initialize data with the positions of the next entries - stride walks
void initSteps(unsigned int * data, unsigned long entries, unsigned int stepsKB){
    
    unsigned int pos = 0;
    while(pos < entries){
        data[pos] = pos + stepsKB / sizeof(int) * 1024;
        pos = data[pos];
    }
}

// round to next power of two
unsigned int getNextPowerOfTwo (unsigned int x)
{
    unsigned int powerOfTwo = 1;
    
    while (powerOfTwo < x && powerOfTwo < 2147483648)
       powerOfTwo *= 2;
     
    return powerOfTwo;
}


// --------------------------------- main part ------------------------------
            
int main(int argc, char **argv)
{
    unsigned int iterations = 5;
    unsigned int devNo = 0;
        
    // ------------- handle inputs ------------
    
    if (argc < 5) {
        cout << "usage: " << argv[0] << " data_from_MB data_to_MB stride_from_KB stride_to_KB Device_No=0" << endl;
        return 0;
    }
     
     float dataFromMB = atof(argv[1]);
     float dataToMB = atof(argv[2]);
     unsigned int tmpFrom = atoi(argv[3]);
     unsigned int tmpTo =atoi(argv[4]);
     if (argc > 5)
         unsigned int devNo =atoi(argv[5]);

    // ------------- round inputs to power of twos ------------
     
     unsigned int strideFromKB = getNextPowerOfTwo(tmpFrom);
     unsigned int strideToKB = getNextPowerOfTwo(tmpTo);

    if (tmpFrom != strideFromKB) cout << "strideFrom: " << tmpFrom << " not power of two, I take " << strideFromKB << endl;
    if (tmpTo != strideToKB) cout << "strideTo: "<< tmpTo << " not power of two, I take " << strideToKB << endl;
     
     if (strideToKB < strideFromKB) {
         unsigned int tmp = strideToKB;
         strideToKB =strideFromKB;
         strideFromKB = tmp;
     }
     
     unsigned int divisionTester = ((unsigned int)(dataFromMB * 1024))/ strideToKB ;
     if ( divisionTester * strideToKB != (unsigned int)(dataFromMB * 1024) ) dataFromMB = (divisionTester * strideToKB) / 1024;

     divisionTester = ((unsigned int)(dataToMB * 1024))/ strideToKB ;
     if ( divisionTester * strideToKB != (unsigned int)(dataToMB * 1024) ) dataToMB = (divisionTester * strideToKB) / 1024;
     
     if (dataToMB < dataFromMB){
         float tmp = dataFromMB;
         dataFromMB = dataToMB;
         dataToMB = tmp;
     }
     
    cout << "#testing: from " << dataFromMB << "MB to " << dataToMB << "MB, in strides from " <<  strideFromKB << "KB to " << strideToKB << "KB -- " << iterations << " iterations" << endl;
    
    unsigned int tmp = strideFromKB;
    unsigned int stridesNo = 0;
    while (tmp <= strideToKB){
        stridesNo++;
        tmp *= 2;
    }
    unsigned int stepsNo =  ((((unsigned int)(dataToMB*1024))/ strideFromKB) - (((unsigned int)(dataFromMB*1024))/strideFromKB))+1;
    cout <<"# " <<  stepsNo << " steps for " << stridesNo << " strides " << endl;
        
    // ------------- open output file ------------    
        
    char fileName[500];
    sprintf(fileName, "TLB-Test-%u-%u-%u-%u.csv", (unsigned int) dataFromMB, (unsigned int) dataToMB, strideFromKB, strideToKB);
    ofstream output (fileName);

     
    // ------------- setup Cuda and Input data and result data ------------    
     
     size_t sizeMB = dataToMB+1;
        
    int devCount;
    checkCuda(cudaGetDeviceCount(&devCount));
    
    // check Dev Count
    if (devNo >= devCount){
        cout << "Can not choose Dev " << devNo << ", only " << devCount << " GPUs " << endl;
        exit(0);
    }
    
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, devNo));
    cout << "#" << props.name << ": cuda " << props.major << "." << props.minor << endl;
      output << "#" << props.name << ": cuda " << props.major << "." << props.minor << endl;

    checkCuda(cudaSetDevice(devNo));

    unsigned int * hostData = new unsigned int[sizeMB * 1024 * 1024 / sizeof(unsigned int)];
        
    unsigned int * data;
    checkCuda(cudaMalloc(&data, sizeMB * 1024 * 1024));
    checkCuda(cudaMemset(data, 0, sizeMB * 1024 * 1024));
    
    // alloc space for results.
    unsigned int** results = new unsigned int*[stepsNo];
    for(int i = 0; i < stepsNo; ++i){
        results[i] = new unsigned int[stridesNo];
        memset(results[i], 0, sizeof(unsigned int) * stridesNo);
    }
    
    // ------------- actual eveluation is done here ------------    
    
    // for each iteration
    for (unsigned int iter = 0; iter < iterations; iter++){
        cout << "iteration " << iter << " of " << iterations << endl;
    
        unsigned int indexX = 0;
        // for each stide size
        for (unsigned int steps = strideFromKB; steps <= strideToKB; steps*=2){
            
            // setup access strides in the input data
            memset(hostData, 0, sizeMB * 1024 * 1024);
            initSteps(hostData, (sizeMB/sizeof(int)) * 1024 * 1024, steps);
            
            // copy data
            checkCuda(cudaMemcpy(data, hostData, sizeMB * 1024 * 1024, cudaMemcpyHostToDevice));
            checkCuda(cudaThreadSynchronize());    
            
            // run it once to initialize all pages (over full data size)
            TLBtester<<<1, 1>>>(data,  (unsigned int)  ((sizeMB*1024 / steps)-5));
            checkCuda(cudaThreadSynchronize());    
            
            unsigned int indexY = 0;    
            // run test for all steps of this stride
            for (unsigned int i = ((unsigned int)(dataFromMB*1024))/steps; i <= ((unsigned int)(dataToMB*1024))/ steps ; i++){    
                if (i == 0) continue;
                
                // warmup and initialize TLB
                TLBtester<<<1, 1>>>(data, i);
                checkCuda(cudaThreadSynchronize());

                // real test
                TLBtester<<<1, 1>>>(data, i);
                checkCuda(cudaThreadSynchronize());    
                            
                unsigned int myResult = 0;
            
                // find our result position:
                unsigned int pos =  ((steps) / sizeof(int) * 1024 * (i-1))+1;
                checkCuda(cudaMemcpy(&myResult, data+pos, sizeof(unsigned int), cudaMemcpyDeviceToHost));
                
                // write result at the right csv position
                results[ indexY ][ indexX ] += myResult;
                
                indexY += steps / strideFromKB;
            }
            
            indexX++;
        }    
    }
    
    // cleanup
    checkCuda(cudaFree(data));
    delete hostData;

    // ------------------------------------ CSV output --------------------------
        
    output << "#,";
    for (unsigned int steps = strideFromKB; steps <= strideToKB; steps*=2)
        output << steps << ",";
    output << endl;    
    
    for(unsigned int y = 0; y < stepsNo; y++){
        output << dataFromMB + (float)(y * strideFromKB)/1024 << ",";
        for(unsigned int x = 0; x < stridesNo; x++){
            if (results[y][x] != 0) output << results[y][x]/iterations;
            output << ",";
        }
        output << endl;
    }
            
    output.close();

    cout << "result stored in " << fileName << endl;

    
    // cleanup
    for(int i = 0; i < stepsNo; ++i)
        delete[] results[i]; 
    delete[] results;


return 0;
}
