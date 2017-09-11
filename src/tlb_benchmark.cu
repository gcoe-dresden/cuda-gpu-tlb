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

#include "helper.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

enum Metric { METRIC_AVG, METRIC_MIN };

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

// initialize data with the positions of the next entries - stride walks
void initSteps(unsigned int * data, unsigned long entries, unsigned int stepsKB){
    unsigned int pos = 0;
    while(pos < entries){
        data[pos] = pos + stepsKB / sizeof(int) * 1024;
        pos += stepsKB / sizeof(int) * 1024;
    }
}

// round to next power of two
unsigned int getNextPowerOfTwo (unsigned int x)
{
    unsigned int powerOfTwo = 1;

    while (powerOfTwo < x && powerOfTwo < (1u<<31))
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
        cerr << "usage: " << argv[0] << " data_from_MB data_to_MB stride_from_KB stride_to_KB Device_No=0 min_instead_avg=0" << endl;
        return 0;
    }

    float dataFromMB = atof(argv[1]);
    float dataToMB = atof(argv[2]);
    unsigned int dataFromKB = dataFromMB * 1024;
    unsigned int dataToKB = dataToMB * 1024;
    unsigned int tmpFrom = atoi(argv[3]);
    unsigned int tmpTo =atoi(argv[4]);
    Metric metric = METRIC_AVG;
    if (argc > 5)
        devNo =atoi(argv[5]);
    if (argc > 6 && strcmp(argv[6],"1")==0)
        metric = METRIC_MIN;

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

    if( strideToKB > dataFromKB ) {
        cerr << "data_from_MB*1024 must be greater or equal to stride_to_KB\n";
        exit(1);
    }

    unsigned int divisionTester = dataFromKB / strideToKB ;
    if ( divisionTester * strideToKB != dataFromKB ) {
        dataFromKB = (divisionTester * strideToKB);
        dataFromMB = dataFromKB / 1024;
    }

    divisionTester = dataToKB / strideToKB ;
    if ( divisionTester * strideToKB != dataToKB )
        dataToKB = (divisionTester * strideToKB);
        dataToMB = dataToKB / 1024;

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
    unsigned int stepsNo =  (dataToKB / strideFromKB) - (dataFromKB / strideFromKB) + 1;
    cout <<"# " <<  stepsNo << " steps for " << stridesNo << " strides " << endl;

    // ------------- open output file ------------

    char fileName[256];
    sprintf(fileName, "TLB-Test-%u-%u-%u-%u.csv", (unsigned int) dataFromMB, (unsigned int) dataToMB, strideFromKB, strideToKB);
    ofstream output (fileName);


    // ------------- setup Cuda and Input data and result data ------------

    size_t sizeMB = dataToMB+1;
    size_t sizeBytes = sizeMB*1024*1024;
    size_t sizeInts = sizeBytes / sizeof(int);

    int devCount;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));

    // check Dev Count
    if (devNo >= devCount){
        cerr << "Can not choose Dev " << devNo << ", only " << devCount << " GPUs " << endl;
        exit(0);
    }

    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, devNo));
    cout << "#" << props.name << ": cuda " << props.major << "." << props.minor;
    cout << ", reduction mode: " << (metric == METRIC_MIN ? "min" : "avg") << endl;
    output << "#" << props.name << ": cuda " << props.major << "." << props.minor;
    output << ", reduction mode: " << (metric == METRIC_MIN ? "min" : "avg") << endl;
    CHECK_CUDA(cudaSetDevice(devNo));

    unsigned int * hostData = new unsigned int[sizeInts];

    unsigned int * data;
    CHECK_CUDA(cudaMalloc(&data, sizeBytes));
    CHECK_CUDA(cudaMemset(data, 0, sizeBytes));

    // alloc space for results.
    const unsigned int init = metric == METRIC_AVG ? 0 : std::numeric_limits<unsigned int>::max();
    unsigned int** results = new unsigned int*[stepsNo];
    for(int i = 0; i < stepsNo; ++i){
        results[i] = new unsigned int[stridesNo];
        memset(results[i], init, sizeof(unsigned int) * stridesNo);
    }

    // ------------- actual evaluation is done here ------------

    // for each iteration
    for (unsigned int iter = 0; iter < iterations; iter++){
        cout << "iteration " << iter << " of " << iterations << endl;

        unsigned int indexX = 0;
        // for each stide size
        for (unsigned int steps = strideFromKB; steps <= strideToKB; steps*=2){
            // setup access strides in the input data
            memset(hostData, 0, sizeBytes);
            initSteps(hostData, sizeInts, steps);

            // copy data
            CHECK_CUDA(cudaMemcpy(data, hostData, sizeBytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());

            // run it once to initialize all pages (over full data size)
            TLBtester<<<1, 1>>>(data,  (unsigned int) ((sizeMB*1024 / steps)) );
            CHECK_CUDA(cudaDeviceSynchronize());

            unsigned int indexY = 0;
            // run test for all steps of this stride
            for (unsigned int i = dataFromKB/steps; i <= dataToKB/steps; i++){
                if (i == 0) continue;
                // warmup and initialize TLB
                TLBtester<<<1, 1>>>(data, i);
                CHECK_CUDA(cudaDeviceSynchronize());

                // real test
                TLBtester<<<1, 1>>>(data, i);
                CHECK_CUDA(cudaDeviceSynchronize());
                unsigned int myResult = 0;

                // find our result position:
                unsigned int pos =  (steps/sizeof(int) * 1024 * (i-1)) + 1;
                CHECK_CUDA(cudaMemcpy(&myResult, data+pos, sizeof(unsigned int), cudaMemcpyDeviceToHost));
                // write result at the right csv position
                if(metric == METRIC_AVG)
                    results[ indexY ][ indexX ] += myResult;
                else
                    results[ indexY ][ indexX ] = std::min(myResult, results[ indexY ][ indexX ]);

                indexY += steps / strideFromKB;
            }
            indexX++;
        }
    }

    // cleanup
    CHECK_CUDA(cudaFree(data));
    delete hostData;

    // ------------------------------------ CSV output --------------------------

    output << "#,";
    for (unsigned int steps = strideFromKB; steps <= strideToKB; steps*=2)
        output << steps << ",";
    output << endl;

    for(unsigned int y = 0; y < stepsNo; y++){
        output << dataFromMB + (float)(y * strideFromKB)/1024 << ",";
        for(unsigned int x = 0; x < stridesNo; x++) {
            if ( results[y][x] != 0 && results[y][x] != init) {
                if (metric == METRIC_MIN)
                    output << results[y][x];
                else
                    output << results[y][x] / iterations;
            }
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
