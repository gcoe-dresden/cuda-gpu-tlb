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
#include <fstream>
#include <vector>

#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

#include <cuda_runtime.h>


using namespace std;

// --------------------------------- GPU Kernel ------------------------------

template<bool DISRUPT, bool GET_DURATION>
static __global__
void tlb_latency_with_disruptor(unsigned int * hashtable,
                                unsigned hashtable_count,
                                unsigned iterations,
                                unsigned stride_count,
                                unsigned offset,
                                int smid0,
                                int smxxx)
{
    extern __shared__ unsigned duration[]; // shared memory should be large enough to fill one SM

    unsigned smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid) );

    if(!(DISRUPT || smid==smid0)) // only take 1st SM in non-disrupting mode
        return;
    if(DISRUPT && smid!=smxxx) // only SMxxx does run in disrupting mode
        return;
    if(threadIdx.x!=0)
        return;

    unsigned long start;
    unsigned int sum = 0;
    unsigned int pos = DISRUPT ? (stride_count*iterations + offset) % hashtable_count : offset;
    sum += pos; // ensure pos is set before entering loop
    for (unsigned int i = 0; i < iterations; i++) {
        start = clock64();
        pos = hashtable[pos];
        sum += pos; // ensure pos is set before taking clock
        duration[i] = static_cast<unsigned>(clock64()-start);
    }
    if(sum == 0)
        hashtable[hashtable_count+1] = sum;
    if(GET_DURATION && smid==smid0) { // only store durations one time
        for (unsigned int i = 0; i < iterations; i++) {
            hashtable[hashtable_count+2+i] = duration[i];
        }
    }
}


// --------------------------------- support functions ------------------------------

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif

inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
    if (code != cudaSuccess) {
        cerr << "CUDA ERROR: " << cudaGetErrorString(code) << " in Line " << line << endl;
        exit(code);
    }
}



// --------------------------------- main part ------------------------------
int main(int argc, char **argv)
{
    unsigned int devNo = 0;
    
    // ------------- handle inputs ------------
      
    if (argc < 3) {
        cout << "usage: " << argv[0] << " stride_KB iterations device_No=0" << endl;
        return 0;
    }

    int stride_KB = atoi(argv[1]);
    int iterations = atoi(argv[2]);

    if (argc > 3)
        devNo = atoi(argv[3]);

    
    // --------------- init CUDA ---------
    int devCount;
    int SMcount =  0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    
    // check Dev Count
    if (devNo >= devCount){
        cout << "Can not choose Dev " << devNo << ", only " << devCount << " GPUs " << endl;
        exit(0);
    }
    CHECK_CUDA(cudaSetDevice(devNo));
        
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, devNo));
    cout << "#" << props.name << ": cuda " << props.major << "." << props.minor << endl;
    SMcount = props.multiProcessorCount;


    // --------------- setup input data ---------
    size_t hashtable_size_MB = ((iterations+1) * stride_KB * 2) / 1024;
    
    
    CHECK_CUDA(cudaDeviceReset());
      
    unsigned int * hashtable;
    unsigned* hduration = new unsigned [iterations];
    size_t N = hashtable_size_MB*1048576llu/sizeof(unsigned int);
     
    CHECK_CUDA(cudaMalloc(&hashtable, hashtable_size_MB*1048576llu+(iterations+2llu)*sizeof(unsigned int)));
      
    // init data
    unsigned int* hdata = new unsigned int[N+1];
    size_t stride_count = stride_KB*1024llu/sizeof(unsigned);
    for(size_t t=0; t<N; ++t) {
        hdata[t] = ( t+stride_count ) % N;
    }
    hdata[N] = 0;
    CHECK_CUDA(cudaMemcpy(hashtable, hdata, (N+1)*sizeof(unsigned), cudaMemcpyHostToDevice));
    delete[] hdata;

    // alloc output space
    double ** results = new double * [SMcount];
    for(int i = 0; i < SMcount; ++i){
        results[i] = new double[SMcount];
        memset(results[i], 0, sizeof(double) * SMcount);
    }
    
    
    // --------------- test all SMx to SMy combinations ---------

    for (int smid0 = 0; smid0 < SMcount; smid0++){
        for  (int smxxx = 0; smxxx < SMcount; smxxx++) {

            CHECK_CUDA(cudaDeviceSynchronize());
                
            // fill TLB
            tlb_latency_with_disruptor<false, false><<<2*SMcount, 1, iterations*sizeof(unsigned)>>>(hashtable, N,  iterations, stride_count, 0, smid0,smxxx);
              
            // disrupt TLB if TLB is shared
            tlb_latency_with_disruptor<true, false><<<2*SMcount, 1, iterations*sizeof(unsigned)>>>(hashtable, N,  iterations, stride_count, 0, smid0,smxxx);
              
            // check if values in TLB
            tlb_latency_with_disruptor<false, true><<<2*SMcount, 1, iterations*sizeof(unsigned)>>>(hashtable, N,  iterations, stride_count, 0, smid0,smxxx);
             
             
            CHECK_LAST( "Kernel failed." );
            CHECK_CUDA(cudaDeviceSynchronize());
    
            
            // get needed cycles
            CHECK_CUDA(cudaMemcpy(hduration, hashtable+N+2, iterations*sizeof(unsigned), cudaMemcpyDeviceToHost));
                    
            double avgc=0;
            for(int b=0; b<iterations;++b) {
                avgc+=hduration[b];
            }
            results[smid0][smxxx] =  avgc;
        }
    }
    
    CHECK_CUDA(cudaFree(hashtable));
    delete[] hduration;
    
 
    // ---------------output handling ---------
 
    cout << "#----------- absolute values ---------------" << endl;
 
    cout << "# ";
    for (unsigned int steps = 0; steps < SMcount; steps++)
        cout << steps << " ";
    cout << endl;    
    
    for(unsigned int y = 0; y < SMcount; y++){
        cout << y << " ";
        for(unsigned int x = 0; x < SMcount; x++){
            cout << (unsigned int) (results[y][x]/iterations);
            cout << " ";
        }
        cout << endl;
    }
    
    cout << "#----------- which SMs share TLB ---------------" << endl;
    
    cout << "# ";
    for (unsigned int steps = 0; steps < SMcount; steps++)
        cout << steps << " ";
    cout << endl;    
    
    for(unsigned int y = 0; y < SMcount; y++){
        cout << y << " ";
        
        double avg = 0;
        for(unsigned int x = 0; x < SMcount; x++)
            avg += (results[y][x]/iterations);
        // build average and add some buffer
        avg = (avg / SMcount)+3;
        
        
        //    cout << avg << endl;
                
        for(unsigned int x = 0; x < SMcount; x++){
            if (results[y][x]/iterations > avg) cout << ".X ";
            else cout << ".. ";
        }
        cout << endl;
    }

 
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
