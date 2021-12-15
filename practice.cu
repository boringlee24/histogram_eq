#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#define BIN_COUNT 10
#define N_THREADS 512
#define N_TOTAL 1024
#define RANGE 100

using namespace std;

// GPU kernel for computing a histogram
__global__ void kernel(int *input, int *bins, int N, int N_bins, int DIV){
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        int bin = input[tid] / DIV;
//        bins[bin] += 1;
        atomicAdd(&bins[bin], 1);
    }
}

// Initializes our input array
// Takes:
//  a: array of integers
//  N: Length of the array
//  takes random number from 0-99
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % RANGE;
    }
}

int main(){
    // Declare our problem size
    int N = N_TOTAL;
    size_t bytes = N * sizeof(int);

    int N_bins = BIN_COUNT;
    size_t bytes_bins = N_bins * sizeof(int);


    // Allocate unified memory
    int *input = new int[N];
    int *bins = new int[N_bins];
    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&bins, bytes_bins);


    // Initialize the array
    init_array(input, N);

    // divisor for finding correct bin
    int DIV = (RANGE + N_bins - 1) / N_bins;

    // initialize bin to 0
    for(int i=0; i < N_bins; i++){
        bins[i] = 0;
    }
    
    // threads and blocks
    int THREADS = N_THREADS;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Launch the kernel
    kernel<<<BLOCKS, THREADS>>>(input, bins, N, N_bins, DIV);

    cudaDeviceSynchronize();

//    int tmp = 0;
//    for (int i = 0; i < N_bins; i++){
//        tmp += bins[i];
//    }
//
//    cout << "total number: " + to_string(tmp) << endl;

    // Write the data out for gnuplot
    ofstream output_file;
    output_file.open("histogram.dat", ios::out | ios::trunc);
    for(int i = 0; i < N_bins; i++){
        output_file << bins[i] << " \n";
    }
    output_file.close();

    return 0;
}
