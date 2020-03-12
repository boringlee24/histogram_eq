
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
 
#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
                
// Add GPU kernel and functions
// HERE!!!
__global__ void probability_func(unsigned char *input, 
                       unsigned char *output,
                       unsigned int width,
                       unsigned int height){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    
	unsigned char value = input[location];


    output[location] = x%255;

}

__global__ void prefix_sum(double *output, double *input, int n)
{
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    temp[thid] = (thid > 0) ? input[thid] : 0;
    __syncthreads();

    for( int offset = 1; offset < n; offset <<= 1 )
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;

        if (thid >= offset)
            temp[pout*n+thid] += temp[pin*n+thid - offset];
        else
            temp[pout*n+thid] = temp[pin*n+thid];
        __syncthreads();
    }

    output[thid] = temp[pout*n+thid]; // write output
}

__global__ void probability_function(unsigned int *input, double *output, unsigned int size, int bucket_count)
{

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    	int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    	//int size = height*width;
    	int location = y*TILE_SIZE*gridDim.x+x;
    	if(location<bucket_count)
	{
		//printf("initial[%d]=%d\n",location,input[location]);
		double value = input[location];
                output[location] =value/size;
		//printf("probability[%d]=%lf\n",location,value/size);
	}
}

__global__ void frequency_function(unsigned char *input, unsigned int *output,int size, int bucket_size)
{

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
   /* 
	unsigned char value = input[location];
	int buck = ((int)value)/bucket_size;

    output[buck] += 1;
*/

	if (location < (size))
    	{
        	atomicAdd(&output[(unsigned int)(input[location])], 1);
        	//atomicAdd(&output[(unsigned int)(input[location] & 0xFF000000)], 1);
        	//atomicAdd(&output[(unsigned int)(input[location] & 0x00FF0000)], 1);
        	//atomicAdd(&output[(unsigned int)(input[location] & 0x0000FF00)], 1);
        	//atomicAdd(&output[(unsigned int)(input[location] & 0x000000FF)], 1);
    	}	

}

__global__ void cdf_normalization(double *input,double *output,int count, int bucket_count, double offset, double alpha)
{

    	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    	int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    	int location = 	y*TILE_SIZE*gridDim.x+x;
    
	if (location <bucket_count)
	{
		double value = input[location];
		output[location]=(value-offset)*(bucket_count-1)*alpha;
	}
}


__global__ void final_output(unsigned char *input,unsigned char *output,double *cdf,int bucket_size)
{

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;

	int location = 	y*TILE_SIZE*gridDim.x+x;
    
	unsigned char value = input[location];
	int buck =(int)value; 
	output[location]=cdf[buck];

}




__global__ void warmup(unsigned char *input,unsigned char *output)
{

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
	output[location] = 0;

}

// NOTE: The data passed on is already padded
void gpu_function(unsigned char *data,unsigned int height,unsigned int width)
{

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	int size = XSize*YSize;

	int bucket_count = 256;
	int bucket_size = 1;	
	//int max = 255/size;
	
	unsigned char *input_gpu;
	unsigned char *output_gpu;
	double *probability_vector;
    double *cdf_cpu_test_gpu;
	unsigned int *frequency_vector;
	double *cdf_vector;
	double probability_cpu_double[bucket_count];
    double cdf_cpu_test[bucket_count];
	unsigned int probability_cpu_int[bucket_count];
	double cdf_cpu[bucket_count];
	double *cdf_norm;
	unsigned int frequency_cpu[bucket_count];
	
	//int length = sizeof(data)/sizeof(data[0]);
	//printf("LENGTH == %d\n",length);
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&probability_vector  , bucket_count*sizeof(double)));
	checkCuda(cudaMalloc((void**)&cdf_cpu_test_gpu  , bucket_count*sizeof(double)));

	checkCuda(cudaMalloc((void**)&cdf_vector  , bucket_count*sizeof(double)));
        checkCuda(cudaMalloc((void**)&frequency_vector  , bucket_count*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&cdf_norm,bucket_count*sizeof(double)));
/*
	for(int i=0;i<width*height;i++)
	{
		printf("DATA[%d]=%s\n",i,data[i]);
	}
*/

	//Initiliaze probability_cpu to 0
    	for(int i=0;i<bucket_count;i++)
	{
		probability_cpu_int[i]=0;
	}
	
	for(int i =0;i<bucket_count;i++)
	{
		probability_cpu_double[i]=0;
	}	

    	// Copy data to GPU
    	checkCuda(cudaMemcpy(input_gpu, data,size*sizeof(char), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
	checkCuda(cudaMemcpy(probability_vector,probability_cpu_double,bucket_count*sizeof(double),cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(frequency_vector,probability_cpu_int,bucket_count*sizeof(unsigned int),cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    	// Execute algorithm

    	dim3 dimGrid(gridXSize, gridYSize);
    	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// Kernel Call
	#ifdef CUDA_TIMING
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        // Add more kernels and functions as needed here
        //norm_function<<<dimGrid, dimBlock>>>(input_gpu, output_gpu,width,height);
	frequency_function<<<dimGrid, dimBlock>>>(input_gpu,frequency_vector,size, bucket_size);
        
	checkCuda(cudaMemcpy(frequency_cpu,frequency_vector,bucket_count*sizeof(unsigned int),cudaMemcpyDeviceToHost));
	
	int count = 0;
	for(int i=0;i<bucket_count;i++)
	{
		count += frequency_cpu[i];
	} 

	printf("LENGTH = %d\n",count);
	probability_function<<<dimGrid, dimBlock>>>(frequency_vector,probability_vector,count,bucket_count);
	
        // From here on, no need to change anything
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());

    //prefix_sum<<<dimGrid, dimBlock>>>(probability_vector, cdf_cpu_test_gpu, bucket_count);

    //checkCuda(cudaMemcpy(cdf_cpu_test, cdf_cpu_test_gpu,bucket_count*sizeof(double),cudaMemcpyDeviceToHost));

	checkCuda(cudaMemcpy(probability_cpu_double,probability_vector,bucket_count*sizeof(double),cudaMemcpyDeviceToHost));
	/*
	int min;

	for(int i=0;i<256;i++)
	{
		if(probability_cpu_double[i]>0)
		{
			min = i;
		}	
	
	}

	if (max>0 && max <=150)
	{
		max = max+100;
	}
	else if(max>150 && max <=200)
	{
		max = max+50;
	}
	else if(max>200 && max<255)
	{
		max = max;
	}
	
	printf("MAX = %d",max);
*/	
	//double count = probability_cpu_double[0];
	cdf_cpu[0]= probability_cpu_double[0];
    ////printf("at 0, cdf = %f, cdf_test = %f\n", cdf_cpu[0], cdf_cpu_test[0]);
    printf("at 0, cdf = %f\n", cdf_cpu[0]);
  
	for(int i=1;i<bucket_count;i++)
	{
		cdf_cpu[i] = probability_cpu_double[i]+cdf_cpu[i-1];		
		//count = count+ probability_cpu_double[i];
        printf("at %d, cdf = %f\n", i, cdf_cpu[i]);
	}

    double offset, range,alpha;
	offset = cdf_cpu[0];
	range = cdf_cpu[bucket_count-1]-cdf_cpu[0];
	alpha = 1/range; 
	
/*	
	for(int i= 0;i<256;i++)
	{
		printf("probability[%d]=%lf \n",i,probability_cpu_double[i]);
	
	}

	for(int i= 0;i<256;i++)
	{
		printf("cdf[%d]=%lf\n",i,cdf_cpu[i]);
	
	}
*/	
	//printf("COUNT = %lf\n",count);

	checkCuda(cudaMemcpy(cdf_vector,cdf_cpu,bucket_count*sizeof(double),cudaMemcpyHostToDevice));

	cdf_normalization<<<dimGrid, dimBlock>>>(cdf_vector,cdf_norm,size, bucket_count, offset, alpha);

	final_output<<<dimGrid,dimBlock>>>(input_gpu,output_gpu, cdf_norm, bucket_size);
	#ifdef CUDA_TIMING
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data,output_gpu,size*sizeof(unsigned char),cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
	checkCuda(cudaFree(probability_vector));
	checkCuda(cudaFree(frequency_vector));
	checkCuda(cudaFree(cdf_vector));
	checkCuda(cudaFree(cdf_norm));

}

void gpu_warmup(unsigned char *data, unsigned int height,unsigned int width){
    
    	unsigned char *input_gpu;
    	unsigned char *output_gpu;
     
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
    	checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
            
    	// Copy data to GPU
    	checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
    // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
    	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    	warmup<<<dimGrid, dimBlock>>>(input_gpu, 
                                  output_gpu);
                                         
    	checkCuda(cudaDeviceSynchronize());
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
    	// Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
			
	
}


