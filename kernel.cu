#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

__global__ void Gaussian(float *input_d, float *output_d, int u_size){

	int idx = threadIdx.x;
	int idy = threadIdx.y;

	__shared__ float S[16][16];

	S[idy][idx] = input_d[(idy * (u_size + 1)) + idx];

	for (int i = 1; i < u_size; i++)
	{
		if ((idy + i) < u_size)
		{
			float Q = (-1)*(S[i - 1][i - 1] / S[i + idy][i - 1]);
			S[i + idy][idx] = S[i - 1][idx] + ((Q)* (S[i + idy][idx]));
		}
		__syncthreads();
	}

	output_d[idy * (u_size + 1) + idx] = S[idy][idx];
}

int main(int argc, char **argv){
	float *output_h = NULL;
	float *input_d,
	float *output_d;
	float *op_res, add, r;
	int unknowns, j;

	unknowns = 10;

	float input_h[11][10] = { 5393, 1190, 7142, 5286, 3511, 6181, 8170, 4859, 4792, 2336, 2362, 7596, 9238, 1654, 7061, 1840, 6848, 1481, 4925, 7249, 4966, 4472, 3297, 6822, 6247, 0627, 7489, 9559, 2693, 9945, 2189, 5459, 5647, 9772, 0052, 1868, 6421, 2763, 8424, 1441, 7702, 8918, 2616, 4178, 2148, 8236, 7316, 7422, 7102, 8063, 5179, 3831, 5220, 1054, 6207, 3404, 6501, 7209, 7743, 6141, 1947, 6527, 2443, 9385, 8557, 6354, 0306, 8450, 6675, 9345, 1334, 8023, 7956, 5500, 2870, 8104, 2402, 4818, 7570, 4004, 7356, 4645, 5433, 0413, 8130, 7636, 6461, 9986, 2148, 8236, -7316, 7422, 7102, 8063, 5179, 3831, -5220, 1054, -6207, 5032, 6929, 5927, 1588, 4605, 8211, 2082, 5780, 4686, 2656, 8745 };

	output_h = (float*)malloc(sizeof(float)*unknowns*(unknowns + 1));

	cudaMalloc(&input_d, sizeof(float)*(unknowns)*(unknowns + 1));
	cudaMalloc(&output_d, sizeof(float)*(unknowns)*(unknowns + 1));

	cudaMemcpy(input_d, input_h, sizeof(float)*unknowns*(unknowns + 1), cudaMemcpyHostToDevice);

	dim3 dimBlock(unknowns + 1, unknowns, 1);
	dim3 dimGrid(1, 1, 1);

	cudaEvent_t startEvent, stopEvent; 
	cudaEventCreate(&startEvent); 
	cudaEventCreate(&stopEvent); 
	float ms;

	cudaEventRecord(startEvent, 0);

	Gaussian << < dimGrid, dimBlock >> >(input_d, output_d, unknowns);

	cudaEventRecord(stopEvent, 0);

	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms, startEvent, stopEvent);

	cudaMemcpy(output_h, output_d, sizeof(float)*unknowns*(unknowns + 1), cudaMemcpyDeviceToHost);
	

	//data coming from GPU 
	printf("\nOutput from GPU\n\n");

	for (int i = 0; i< unknowns; i++)
	{
		for (int j = 0; j< unknowns + 1; j++)
		{
			printf("%f\n", output_h[i*(unknowns + 1) + j]);
		}
		printf("\n");
	}

	//Back substitution 
	op_res = (float*)malloc(sizeof(float)*(unknowns));
	for (int i = 0; i< unknowns; i++)
	{
		op_res[i] = 1.0;
	}

	for (int i = unknowns - 1; i >= 0; i--)
	{
		add = 0.0;

		for (j = unknowns - 1; j>i; j--)
		{
			add = add + op_res[j] * output_h[i*(unknowns + 1) + j];
		}
		r = output_h[i*(unknowns + 1) + unknowns] - add;
		op_res[i] = r / output_h[i *(unknowns + 1) + j];
	}

	//Displaying the Unknown Variables
	printf("\n\t\tUNKNOWNS\n\n");
	for (int i = 0; i<unknowns; i++)
	{
		printf("[x%d] = %+f\n", i, op_res[i]);
	}

	//Print Execution Time
	printf("\nKernel Performance...\n\n"); 
	printf("Execution Time = %f miliseconds \n\n", ms);

	free(input_h);
	free(output_h);
	cudaFree(input_d);
	cudaFree(output_d);
	cudaThreadExit();
	return 0;
}