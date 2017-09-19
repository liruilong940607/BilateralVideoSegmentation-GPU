#include "bilateralSpace.h"
#define DATA_SIZE 1048576

int data[DATA_SIZE];
void GenerateNumbers(int *number, int size)
{       
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    int i;
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i];
    }
    *result = sum;
}
void run(){
    GenerateNumbers(data, DATA_SIZE);
    int* gpudata, *result;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
            cudaMemcpyHostToDevice);
    sumOfSquares<<<1, 1, 0>>>(gpudata, result);
    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d/n", sum);
}
