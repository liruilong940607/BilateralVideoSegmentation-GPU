#define _DEBUG
#include "cutil.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_memory.h"
#include <sys/time.h>

#include "MirroredArray.h"
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
    
    MirroredArray<int> data_mirror(data, DATA_SIZE);
    MirroredArray<int> result_mirror(1);

    sumOfSquares<<<1, 1, 0>>>(data_mirror.device, result_mirror.device);
    result_mirror.deviceToHost();

    printf("sum: %d/n", result_mirror.host);
}
