#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void filter(float *outputs, float *inputs, float *positions, int pd, int vd, int f, int w, int h, bool accurate, int *gridsize, float *masks);
