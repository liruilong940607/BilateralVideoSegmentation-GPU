#define _USE_MATH_DEFINES

#include <mat.h>
#include <math.h>
#include <iostream>
#include <stdio.h>

// Usage:
// - data: a double array
// - rows: rows of the created mat array
// - cols: cols of the created mat array
// - out_file: .mat filename. the valuename in it is 'pData'
int arrayptr2mat(float* data, int rows, int cols, char* out_file);
