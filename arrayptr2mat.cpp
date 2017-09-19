#include "arrayptr2mat.h"
// Usage:
// - data: a double array
// - rows: rows of the created mat array
// - cols: cols of the created mat array
// - out_file: .mat filename. the valuename in it is 'pData'
int arrayptr2mat(float* data, int rows, int cols, char* out_file){
    MATFile * pMatFile;
    pMatFile = matOpen(out_file, "w");
    if (!pMatFile) return -1;
    mxArray * pMxArray;
    pMxArray = mxCreateDoubleMatrix(rows, cols, mxREAL);
    if (!pMxArray) return -1;
    double *pData;
    pData = (double *)mxCalloc(rows*cols, sizeof(double));
    for (int i = 0; i < rows*cols; i++){
        pData[i] = double(data[i]);
    }
    mxSetData(pMxArray, pData);
    matPutVariable(pMatFile, "pData", pMxArray);
    mxFree(pData);
    matClose(pMatFile);
    return 1;
}
