
#include "mex.h"
#include "GCoptimization.h"
#include "GraphCut.h"
#include <stdlib.h>

/* Defines */


/*
 * Matlab wrapper for Weksler graph cut implementation
 *
 * usage:
 * [gch] = GraphCutConstrSparse(dc, sc, SparseSc)
 *
 * Note that data types are crucials!
 * 
 * Inputs:
 *  dc - of type float, array size [#labels*#nodes], the data term for node
 *             n recieving label l is stroed at [n*#labels + l]
 *  sc - of type float, array size [#labels.^2] the cost between l1 and l2 is
 *                   stored at [l1+l2*#labels] = Vpq(lp,lq)
 *  SparseSc - Sparse matrix of type double defining both the graph structure 
 *              and spatially dependent smoothness term
 *
 * Outputs:
 *  gch - of type int32, graph cut handle - do NOT mess with it!
 */


//    int		  nlhs, 	/* number of expected outputs */
//    mxArray	  *plhs[],	/* mxArray output pointer array */
//    int		  nrhs, 	/* number of inputs */
//    const mxArray	  *prhs[]	/* mxArray input pointer array */

GraphHandle* GraphCutConstrSparse(Mat* dc, Mat* sc, SparseMat* SparseSc, int npixels, int nlabels) {   

    GCoptimization::PixelType num_pixels;
    int num_labels;
    
    num_pixels = npixels;//mxGetN(prhs[2]);
    num_labels = nlabels;//mxGetNumberOfElements(prhs[0])/num_pixels;
    
    /* construct the graph */
    GCoptimization* MyGraph = new GCoptimization(num_pixels, num_labels, SET_ALL_AT_ONCE, SET_ALL_AT_ONCE);
    
    /* set the nieghbors and weights according to sparse matrix */
    
    double   *pr;
    mwIndex  *ir, *jc;
    mwSize   col, total=0;
    mwIndex  starting_row_index, stopping_row_index, current_row_index;

  
    /* Get the starting positions of all four data arrays. */
    pr = mxGetPr(prhs[2]);
    ir = mxGetIr(prhs[2]);
    jc = mxGetJc(prhs[2]);
    
    for (col=0; col<num_pixels; col++)  {
        starting_row_index = jc[col];
        stopping_row_index = jc[col+1];
        if (starting_row_index == stopping_row_index) {
            continue;
        } else {
            for (current_row_index = starting_row_index;
                current_row_index < stopping_row_index;
                current_row_index++)  {
                    /* use only upper triangle of matrix */
                    if ( ir[current_row_index] >= col ) {
                        MyGraph->setNeighbors(ir[current_row_index], col, pr[total++]);
                    } else {
                        total++;
                    }
                    
            }
        }
    }
    Graph::captype *DataCost = (Graph::captype*)mxGetData(prhs[0]);
    Graph::captype *SmoothnessCost = (Graph::captype*)mxGetData(prhs[1]);
    
    /* set data term */
    MyGraph->setData(DataCost);
    /* set the smoothness term */
    MyGraph->setSmoothness(SmoothnessCost);
    
        
    /* create a container for the pointer */
    const mwSize dims[2] = {1,0};
    plhs[0] = mxCreateNumericArray(1, /*(int*)*/dims, MATLAB_POINTER_TYPE, mxREAL);
    
    GraphHandle* gh;
    gh = (GraphHandle*) mxGetData(plhs[0]);
    *gh = (GraphHandle)(MyGraph);
}
