
#define BLOCK_SIZE 64

#define _DEBUG
#include "cutil.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_memory.h"
#include <sys/time.h>

#include "MirroredArray.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "graph.h"
#include <map>

struct MatrixEntry {
    //int index;
    float weight;
};

template<int pd>
__global__ static void createMatrix(const int f, const int w, const int h, 
				    const float *positions, 
				    MatrixEntry *matrix, int dimension,
				    float *VerticsValue, int *gridsize, int nVertics, int* mydims,
				    float *VerticsUnary, float *masks) 
{

    // 8x8 blocks    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int t = threadIdx.z + blockIdx.z * blockDim.z;
    const int idx = t*w*h + y*w + x;
    const bool outOfBounds = (t >= f) || (x >= w) || (y >= h);

    if (!outOfBounds) {
	    const float *myPosition = positions + idx*pd;
	    const float *masksPtr = masks + idx;
	    int myfloors[pd];
	    int mycells[pd];
	    for (int i = 0; i < pd; i++) {
			myfloors[i] = floorf(myPosition[i]);
		    mycells[i] = ceilf(myPosition[i]);//up
		    if (mycells[i]==myfloors[i])
		    	mycells[i]++;
	    }
	    for (int dim = 0; dim < dimension; ++dim){
			MatrixEntry r;
			r.weight = 1;
			int bin[pd];
			int tmpdim = dim;
			for (int i = pd-1; i >= 0; --i){
				if(tmpdim%2){//up
					bin[i] = 1;
					r.weight *= 1.0-(mycells[i]-myPosition[i]); //?? *= ??
				}else{
					bin[i] = 0;
					r.weight *= 1.0-(myPosition[i]-myfloors[i]); //?? *= ??
				}
				tmpdim /= 2;
			}
		    matrix[idx*dimension + dim] = r;

		    //bin: 100101
		    int Ver[6];
		    int idxVertic = 0;
		    for (int i = 0; i < pd; ++i){
		    	Ver[i] = myfloors[i]+bin[i];//first dim: [1,35]
		    	idxVertic += mydims[i]* (Ver[i]-1);
		    }
		    if (idxVertic>=nVertics) return;
		    atomicAdd(VerticsValue+idxVertic, 1*r.weight);
		    atomicAdd(VerticsUnary+idxVertic*2, masksPtr[0]*r.weight);
		    atomicAdd(VerticsUnary+idxVertic*2+1, (1-masksPtr[0])*r.weight);
		    if(idx==205766){
		    	
		    }
	    }
	}
}

template<int pd>
__global__ static void split(float* output, const int f, const int w, const int h, 
				const float *positions, 
				MatrixEntry* matrix, float* VerticsValue, int dimension,int nVertics, int* mydims) 
{

    // 8x8 blocks    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int t = threadIdx.z + blockIdx.z * blockDim.z;
    const int idx = t*w*h + y*w + x;
    const bool outOfBounds = (t >= f) || (x >= w) || (y >= h);
    if (!outOfBounds) {
	    const float *myPosition = positions + idx*pd;
	    float myfloors[pd];
	    for (int i = 0; i <= pd; i++) {
			myfloors[i] = floorf(myPosition[i]);//down
	    }
	    for (int dim = 0; dim < dimension; ++dim){
			int bin[pd];
			int tmpdim = dim;
			for (int i = pd-1; i >= 0; --i){
				if(tmpdim%2){//up
					bin[i] = 1;
				}else{
					bin[i] = 0;
				}
				tmpdim /= 2;
			}
		    float weight = matrix[idx*dimension + dim].weight;

		    //bin: 100101
		    int Ver[6];
		    int idxVertic = 0;
		    for (int i = 0; i < pd; ++i){
		    	Ver[i] = myfloors[i]+bin[i];//first dim: [1,35]
		    	idxVertic += mydims[i]* (Ver[i]-1);
		    }
		    if (idxVertic>=nVertics) return;
		    float value = VerticsValue[idxVertic*2];
		    output[idx]+=weight*value;

	    }
	}
}

/* Usage: 
 */
template<int pd>
void createAdjacencyMatrix(const int f, const int w, const int h,
	float* VerticsValue, float* pairwise, int *gridsize){
	// parameters
	float temporalWeight = 100000;
	float colorWeight = 0.03;
	float spatialWeight = 0.3;
	float minGraphWeight = 0.01;
	float pairwiseWeight[6] = {colorWeight,colorWeight,colorWeight,
		spatialWeight,spatialWeight,temporalWeight};
	int nVertics = 1;
	for (int i = 0; i < pd; ++i){
		nVertics *= gridsize[i];
	}
	int dims[pd];
    for (int i = 0; i < pd; ++i){
    	if (i==0) dims[i] = nVertics/gridsize[i];
    	else dims[i] = dims[i-1]/gridsize[i];
    }
	// pairwise
	for (int r = 0; r < gridsize[0]-1; ++r){
		for (int g = 0; g < gridsize[1]-1; ++g){
			for (int b = 0; b < gridsize[2]-1; ++b){
				for (int x = 0; x < gridsize[3]-1; ++x){
					for (int y = 0; y < gridsize[4]-1; ++y){
						for (int t = 0; t < gridsize[5]-1; ++t){
							int idxVertic = r*dims[0]+g*dims[1]+b*dims[2]+x*dims[3]+y*dims[4]+t*dims[5];
							for (int i = 0; i < pd; ++i){
								if (VerticsValue[idxVertic]){
									pairwise[idxVertic*6 + i] = pairwiseWeight[i] * VerticsValue[idxVertic]*VerticsValue[idxVertic+dims[i]];//Weight([rgbxyt],[(r+1)gbxyt])
									pairwise[idxVertic*6 + i] = max(pairwise[idxVertic*6 + i],minGraphWeight);
								}
							}

							// if (idxVertic==(32-1)*dims[0]+(14-1)*dims[1]+(13-1)*dims[2])
							// {
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 +0]);
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 + 1]);
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 + 2]);
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 + 3]);
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 + 4]);
							// 	printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 + 5]);
							// }
							// for (int tmp = 0; tmp < 6; ++tmp)
							// {
							// 	if (idxVertic+dims[tmp]==(32-1)*dims[0]+(14-1)*dims[1]+(13-1)*dims[2]){
							// 		printf("[pairwise 0]: %f\n", pairwise[idxVertic*6 +tmp]);
							// 	}
							
							// }

						}
					}
				}
			}
		}
	}
}

template<int pd>
void graphcut(float* unary, float* pairwise, int *gridsize, float* VerticsValue){
	float pairwiseWeight = 1;
	float unaryWeight = 50;

	int nVertics = 1;
	for (int i = 0; i < pd; ++i){
		nVertics *= gridsize[i];
	}
	int dims[6];
    for (int i = 0; i < pd; ++i){
    	if (i==0) dims[i] = nVertics/gridsize[i];
    	else dims[i] = dims[i-1]/gridsize[i];
    }

    std::map<int, int> idx2CoarseIdx;
    std::map<int, int> coarseIdx2Idx;
    int point_count = 0;
    for (int i = 0; i<nVertics; i++){
    	if (VerticsValue[i]){
    		idx2CoarseIdx[point_count] = i;
    		coarseIdx2Idx[i]=point_count;
    		point_count++;
    	}
    }

    typedef Graph<float,float,float> GraphType;
    GraphType *g = new GraphType(point_count, point_count*6);
    printf("point_count: %d\n", point_count);
    for (int i = 0; i<point_count; i++) {
        g -> add_node();
    }
    // unary
    for (int i = 0; i<point_count; i++) {
    	int idxVertic = idx2CoarseIdx[i];
        g -> add_tweights( i, unaryWeight*(unary[idxVertic*2]), unaryWeight*(unary[idxVertic*2+1]));
    }
    // pairwise
    int edge_count = 0;
    for (int rr = 0; rr < gridsize[0]-1; ++rr){
		for (int gg = 0; gg < gridsize[1]-1; ++gg){
			for (int bb = 0; bb < gridsize[2]-1; ++bb){
				for (int x = 0; x < gridsize[3]-1; ++x){
					for (int y = 0; y < gridsize[4]-1; ++y){
						for (int t = 0; t < gridsize[5]-1; ++t){
							int idxVertic = rr*dims[0]+gg*dims[1]+bb*dims[2]+
								x*dims[3]+y*dims[4]+t*dims[5];
							if (VerticsValue[idxVertic]){	
								edge_count++;			
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[0]], 
									pairwiseWeight * pairwise[idxVertic*6 + 0], 
									pairwiseWeight * pairwise[idxVertic*6 + 0] );
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[1]], 
									pairwiseWeight * pairwise[idxVertic*6 + 1], 
									pairwiseWeight * pairwise[idxVertic*6 + 1] );
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[2]], 
									pairwiseWeight * pairwise[idxVertic*6 + 2], 
									pairwiseWeight * pairwise[idxVertic*6 + 2] );
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[3]], 
									pairwiseWeight * pairwise[idxVertic*6 + 3], 
									pairwiseWeight * pairwise[idxVertic*6 + 3] );
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[4]], 
									pairwiseWeight * pairwise[idxVertic*6 + 4], 
									pairwiseWeight * pairwise[idxVertic*6 + 4] );
								g->add_edge( coarseIdx2Idx[idxVertic], coarseIdx2Idx[idxVertic+dims[5]], 
									pairwiseWeight * pairwise[idxVertic*6 + 5], 
									pairwiseWeight * pairwise[idxVertic*6 + 5] );
							}
						}
					}
				}
			}
		}
	}
	printf("edge_count: %d\n", edge_count);
	printf("cuting....\n");
	float flow = g -> maxflow();
	//result
	for (int i = 0; i<nVertics*2; i++){
    	unary[i] = 0;
    }
    int source_count =  0;
	for (int i = 0; i<point_count; i++) {
    	int idxVertic = idx2CoarseIdx[i];
    	if (g->what_segment(i) == GraphType::SOURCE) {
    		unary[idxVertic*2] = 1;
    		source_count++;
    	}
    }

    printf("source_count: %d\n", source_count);
    delete g;   
}

void checkMinMax(float* array, int n, char* name){
	float max = -9999.0, min = 9999.0;
	int nonzero_count = 0;
	for (int i = 0; i < n; ++i)
		if (array[i]){
			if (array[i]>max)
				max = array[i];
			if (array[i]<min)
				min = array[i];
			nonzero_count++;
		}
	printf("[check %s]: n: %d, nonzero_count: %d, min: %f, max: %f.\n", name, n, nonzero_count, min, max);
}

template<int pd, int vd>
void filter_(float *out, float *im, float *ref, int f, int w, int h, bool accurate, int *gridsize,
	float *masks) {    
	
	int n = f*w*h;
	int nVertics = 1;
	for (int i = 0; i < pd; ++i){
		nVertics *= gridsize[i];
	}
	int dimension = pow(2,pd);

	int mydims[6];
    for (int i = 0; i < pd; ++i){
    	if (i==0) mydims[i] = nVertics/gridsize[i];
    	else mydims[i] = mydims[i-1]/gridsize[i];
    	printf("%d\n", mydims[i]);
    }

	MirroredArray<float> values(im, n*vd);
    MirroredArray<float> positions(ref, n*pd);
    MirroredArray<MatrixEntry> matrix(n*dimension);//store weights:n*64
    float *Verticsf = new float[nVertics];
    for (int i = 0; i < nVertics; ++i)
    	Verticsf[0] = 0.0f;
    MirroredArray<float> VerticsValue(Verticsf,nVertics);//store vertics
    MirroredArray<int> gridsize_gpu(gridsize,pd);
    MirroredArray<int> mydims_gpu(mydims,pd);
    MirroredArray<float> mask_gpu(masks,n);
    float *Unaryf = new float[nVertics*2];
    for (int i = 0; i < nVertics*2; ++i)
    	Unaryf[0] = 0.0f;
    MirroredArray<float> VerticsUnary(Unaryf,nVertics*2);//store vertics

    dim3 blocks((w-1)/8+1, (h-1)/8+1, (f-1)/8+1);
    dim3 blockSize(8, 8, 8);
    timeval t[7];
    gettimeofday(t+0, NULL); 

    printf("%d\n", nVertics); 
    createMatrix<pd><<<blocks, blockSize>>>(f, w, h, positions.device, matrix.device, dimension, 
    	VerticsValue.device, gridsize_gpu.device, nVertics, mydims_gpu.device,
    	VerticsUnary.device, mask_gpu.device);
    CUT_CHECK_ERROR("Matrix creation failed\n");
    gettimeofday(t+1, NULL); 

	//------------------------ check ----------------------
    VerticsValue.deviceToHost();
    checkMinMax(VerticsValue.host,nVertics,"VerticsValue");
    //int idxVertic = (32-1)*mydims[0]+(14-1)*mydims[1]+(13-1)*mydims[2];
    //printf("[first point]: VerticsValue: %f\n", VerticsValue.host[idxVertic]);
    VerticsValue.hostToDevice();
    //-----------------------------------------------------

    //pairwise
	float* pairwise = new float[nVertics*6]; //np = 6
	for (int i = 0; i < nVertics*6; ++i)
		pairwise[i] = 0;
    VerticsValue.deviceToHost();
    createAdjacencyMatrix<pd>(f, w, h, VerticsValue.host, pairwise, gridsize);

    //------------------------ check ----------------------
    checkMinMax(pairwise,nVertics*6,"pairwise");
    //-----------------------------------------------------

    //unary
    VerticsUnary.deviceToHost();
    float* unary = VerticsUnary.host;

    //------------------------ check ----------------------
    checkMinMax(unary,nVertics*2,"unary");
    //-----------------------------------------------------

    //graphcut
    printf("start graphcut...\n");
    graphcut<pd>(unary, pairwise, gridsize, VerticsValue.host);
    gettimeofday(t+2, NULL);

    //------------------------ check ----------------------
    checkMinMax(unary,nVertics*2,"graphcut");
    //-----------------------------------------------------

    MirroredArray<float> output(out, n);
    VerticsUnary.hostToDevice();
    split<pd><<<blocks, blockSize>>>(output.device, f, w, h, positions.device, matrix.device,
    	VerticsUnary.device, dimension, nVertics, mydims_gpu.device);

    //------------------------ check ----------------------
    output.deviceToHost();
    checkMinMax(output.host,n,"split");
    output.hostToDevice();
    //-----------------------------------------------------
    gettimeofday(t+3, NULL);

    printf("%s: %3.3f ms\n", "Matrix", (t[1].tv_sec - t[0].tv_sec)*1000.0 + (t[1].tv_usec - t[0].tv_usec)/1000.0);
    printf("%s: %3.3f ms\n", "split", (t[2].tv_sec - t[1].tv_sec)*1000.0 + (t[2].tv_usec - t[1].tv_usec)/1000.0);
    printf("%s: %3.3f ms\n", "pairwiseMatrix", (t[3].tv_sec - t[2].tv_sec)*1000.0 + (t[3].tv_usec - t[2].tv_usec)/1000.0);

    printf("Total GPU memory usage: %f Gbytes\n", (float)GPU_MEMORY_ALLOCATION/1024.0/1024.0/1024.0);

    printf("[Test] output\n" );
    output.deviceToHost();
    for (int i = 0; i < 10; ++i)
    {
    	printf("[%d]: %f\n", i, out[i]);
    }
}

extern "C"
void filter(float *outputs, float *inputs, float *positions, int pd, int vd, int f, int w, int h, bool accurate, int *gridsize,
	float *masks){
	if (pd==6 && vd==3){
		filter_<6, 3>(outputs, inputs, positions, f, w, h, accurate, gridsize, masks);
	}
	else
		printf("Unsupported channel counts.\n");	    
}
