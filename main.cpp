#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arrayptr2mat.h"
using namespace cv;
using namespace std;

#define GET_ARRAY_LEN(array,len){len = (sizeof(array) / sizeof(array[0]));}

int DEBUG=1;

void lift(float *bilateralData, vector<Mat> vid, vector<int> gridSize, vector<int> Frames){
    int h = vid[0].rows;
    int w = vid[0].cols;
    int f = vid.size();
    cout<<"[In lift()] h = "<<h<<" w = "<<w<<" f = "<<f<<endl;
    float *pPtr = bilateralData;
    for (int t = 0; t < f; ++t){
		for (int y = 0; y < h; y++) {
            uchar* data= vid[t].ptr<uchar>(y);
			for (int x = 0; x < w; x++) {
				// resize: [0,255] => [1,35], [0,w] => [1,20], [0,h] => [1,20], [0,f] => [1,5]
				*pPtr++ = float(*data++);//*(gridsize[0]-1)+1;
			    *pPtr++ = float(*data++);//*(gridsize[1]-1)+1;
			    *pPtr++ = float(*data++);//*(gridsize[2]-1)+1;
			    *pPtr++ = float(x);///inputs.width*(gridsize[3]-1)+1;
			    *pPtr++ = float(y);///inputs.height*(gridsize[4]-1)+1;
			    *pPtr++ = float(Frames[t]);///inputs.frames*(gridsize[5]-1)+1;
			}
		}
    }
    // scale to grid size
    float eps = 0.004;
    float lBounds[6], uBounds[6];
    for (int i = 0; i < 6; ++i){
     	lBounds[i] = 9999.0;
     	uBounds[i] = -9999.0;
    }
    for (int i = 0; i < f*w*h; ++i){
    	for (int j = 0; j < 6; ++j){
    		float value = bilateralData[i*6+j];
    		if (value<lBounds[j])
    			lBounds[j] = value;
    		if (value>uBounds[j])
    			uBounds[j] = value;
    	}
    }
    for (int i = 0; i < 6; ++i){
     	lBounds[i] -= eps;
     	uBounds[i] += eps;
    }
    for (int i = 0; i < 6; ++i)
    {
    	printf("[check Bounds]: lBounds: %f uBounds: %f\n", lBounds[i],uBounds[i]);
    }
    // resize: [0,255] => [1,35], [0,w] => [1,20], [0,h] => [1,20], [0,f] => [1,5]
    for (int i = 0; i < f*w*h; ++i){
    	for (int j = 0; j < 6; ++j){
    		float value = bilateralData[i*6+j];
    		bilateralData[i*6+j] = (value - lBounds[j])/(uBounds[j]-lBounds[j])*(gridSize[j]-1)+1;
    	}
    }
    char out_file[100] = "temp/bilateralData.mat";
    int flag =  arrayptr2mat(bilateralData, f*w*h*6, 1, out_file);
}

// core function
vector<Mat> bilateralSpaceSegmentation(vector<Mat> vid, vector<Mat> mask, vector<int> maskFrames,
            vector<int> gridSize, vector<float> dimensionWeights, float unaryWeight, float pairwiseWeight){
    cout<<"[In bilateralSpaceSegmention()]"<<endl;
    vector<Mat> segmentation;
    // lift
    int h = vid[0].rows;
    int w = vid[0].cols;
    int f = vid.size();
    int fmask = maskFrames.size();
    float *bilateralData = new float[f*w*h*6];
    float* bilateralMask = new float[fmask*w*h*6];
    vector<int> Frames;
    for (int i = 0; i<f; i++)
        Frames.push_back(i);
    vector<Mat> vidmask;
    for (int i = 0; i<fmask; i++){
        vidmask.push_back(vid[maskFrames[i]]);
    }
    lift(bilateralData, vid, gridSize, Frames);
    lift(bilateralMask, vidmask, gridSize, maskFrames);
    // splatting


    delete bilateralMask;
    delete bilateralData;
    cout<<"[Out bilateralSpaceSegmention()]"<<endl;
    return segmentation;    
}

int main(int argc,char* argv[]){
    // load frameFns and maskFns
    cout<< "main start"<<endl;
    char tmp_file_name[128];
    
    vector<string> frameFns;
    for (int i = 191; i < 200; i++){
        sprintf(tmp_file_name, "./data/test2_small/frames/%05d.png", i);
        frameFns.push_back(string(tmp_file_name));
    }
    vector<string> maskFns;
    for (int i = 195; i < 200; i++){
        sprintf(tmp_file_name, "./data/test2_small/pred/%05d_pred.png", i);
        maskFns.push_back(string(tmp_file_name));
    }

    vector<int> maskFrames;
    for (int i = 195; i< 200; i++){
        maskFrames.push_back(i-191);
    }
    
    // Grid parameters
    int intensityGridSize = 35;
    int chromaGridSize = 15;
    int spatialGridSize = 20;
    int temporalGridSize = 5;

    // Graph Cut Parameters
    float pairwiseWeight = 1;
    float unaryWeight = 100000;
    float temporalWeight = 1e5;
    float intensityWeight = 0.05;
    float colorWeight = 0.03;
    float spatialWeight = 0.3;
    float minGraphWeight = 0.001;


    //If you run out of memory, try a smaller video
    float scale = .25;

    float dimensionWeights_data[6] = {colorWeight, colorWeight, colorWeight, spatialWeight, spatialWeight, temporalWeight};
    vector<float> dimensionWeights(&dimensionWeights_data[0], &dimensionWeights_data[5]);
    int gridSize_data[6] = {intensityGridSize, chromaGridSize, chromaGridSize, spatialGridSize, spatialGridSize, temporalGridSize};
    vector<int> gridSize(&gridSize_data[0], &gridSize_data[5]);

    // load video
    long totalFrameNumber = frameFns.size();
    vector<Mat> vid;
    Mat frame, frame_dst;
    for (int i = 0; i<totalFrameNumber; i++){    
        cout<<frameFns[i]<<endl;
        frame = imread(frameFns[i]);
        resize(frame, frame_dst, Size(0, 0), scale, scale, cv::INTER_LINEAR);
        vid.push_back(frame_dst.clone());
    }
    
    // load mask
    vector<Mat> mask;
    Mat mask_dst;
    int maskFrames_size = maskFrames.size();
    for (int i = 0; i<maskFrames_size; i++){
        mask_dst = imread(maskFns[i], 0); // 0-255, score(soft edge)
        resize(mask_dst, mask_dst, Size(0, 0), scale, scale, cv::INTER_LINEAR);
        mask.push_back(mask_dst);
    }

    if (maskFrames.size() != mask.size()){
        cout<<"Error! maskFrames and mask should have equal size!"<<endl;
    }
    if (DEBUG){
        cout<<"[TotalFrameNumber in video]: "<<totalFrameNumber<<endl; 
        cout<<"[LoadFrameNumber of video]: "<<vid.size()<<endl;
        cout<<"[LoadFrameSize]: "<<vid[0].rows<<" rows; "<<vid[0].cols<<" cols."<<endl;
        cout<<"[mask Size]:"<<mask.size()<<endl;
        cout<<"[maskFrames]: ";
        for (int i=0; i<maskFrames.size(); i++){
            cout<<maskFrames[i]<<" ";
        }
        cout<<endl;
    }
    
    vector<Mat> segmentation = bilateralSpaceSegmentation(vid,mask,maskFrames,gridSize,dimensionWeights,unaryWeight,pairwiseWeight);

    Mat img;
    string imgpath = argv[1];  
    img = imread(imgpath, 1);  
    imwrite("img.jpg", img);
    return 0;
}

