#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace cv;
using namespace std;

#define GET_ARRAY_LEN(array,len){len = (sizeof(array) / sizeof(array[0]));}

int DEBUG=1;

int main(int argc,char* argv[]){
    // parameters
    string vidFn = "./data/ducks/ducks.mp4";
    vector<string> maskFn;
    maskFn.push_back("./data/ducks/ducks01_0001_gt.ppm");
    maskFn.push_back("./data/ducks/ducks01_0100_gt.ppm");
    maskFn.push_back("./data/ducks/ducks01_0200_gt.ppm");
    maskFn.push_back("./data/ducks/ducks01_0300_gt.ppm");
    maskFn.push_back("./data/ducks/ducks01_0400_gt.ppm");
    vector<int> maskFrames;
    maskFrames.push_back(1-1);
    maskFrames.push_back(100-1);
    maskFrames.push_back(200-1);
    maskFrames.push_back(300-1);
    maskFrames.push_back(400-1);


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

    // Display parameters
    float threshold = .2;

    //If you run out of memory, try a smaller video
    float maxtime = 200;
    float scale = .25;
    float speedscale = 1; // not absolute right.

    float dimensionWeights_data[6] = {colorWeight, colorWeight, colorWeight, spatialWeight, spatialWeight, temporalWeight};
    vector<float> dimensionWeights(&dimensionWeights_data[0], &dimensionWeights_data[5]);
    int gridSize_data[6] = {intensityGridSize, chromaGridSize, chromaGridSize, spatialGridSize, spatialGridSize, temporalGridSize};
    vector<int> gridSize(&gridSize_data[0], &gridSize_data[5]);

    // load video
    VideoCapture capture(vidFn);
    if(!capture.isOpened())  
        cout<<"fail to open!"<<endl; 
    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    vector<Mat> vid;
    Mat frame, frame_dst;
    bool success = capture.read(frame);
    while (success){
        if (vid.size() >= maxtime)
            break;
        resize(frame, frame_dst, Size(0, 0), scale, scale, cv::INTER_LINEAR);
        vid.push_back(frame_dst);
        for (int iter = 0; iter < speedscale; iter++){
            success = capture.read(frame);
        }
    }
    // load mask
    for (int i = 0; i<maskFn.size(); i++){
        maskFrames[i] = ceil(double(maskFrames[i])/speedscale);
    }
    vector<Mat> mask;
    Mat mask_dst;
    for (int i = 0; i<maskFrames.size(); i++){
        if (maskFrames[i] >= maxtime)
            continue;
        mask_dst = imread(maskFn[i], 0); // 0-255, score(soft edge)
        resize(mask_dst, mask_dst, Size(0, 0), scale, scale, cv::INTER_LINEAR);
        mask.push_back(mask_dst);
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
       
    Mat img;
    string imgpath = argv[1];  
    img = imread(imgpath, 1);  
    imwrite("img.jpg", img);
    return 0;
}

