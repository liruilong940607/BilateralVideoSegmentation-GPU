#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "graph.h"
//#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <ctime>
clock_t  clockBegin, clockEnd;
void PrintfContainerElapseTime(char *pszContainerName, char *pszOperator, long lElapsetime)
{
    printf("%s - %s cost  %ld ms\n", pszContainerName, pszOperator, lElapsetime);
}
//          SOURCE
//        /       \
//      1/         \2
//      /      3    \
//    node0 -----> node1
//      |   <-----   |
//      |      4     |
//      \            /
//      5\          /6
//        \        /
//           SINK
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

#ifndef WIN32

#define fscanf_s fscanf

inline void fopen_s(FILE **file, const char *name, const char *mode)
{
    *file = fopen(name, mode);
}
#endif


void loadMiddleburyMRFData(const std::string &filename, int *&dataCostArray, int *&hCueTransposed, int *&vCue, int &width, int &height, int &nLabels)
{
    FILE *fp;
    fopen_s(&fp, filename.c_str(),"rb");
    
    fscanf_s(fp,"%d %d %d",&width,&height,&nLabels);
    
    int i, n, x, y;
    int gt;
    
    for (i = 0; i < width * height; i++)
    {
        fscanf_s(fp,"%d",&gt);
    }
    
    dataCostArray = (int *) malloc(width * height * nLabels * sizeof(int));
    n = 0;
    int v;
    
    for (int c=0; c < nLabels; c++)
    {
        for (i = 0; i < width * height; i++)
        {
            fscanf_s(fp,"%d",&v);
            dataCostArray[n++] = v;
        }
    }
    
    hCueTransposed = (int *) malloc(width * height * sizeof(int));
    vCue = (int *) malloc(width * height * sizeof(int));
    
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width-1; x++)
        {
            fscanf_s(fp,"%d",&v);
            hCueTransposed[x*height+y] = v;
        }
        
        hCueTransposed[(width-1)*height+y] = 0;
    }
    
    for (y = 0; y < height-1; y++)
    {
        for (x = 0; x < width; x++)
        {
            fscanf_s(fp,"%d",&v);
            vCue[y*width+x] = v;
        }
    }
    
    for (x = 0; x < width; x++)
    {
        vCue[(height-1)*width+x] = 0;
    }
    
    fclose(fp);
    
}



int main()
{
    
    int width, height, nLabels;
    int *hCue, *vCue, *dataCostArray;
    int file_errors = 0;
    std::string sFilename ="person.txt" ;
    std::ifstream infile(sFilename.data(), std::ifstream::in);
    
    if (infile.good())
    {
        std::cout << "imageSegmentationNPP opened: <" <<sFilename << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }

    loadMiddleburyMRFData(sFilename, dataCostArray, hCue, vCue, width, height, nLabels);
    typedef Graph<int,int,int> GraphType;
    GraphType *g = new GraphType(/*estimated # of nodes*/ width*height, /*estimated # of edges*/ width*height*2);
    for (int i = 0; i<width*height; i++) {
        g -> add_node();
    }
    for (int i = 0; i<width*height; i++) {
        g -> add_tweights( i,   /* capacities */  dataCostArray[i], dataCostArray[width*height+i] );
    }
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width-1; x++)
        {
            g -> add_edge( y*width+x, y*width+x+1,    /* capacities */  hCue[x*height+y], hCue[x*height+y] );
        }
    }
    for (int y = 0; y < height-1; y++)
    {
        for (int x = 0; x < width; x++)
        {
            g -> add_edge( y*width+x, (y+1)*width+x,    /* capacities */  vCue[y*width+x], vCue[y*width+x] );
        }
    }

    clockBegin = clock();
    int flow = g -> maxflow();
    clockEnd = clock();
    PrintfContainerElapseTime("graphcut", "maxflow", clockEnd - clockBegin);
    
    printf("node count: %d\n", width*height);
    printf("edge count: %d\n", width*(height-1)+height*(width-1));
    printf("Flow = %d\n", flow);
    printf("Minimum cut:\n");
    if (g->what_segment(0) == GraphType::SOURCE)
        printf("node0 is in the SOURCE set\n");
    else
        printf("node0 is in the SINK set\n");
    if (g->what_segment(1) == GraphType::SOURCE)
        printf("node1 is in the SOURCE set\n");
    else
        printf("node1 is in the SINK set\n");
    
#ifdef USE_OPENCV
    printf("use opencv to write\n");
    cv::Mat out(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            if (g->what_segment(i*width+j) == GraphType::SOURCE) {
                out.at<uchar>(i,j)=0;
            }else{
                out.at<uchar>(i,j)=255;
            }
        }
    }
    cv::imwrite( "/Users/dalong/Desktop/Graphcut/Graphcut/out.png", out );
#endif
    
    delete g;
    
    return 0;
}
