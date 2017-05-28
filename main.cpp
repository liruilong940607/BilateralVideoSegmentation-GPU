#include <stdio.h>
#include "macros.h"
#include "Image.h"
#include <fstream>  
#include <string> 
#include <vector>
#include "png_file.h"
#include "jpg_file.h"

using namespace std;
bool endswith(const char *str, const char *suffix) {
    return (strcmp(str + strlen(str) - strlen(suffix), suffix) == 0);
}

Image load_images(vector<string> names){
	if (names.size()<1){
		printf("Error! Name list is empty!\n" );
	}
	Image temp;
    if (endswith(names[0].c_str(), ".png")) {
		temp = PNG::load(names[0].c_str());
    } else if (endswith(names[0].c_str(), ".jpg")) {
		temp = JPG::load(names[0].c_str());
    } else {
		printf("Image is neither .png or .jpg file\n");
    } 
    
    int width = temp.width;
	int height = temp.height;
	int channels = temp.channels;
	int frames = names.size();

	Image result(frames,width,height,channels);
	for (int t = 0; t < frames; ++t)
	{
		Image input;
	    if (endswith(names[t].c_str(), ".png")) {
			input = PNG::load(names[t].c_str());
	    } else if (endswith(names[t].c_str(), ".jpg")) {
			input = JPG::load(names[t].c_str());
	    } else {
			printf("Image is neither .png or .jpg file\n");
	    } 
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; ++c)
				{
					result(t,x,y)[c] = input(x, y)[c];
				}
			}
		}
	}
	return result;
}

void save_images(vector<string> names, Image images){
	if (names.size()<1){
		printf("Error! Name list is empty!\n" );
	}

	int width = images.width;
	int height = images.height;
	int channels = images.channels;
	int frames = images.frames;

	for (int t = 0; t < frames; ++t)
	{
		Image output(1, width, height, channels);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; ++c)
				{
					output(x,y)[c] = images(t, x, y)[c];
				}
			}
		}
		if (endswith(names[t].c_str(), ".png")) {
		PNG::save(output, names[t].c_str());
	    } else if (endswith(names[t].c_str(), ".jpg")) {
		JPG::save(output, names[t].c_str(), 99);
	    } else {
		printf("Output image is neither .png or .jpg file\n");
	    }
	}
}

extern "C" void filter(float *outputs, float *inputs, float *positions, 
	int pd, int vd, int f, int w, int h, bool accurate, int *gridsize,
	float *masks);

//useage: ./main list_f.txt list_p.txt ./data/segfine
int main(int argc, char **argv){

	//=======  load image data  ==========
	string input_dir = "./data/frames/";
	string output_dir = "./data/segfine/";
	vector<string> input_names, output_names;
	string line_f; 
	ifstream myfile_f(argv[1]);
	if (!myfile_f.is_open()){  
		printf("Failed to load list_f file.\n");
		return 1;
	}
	while(getline(myfile_f,line_f)){  
		input_names.push_back(input_dir+line_f);
		output_names.push_back(output_dir+line_f);
	}
	Image inputs =  load_images(input_names);
	myfile_f.close();
	// printf("inputs.frames: %d\n", inputs.frames);
	// printf("inputs.width: %d\n", inputs.width);
	// printf("inputs.height: %d\n", inputs.height);

	//=======  load pred data  ==========
	string pred_dir = "./data/pred/";
	vector<string> pred_names;
	string line_p; 
	ifstream myfile_p(argv[2]);
	if (!myfile_p.is_open()){  
		printf("Failed to load list_f file.\n");
		return 1;
	}
	while(getline(myfile_p,line_p)){  
		pred_names.push_back(pred_dir+line_p);
	}
	Image preds =  load_images(pred_names);
	myfile_p.close();

	//========= params  ==============
	int gridsize[] = {35, 15, 15, 20, 20, 5};//35

    float *positions = new float[inputs.frames*inputs.width*inputs.height*6];
    float *pPtr = positions;
    
	for (int t = 0; t < inputs.frames; ++t){
		for (int y = 0; y < inputs.height; y++) {
			for (int x = 0; x < inputs.width; x++) {
				// resize: [0,255] => [1,35], [0,w] => [1,20], [0,h] => [1,20], [0,f] => [1,5]
				*pPtr++ = float(inputs(t, x, y)[0]);//*(gridsize[0]-1)+1;
			    *pPtr++ = float(inputs(t, x, y)[1]);//*(gridsize[1]-1)+1;
			    *pPtr++ = float(inputs(t, x, y)[2]);//*(gridsize[2]-1)+1;
			    *pPtr++ = float(x);///inputs.width*(gridsize[3]-1)+1;
			    *pPtr++ = float(y);///inputs.height*(gridsize[4]-1)+1;
			    *pPtr++ = float(t);///inputs.frames*(gridsize[5]-1)+1;
			}
		}
	}
	float eps = 0.004;
    float lBounds[6], uBounds[6];
    for (int i = 0; i < 6; ++i){
     	lBounds[i] = 9999.0;
     	uBounds[i] = -9999.0;
    }
    for (int i = 0; i < inputs.frames*inputs.width*inputs.height; ++i){
    	for (int j = 0; j < 6; ++j){
    		float value = positions[i*6+j];
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
    for (int i = 0; i < inputs.frames*inputs.width*inputs.height; ++i){
    	for (int j = 0; j < 6; ++j){
    		float value = positions[i*6+j];
    		positions[i*6+j] = (value - lBounds[j])/(uBounds[j]-lBounds[j])*(gridsize[j]-1)+1;
    	}
    }
	for (int i = 0; i < 3; ++i)
		printf("[input check]: %f\n", inputs(0,0,0)[i]);
	for (int i = 0; i < 6; ++i)
		printf("[positions check]: %f\n", positions[i]);

	
	// Filter the input
	printf("Calling filter...\n");
	float *out = new float[inputs.frames*inputs.width*inputs.height];
    float *outPtr = out;
	for (int t = 0; t < inputs.frames*inputs.width*inputs.height; ++t){
		*outPtr++ = 0.0f;
	}
    filter(out,inputs(0, 0, 0),
    		positions, 6, 3, 
    		inputs.frames, inputs.width, inputs.height, true,
    		gridsize,preds(0, 0, 0));

    printf("Saving output...\n");

    // Save the result
    float max = -9999.0f;
    int index = 0;
    for (int t = 0; t < inputs.frames*inputs.width*inputs.height; ++t){
		if (out[t]>max){
			max=out[t];
			index = t;
		}
	}
	printf("MAX: %f  MAXINDEX: %d\n", max, index);
	for (int t = 0; t < inputs.frames*inputs.width*inputs.height; ++t){
		if(out[t]>1)
			out[t] = 1;
	}
    Image outputs(inputs.frames, inputs.width, inputs.height, 1, out);
 	save_images(output_names,outputs);

	return 0;
}