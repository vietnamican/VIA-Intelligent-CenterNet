#ifndef CenterFace_hpp
#define CenterFace_hpp

#pragma once

#include "gpu.h"
#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    string toString(){
        return to_string(x1) + " " + to_string(y1) + " " + to_string(x2) + " " + to_string(y2) + " " + to_string(score);
    }

} BoxInfo;

class CenterFace{
public:
    CenterFace(const std::string &bin_path, const std::string &param_path, int input_width, int input_length, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1);
    ~CenterFace();
    void detect(ncnn::Mat &img, ncnn::Mat &scores, ncnn::Mat &off, ncnn::Mat &wh);
    void decode(ncnn::Mat &scores, ncnn::Mat off, ncnn::Mat wh);
    void nms(std::vector<BoxInfo> &box_info);
    // void decode();
private:
    ncnn::Net centerFace;
    int image_height;
    int image_width;
    int topk;
    float score_threshold;
    float iou_threshold;
    
    int in_w;
    int in_h;

    const float means_vals[3] = {0.485, 0.456, 0.406};
    const float norm_vals[3] = {1/0.229, 1/0.224, 1/0.225};
};

#endif /* ncnn_UltraFace_hpp */