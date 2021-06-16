#ifndef CenterFace_hpp
#define CenterFace_hpp

#pragma once

#include "gpu.h"
#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

} BoxInfo;

class CenterFace{
public:
    CenterFace(const std::string &bin_path, const std::string &param_path, int input_width, int input_length, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1);
    ~CenterFace();
    void detect(ncnn::Mat &img);
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
    const float norm_vals[3] = {0.229, 0.224, 0.225};
};

#endif /* ncnn_UltraFace_hpp */