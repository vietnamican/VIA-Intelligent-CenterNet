#include "CenterFace.hpp"
#include "mat.h"

CenterFace::CenterFace(const std::string &bin_path, const std::string &param_path, int input_width, int input_length, float score_threshold_, float iou_threshold_, int topk_){
    topk = topk_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    in_w = input_width;
    in_h = input_length;
    centerFace.load_param(param_path.data());
    centerFace.load_model(bin_path.data());
}

CenterFace::~CenterFace() { centerFace.clear(); }
