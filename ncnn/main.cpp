#include <iostream>
#include <opencv2/opencv.hpp>
#include "CenterFace.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    cv::Mat cvimg = cv::imread("../sample.jpeg", cv::IMREAD_COLOR);
    ncnn::Mat ncnnimg = ncnn::Mat::from_pixels(cvimg.data, ncnn::Mat::PIXEL_BGR2RGB, cvimg.cols, cvimg.rows);
    string bin_path = "../centernet.bin";
    string param_path = "../centernet.param";
    CenterFace centerface(bin_path, param_path, 320, 240, 1, 0.7);
    centerface.detect(ncnnimg);
    return 0;
}