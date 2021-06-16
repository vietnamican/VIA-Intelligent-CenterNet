#include <iostream>
#include <opencv2/opencv.hpp>
#include "CenterFace.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    cv::Mat cvimg = cv::imread("../sample.jpg", cv::IMREAD_COLOR);
    ncnn::Mat ncnnimg = ncnn::Mat::from_pixels(cvimg.data, ncnn::Mat::PIXEL_BGR2RGB, cvimg.cols, cvimg.rows);
    string bin_path = "../centernet.bin";
    string param_path = "../centernet.param";
    CenterFace centerface(bin_path, param_path, 240, 180, 1, 0.7);

    ncnn::Mat scores;
    ncnn::Mat off;
    ncnn::Mat wh;
    centerface.detect(ncnnimg, scores, off, wh);
    centerface.decode(scores, off, wh);
    // cout << scores;
    return 0;
}