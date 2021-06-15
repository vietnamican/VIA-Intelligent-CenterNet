#include <iostream>
#include <opencv2/opencv.hpp>
#include "CenterFace.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    cv::Mat img = cv::imread("sample.jpeg", cv::IMREAD_COLOR);
    string bin_path = "../centernet.bin";
    string param_path = "../centernet.param";
    CenterFace centerface(bin_path, param_path, 320, 240, 1, 0.7);
    return 0;
}