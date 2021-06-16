#include <opencv2/opencv.hpp>
#include "CenterFace.hpp"
#include "mat.h"
#define FOR(i, j, k) for(int i=j;i<k;i++)
#define FOD(i, j, k) for(int i=k-1;i>=j;i--)

using namespace std;

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

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

void CenterFace::detect(ncnn::Mat &img, ncnn::Mat &scores, ncnn::Mat &off, ncnn::Mat &wh){
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return;
    }
    ncnn::Extractor ex = centerFace.create_extractor();
    ex.set_num_threads(4);
    ex.input("input.1", img);

    ex.extract("539", scores);
    ex.extract("540", off);
    ex.extract("541", wh);
}

void CenterFace::decode(ncnn::Mat &scores, ncnn::Mat off, ncnn::Mat wh){
    int c = scores.c;
    int h = scores.h;
    int w = scores.w;
    vector<BoxInfo> box_info;
    cv::Mat cvscore = cv::Mat::zeros(h, w, 0);
    const float* off_xs = off.channel(0);
    const float* off_ys = off.channel(1);
    const float* wh_xs = wh.channel(0);
    const float* wh_ys = wh.channel(1);
    FOR(q, 0, c){
        const float* ptr = scores.channel(q);
        FOR(y, 0, h){
            FOR(x, 0, w){
                if (ptr[y*w+x] > score_threshold){
                    BoxInfo res;
                    float off_x = off_xs[y*w+x];
                    float off_y = off_ys[y*w+x];
                    float wh_x = wh_xs[y*w+x];
                    float wh_y = wh_ys[y*w+x];
                    wh_x = exp(wh_x);
                    wh_y = exp(wh_y);
                    float c_x = x + off_x;
                    float c_y = y + off_y;
                    res.x1 = c_x - wh_x / 2;
                    res.x2 = c_x + wh_x / 2;
                    res.y1 = c_y - wh_y / 2;
                    res.y2 = c_y + wh_y / 2;
                    res.score = ptr[y*w+x];
                    box_info.push_back(res);
                    cvscore.at<uchar>(y, x) = 1;
                } else{
                    cvscore.at<uchar>(y, x) = 0;
                }
            }
        }
    }
    nms(box_info);
}

void CenterFace::nms(vector<BoxInfo> &box_info){
    
}