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
                printf("%.4f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

bool compare(BoxInfo i, BoxInfo j){
    float score_i = i.score;
    float score_j = j.score;
    return score_i >= score_j;
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
    const float scaled[3] = {1.0/255, 1.0/255, 1.0/255};
    img.substract_mean_normalize(0, scaled);
    img.substract_mean_normalize(means_vals, norm_vals);
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
    sort(box_info.begin(), box_info.end(), compare);
    int box_num = box_info.size();
    vector<int> merged(box_num, 0);
    vector<float> areas(box_num, 0);
    
    FOR(i, 0, box_num){
        areas[i] = (box_info[i].x2 -box_info[i].x1 + 1) * (box_info[i].y2 - box_info[i].y1 + 1);
    }

    FOR(i, 0, box_num){
        if(merged[i]){
            continue;
        }
        FOR(j, i+1, box_num){
            if(merged[j]){
                continue;
            }

            float xx1 = max<float>(box_info[i].x1, box_info[j].x1);
            float yy1 = max<float>(box_info[i].y1, box_info[j].y1);
            float xx2 = min<float>(box_info[i].x2, box_info[j].x2);
            float yy2 = min<float>(box_info[i].y2, box_info[j].y2);
            float w = xx2 - xx1;
            float h = yy2 - yy1;
            w = std::max<float>(0, w);
            h = std::max<float>(0, h);

            float inter = w * h;
            float outer = areas[i] + areas[j] - inter;
            float overlap = inter / outer;
            if (overlap > iou_threshold || overlap / areas[i] > iou_threshold || overlap / areas[j] > iou_threshold){
                merged[j] = 1;
            }
        }
    }
    vector<BoxInfo> output;
    FOR(i, 0, box_num){
        if(!merged[i]){
            output.push_back(box_info[i]);
        }
    }
    box_info = output;
}



