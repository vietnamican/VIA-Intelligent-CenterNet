#include "CenterFace.hpp"
#include "mat.h"

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

void CenterFace::detect(ncnn::Mat &img){
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return;
    }
    ncnn::Extractor ex = centerFace.create_extractor();
    ex.set_num_threads(4);
    ex.input("input.1", img);

    ncnn::Mat scores;
    ncnn::Mat off;
    ncnn::Mat wh;

    ex.extract("539", scores);
    ex.extract("540", off);
    ex.extract("541", wh);
    std::cout << scores.c << std::endl;
    std::cout << scores.h << std::endl;
    std::cout << scores.w << std::endl;
    // pretty_print(scores);
}
