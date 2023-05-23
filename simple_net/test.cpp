//
// Created by CuiQinPro on 2023/5/22.
//

#include "libNet.h"

using namespace std;
using namespace cv;
using namespace CuiQin;

int main() {
    // 测试模型准确率
    //Get test samples and the label is 0--1
    Mat test_input, test_label;
    get_input_label("data/input_label_0-1_test_upper.xml", test_input, test_label, 294);

    //convert label from 0---1 to -1---1,cause tanh function range is [-1,1]
//    test_label = 2 * test_label - 1;

    //Load the trained net and test.
    libNet net;
    net.load("models/model_sigmoid_1_0.01_500000_{26_50_2}_upper.xml");
    net.test(test_input, test_label);

    getchar();
    return 0;
}