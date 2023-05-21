//
// Created by CuiQinPro on 2023/5/20.
//

#ifndef DECISIONTREE_FUNCTION_H
#define DECISIONTREE_FUNCTION_H
#endif //DECISIONTREE_FUNCTION_H
#include <opencv2/opencv.hpp>
#include <iostream>
namespace CuiQin {
    //sigmoid function
    cv::Mat sigmoid(cv::Mat &x);

    //Tanh function
    cv::Mat tanh(cv::Mat &x);

    //ReLU function
    cv::Mat ReLU(cv::Mat &x);

    //Derivative function
    cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type);

    //Objective function
    void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss);
} // CuiQin
