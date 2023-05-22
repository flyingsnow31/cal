//
// Created by CuiQinPro on 2023/5/19.
//

#ifndef CAL_NET_H
#define CAL_NET_H
#endif //CAL_NET_H
#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include "Function.h"
namespace CuiQin {
    // 定义网络结构
    class libNet {
    public:
        std::vector<int> layer_neuron_num; // 各层神经元数量
        std::string activation_function = "sigmoid"; // 默认激活函数
        int output_interval = 10; // 间隔几次迭代输出一次
        double learning_rate; // 学习率
        double accuracy = 0.; // 准确率
        std::vector<double> loss_vec; // 各层损失
        double fine_tune_factor = 1.0; // 微调
        std::string log_path;
    protected:
        std::vector<cv::Mat> layer; // 各层神经元，矩阵输入
        std::vector<cv::Mat> weights; // 神经元对应值
        std::vector<cv::Mat> bias; // 各层偏值
        std::vector<cv::Mat> delta_err; // 各层对应的德尔塔错误值

        cv::Mat output_error; // 输出
        cv::Mat target;
        cv::Mat board;
        float loss;

    public:
        libNet() {};
        libNet(std::vector<int> layer_neuron_num, double learning_rate, std::string activation_function = "sigmoid", int output_interval=20, double fine_tune_factor=1.0, std::string log_path="./output_log.csv");
        ~libNet() {};
        // 初始化
        void initNet(std::vector<int> layer_neuron_num_); // 初始化网络
        void initWeights(int type = 0, double a = 0., double b = 0.1); // 初始化权重
        void initBias(cv::Scalar& bias); // 初始化偏置
        void forward(); // 向前传播，线性运算和非线性激活，同时计算误差
        void backward(); // 向后传播，updateWeights()更新权值
        // 训练与预测
        void train1(cv::Mat input, cv::Mat target, float accuracy_threshold);
        void train2(cv::Mat input, cv::Mat target_, float loss_threshold, int epochs, bool draw_loss_curve = false); // 训练
        void test(cv::Mat &input, cv::Mat &target_); // 测试
        int predict_one(cv::Mat &input); // 预测一条
        std::vector<int> predict(cv::Mat &input); /// 预测
        // 模型保存与加载
        void save(std::string filename); // 模型保存
        void load(std::string filename); // 模型加载
    protected:
        void initWeight(cv::Mat &dst, int type, double a, double b); // 初始各层权重
        cv::Mat activationFunction(cv::Mat &x, std::string func_type); // 激活函数使用
        void deltaError(); // 计算损失
        void updateWeights(); // 更新权重
    };

    //Get sample_number samples in XML file,from the start column.
    void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);

    // Draw loss curve
    void draw_curve(cv::Mat& board, std::vector<double> points);
} // CuiQin

