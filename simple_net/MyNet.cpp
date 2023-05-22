//
// Created by CuiQinPro on 2023/5/21.
//
#include "libNet.h"
#include <ctime>
using namespace std;
using namespace cv;
using namespace CuiQin;

// 定制数据处理函数
pair<Mat, Mat> get_data(string data_path, int features_nums) {
    // 1. 读取数据
    Ptr<ml::TrainData> raw_data = ml::TrainData::loadFromCSV(data_path, 1, -2, 0); // 1，-2必须设定，否则就会默认最后一列为输出变量
    Mat data = raw_data->getSamples();
    Mat input_ = data(Rect(1, 0, features_nums, data.rows)).t(); // 转置
    Mat label_ = data(Rect(0, 0, 1, data.rows)); // 标签在第一个
    Mat target_(2, input_.cols, CV_32F, Scalar::all(0.));
    for (int i = 0; i < label_.rows; ++i)
    {
        float label_num = label_.at<float>(i, 0);
        target_.at<float>(label_num, i) = label_num;
    }
    return make_pair(input_, target_);
}

// 初始化网络
libNet BuildNet(vector<int> layer_neuron_num,  // 网络定义
             double w,                      // 权重
             double bias,                   // 偏置
             double learning_rate,          // 学习率
             double loss_threshold,         // 损失阈值
             string activate_funtion,       // 损失函数
             int output_interval,           // 日志精度
             int max_epoch,                 // 最大训练轮次
             string log_path,               // 日志路径
             string save_path               // 模型保存路径
             ) {
    // 初始化网络和权重
    libNet net;
    net.initNet(layer_neuron_num);
    net.initWeights(0, 0., w);
    cv::Scalar s = Scalar(bias); // 不能直接作为参数传进去
    net.initBias(s);

    // 设置参数
    net.learning_rate = learning_rate;
    net.output_interval = output_interval;
    net.activation_function = activate_funtion;
    net.log_path = log_path;
    return net;
}
int TrainNet(libNet net,
             Mat input,                     // 输入训练数据
             Mat label                      // 输入训练标签
             ) {
    // 训练然后绘图？
    net.train2(input, label, loss_threshold, max_epoch, false);
    return 0;
}

int TestNet(libNet net,
            Mat test_input,                 // 输入测试标签
            Mat test_label                  // 输入测试标签
            ) {
    net.test(test_input, test_label);
    return 0;
}

int SaveNet(libNet net) {
    net.save(save_path);
    reuturn 0;
}