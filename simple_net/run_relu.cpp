//
// Created by CuiQinPro on 2023/5/19.
//
// only for test and use
/**
g++ Function.cpp libNet.cpp run_relu.cpp -o reluNet  $(pkg-config --cflags --libs opencv4)
 */
#include "libNet.h"
#include <ctime>
using namespace std;
using namespace cv;
using namespace CuiQin;

int main(int args, char* argv[]) {
    // 设置层级数量
    vector<int> layer_neuron_num = { 26,50,50,2 }; // 26个特征 2个输出
    // 初始化网络和权重
    libNet net;
    net.initNet(layer_neuron_num);
    net.initWeights(0, 0., 0.01);
//    cv::Scalar s = Scalar(0.5); // 不能直接作为参数传进去
//    net.initBias(s);

    // 获得训练和测试数据
    Mat input, label, test_input, test_label;
    get_input_label("./data/input_label_0-1_train.xml", input, label, 1504);
    get_input_label("./data/input_label_0-1_test.xml", test_input, test_label, 294);
//    get_input_label("./data/input_label_0-9_1000.xml", input, label, 800);
//    get_input_label("./data/input_label_0-9_1000.xml", test_input, test_label, 200,800);

    // 设置参数
    float loss_threshold = 1;
    net.learning_rate = 0.0005;
    net.output_interval = 20;
    net.activation_function = "ReLU";

    int max_epoch = 100000;

    // 获取当前时间
    time_t t = time(nullptr);
    struct tm* now = localtime(&t);
    stringstream timeStr;
    timeStr<<now->tm_year+1900<<"_";
    timeStr<<now->tm_mon+1<<"_";
    timeStr<<now->tm_mday<<"_";
    timeStr<<now->tm_hour<<"_";
    timeStr<<now->tm_min<<"_";
    timeStr<<now->tm_sec;
    std::string log_path = "./data/relu_" + timeStr.str() + ".csv";
    net.log_path = log_path;

    // 训练然后绘图？
    net.train2(input, label, loss_threshold, max_epoch, false);
//    net.train1(input, label, loss_threshold);
    net.test(test_input, test_label);

    /**
     * todo 将训练过程中输出的epoch,loss输出到csv文件，方便进行绘图整理，同时研究最小
     */
    //Save the model
    net.save("./models/model_relu_1_0.0005_20_100000_{26_50_50_2}.xml");
    return 0;
}
