//
// Created by CuiQinPro on 2023/5/19.
//
// only for test and use
/**
g++ Function.cpp libNet.cpp run_sigmoid.cpp -o sigNet  $(pkg-config --cflags --libs opencv4)
 */
#include "libNet.h"
#include <ctime>
using namespace std;
using namespace cv;
using namespace CuiQin;

int main(int args, char* argv[]) {
    // 设置层级数量
    vector<int> layer_neuron_num = { 26,50,2 }; // 26个特征 2个输出
    // 初始化网络和权重
    libNet net;
    net.initNet(layer_neuron_num);
    net.initWeights(0, 0., 0.01);
    cv::Scalar s = Scalar(0.5); // 不能直接作为参数传进去
    net.initBias(s);

    std::string curType = "smote"; // no, smote , smote_add, upper, upper_add

    // 获得训练和测试数据
    Mat input, label, test_input, test_label;
    std::string train_data_path = "./data/input_label_0-1_train_"+curType+".xml";
    std::string test_data_path = "./data/input_label_0-1_test_"+curType+".xml";
    get_input_label(train_data_path, input, label, 1504);
    get_input_label(test_data_path, test_input, test_label, 294);

    std::cout<< "训练集路径: " <<train_data_path <<std::endl;
    std::cout<< "测试集路径: " <<test_data_path <<std::endl;

    // 设置参数
    float loss_threshold = 1;
    float learning_rate = 0.01;
    net.learning_rate = learning_rate;
    net.output_interval = 20;
    net.activation_function = "sigmoid";

    int max_epoch = 500000;

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
    std::string log_path = "./data/sigmoid_"+curType+"_"+ timeStr.str() + ".csv";
    net.log_path = log_path;

    // 训练然后绘图？
    net.train2(input, label, loss_threshold, max_epoch, false);
//    net.train1(input, label, loss_threshold);
    net.test(test_input, test_label);

    /**
     * todo 将训练过程中输出的epoch,loss输出到csv文件，方便进行绘图整理，同时研究最小
     */
    //Save the model
    // 激活函数 + loss + 学习率 + 最大次数 + 网络结构 + 当前数据集
    stringstream lossStrStream,learningStream,maxepochStream;
    lossStrStream<<loss_threshold;
    learningStream<<learning_rate;
    maxepochStream<<max_epoch;
    std::string loss = lossStrStream.str();
    std::string learning = learningStream.str();
    std::string maxepoch = maxepochStream.str();
    net.save("./models/model_sigmoid_"+loss+"_"+learning+"_"+maxepoch+"_{26_50_2}_"+curType+".xml");
    return 0;
}
