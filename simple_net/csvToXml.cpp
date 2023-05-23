//
// Created by CuiQinPro on 2023/5/20.
//

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
/**
g++ csvToXml.cpp -o dataset $(pkg-config --cflags --libs opencv4)
 * @return
 */
int csv2xml(string input, string output, int num) {
    /** opencv2.x
    CvMLData mlData;
    mlData.read_csv("NetDataTrain.csv");
    Mat data = cv::Mat(mlData.get_values(), true);
    */
    // 1. 读取数据
    Ptr<ml::TrainData> raw_data = ml::TrainData::loadFromCSV("./data/TrainData_upper_add.csv", 1, -2, 0); // 1，-2必须设定，否则就会默认最后一列为输出变量
    Mat data = raw_data->getSamples();
    cout << "Data have been read successfully!" << endl;
    //Mat double_data;
    //data.convertTo(double_data, CV_64F);
    cout<<data.cols<<","<<data.rows<<endl; // 打印数据信息
    Mat input_ = data(Rect(1, 0, 36, data.rows)).t(); // 转置
    Mat label_ = data(Rect(0, 0, 1, data.rows)); // 标签在第一个
    Mat target_(2, input_.cols, CV_32F, Scalar::all(0.));
    cout<<"input_:"<<input_.cols<<","<<input_.rows<<endl;
    cout<<"label_:"<<label_.cols<<","<<label_.rows<<endl;
    for (int i = 0; i < label_.rows; ++i)
    {
        float label_num = label_.at<float>(i, 0);
        //target_.at<float>(label_num, i) = 1.;
        target_.at<float>(label_num, i) = label_num;
    }
//    cout<<"target_:"<<target_.cols<<","<<target_.rows<<endl;
//    cout<<"i 0,1"<<endl;
//    for (int i=0; i<target_.cols;i++) {
//        cout<<i<<":";
//        for(int j=0;j<target_.rows;j++) {
//            cout<<target_.at<float>(j, i)<<",";
//        }
//        cout<<endl;
//    }
//    cout << "输入矩阵第1col,input:" << endl;
//    Mat col_0 = input_.col(0);
//    for (int i=0;i<26;i++) {
//        cout << col_0.at<float>(i)<<",";
//    }
//    cout<<endl;
//    cout << "输入矩阵第1col,target:" << endl;
//    float label0 = target_.at<float>(0, 0);
//    cout << label0 << endl;
//
    string filename = "./data/input_label_0-1_train_upper_add.xml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "input" << input_;
    fs << "target" << target_; // Write cv::Mat
    fs.release();


//    Ptr<ml::TrainData> raw_data = ml::TrainData::loadFromCSV("./NetDataTrain.csv", 1);
//    Mat data = raw_data->getSamples();
//    cout<< "Img:"<<data<<endl;
    return 0;
}